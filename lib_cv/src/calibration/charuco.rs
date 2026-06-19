use calib_targets::{
    GridCoords,
    aruco::{MarkerCell, MarkerDetection, Matcher, ScanDecodeConfig, scan_decode_markers_in_cells},
    charuco::CharucoBoard,
    core::{GrayImageView, Homography, homography_from_4pt},
};
use image::GrayImage;
use imageproc::{
    contours::find_contours, contrast::adaptive_threshold, geometry::approximate_polygon_dp,
};
use log::debug;
use rayon::prelude::*;
use std::collections::HashMap;

use nalgebra::Point2;

/// Quad с ассоциированным контуром
#[derive(Debug, Clone)]
pub struct QuadWithContour {
    pub corners: [Point2<f32>; 4],
    pub contour: Vec<Point2<i32>>,
}

// После find_marker_quads: удалить quads с почти совпадающими центрами
fn dedup_quads(quads: &mut Vec<QuadWithContour>, min_dist: f32) {
    let mut kept = Vec::new();
    for quad in quads.drain(..) {
        let center = quad
            .corners
            .iter()
            .fold(Point2::origin(), |a, p| a + p.coords)
            / 4.0;
        if kept.iter().all(|k: &QuadWithContour| {
            let kc = k.corners.iter().fold(Point2::origin(), |a, p| a + p.coords) / 4.0;
            (center - kc).norm() >= min_dist
        }) {
            kept.push(quad);
        }
    }
    *quads = kept;
}

/// Билинейная интерполяция значения пикселя в субпиксельных координатах.
fn interpolate_pixel(gray: &GrayImage, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let wx = x - x0 as f32;
    let wy = y - y0 as f32;

    let x0 = x0.clamp(0, gray.width() as i32 - 1) as u32;
    let y0 = y0.clamp(0, gray.height() as i32 - 1) as u32;
    let x1 = x1.clamp(0, gray.width() as i32 - 1) as u32;
    let y1 = y1.clamp(0, gray.height() as i32 - 1) as u32;

    let i00 = gray.get_pixel(x0, y0).0[0] as f32;
    let i01 = gray.get_pixel(x0, y1).0[0] as f32;
    let i10 = gray.get_pixel(x1, y0).0[0] as f32;
    let i11 = gray.get_pixel(x1, y1).0[0] as f32;

    let i0 = i00 * (1.0 - wy) + i01 * wy;
    let i1 = i10 * (1.0 - wy) + i11 * wy;

    i0 * (1.0 - wx) + i1 * wx
}

/// Sub-pixel уточнение угла по алгоритму OpenCV cornerSubPix.
/// Решает систему уравнений на основе градиентов с Гауссовым взвешиванием.
fn refine_corner(
    gray: &GrayImage,
    pt: &mut Point2<f32>,
    win_size: u32,
    zero_zone: i32,
    max_iters: usize,
    eps: f32,
) {
    let win_w = (win_size * 2 + 1) as usize;
    let win_h = win_w;
    let win = win_size as i32;

    // Создаём Гауссову маску
    let mut mask: Vec<f32> = vec![0.0; win_w * win_h];
    for i in 0..win_h {
        let y = (i as f32 - win as f32) / win as f32;
        let vy = (-y * y).exp();
        for j in 0..win_w {
            let x = (j as f32 - win as f32) / win as f32;
            mask[i * win_w + j] = vy * (-x * x).exp();
        }
    }

    // Применяем zero_zone (мертвую зону в центре)
    if zero_zone >= 0 {
        let zz = zero_zone as usize;
        for i in (win as usize - zz)..=(win as usize + zz) {
            for j in (win as usize - zz)..=(win as usize + zz) {
                if i < win_h && j < win_w {
                    mask[i * win_w + j] = 0.0;
                }
            }
        }
    }

    let c_initial = *pt;
    let mut c_current = *pt;

    for _ in 0..max_iters {
        // Буфер для субпиксельной интерполяции (окно + 2 для градиентов)
        let buf_w = win_w + 2;
        let buf_h = win_h + 2;
        let mut subpix_buf: Vec<f32> = vec![0.0; buf_w * buf_h];

        // Заполняем буфер через билинейную интерполяцию
        for i in 0..buf_h {
            for j in 0..buf_w {
                let y = c_current.y + (i as f32 - win as f32 - 1.0);
                let x = c_current.x + (j as f32 - win as f32 - 1.0);
                subpix_buf[i * buf_w + j] = interpolate_pixel(gray, x, y);
            }
        }

        // Вычисляем градиенты и накапливаем систему уравнений
        let mut a = 0.0f64; // Σ(gx² * mask)
        let mut b = 0.0f64; // Σ(gx*gy * mask)
        let mut c = 0.0f64; // Σ(gy² * mask)
        let mut bb1 = 0.0f64;
        let mut bb2 = 0.0f64;

        for i in 0..win_h {
            let py = i as f32 - win as f32;
            for j in 0..win_w {
                let px = j as f32 - win as f32;
                let idx = (i + 1) * buf_w + (j + 1);

                // Центральные разности для градиентов
                let tgx = subpix_buf[idx + 1] - subpix_buf[idx - 1];
                let tgy = subpix_buf[idx + buf_w] - subpix_buf[idx - buf_w];

                let m = mask[i * win_w + j] as f64;
                let gxx = (tgx * tgx) as f64 * m;
                let gxy = (tgx * tgy) as f64 * m;
                let gyy = (tgy * tgy) as f64 * m;

                a += gxx;
                b += gxy;
                c += gyy;

                bb1 += gxx * px as f64 + gxy * py as f64;
                bb2 += gxy * px as f64 + gyy * py as f64;
            }
        }

        // Решаем систему 2×2
        let det = a * c - b * b;
        if det.abs() <= f64::EPSILON * f64::EPSILON {
            break;
        }

        let scale = 1.0 / det;
        let c_new = Point2::new(
            (c_current.x as f64 + c * scale * bb1 - b * scale * bb2) as f32,
            (c_current.y as f64 - b * scale * bb1 + a * scale * bb2) as f32,
        );

        // Проверяем сходимость
        let err = (c_new.x - c_current.x).powi(2) + (c_new.y - c_current.y).powi(2);

        // Проверяем выход за границы изображения
        if c_new.x < 0.0
            || c_new.x >= gray.width() as f32
            || c_new.y < 0.0
            || c_new.y >= gray.height() as f32
        {
            break;
        }

        c_current = c_new;

        if err < eps * eps {
            break;
        }
    }

    // Проверяем, не ушли ли слишком далеко от начальной точки
    if (c_current.x - c_initial.x).abs() > win as f32
        || (c_current.y - c_initial.y).abs() > win as f32
    {
        // Плохая сходимость, возвращаем начальную точку
        return;
    }

    *pt = c_current;
}

pub fn detect_aruco_markers(
    img: &GrayImage,
    dict: &calib_targets::aruco::Dictionary,
) -> Vec<MarkerDetection> {
    // Находим четырехугольники
    let mut quads = find_marker_quads(img);
    debug!(
        "detect_aruco_markers: image={}x{}, quads_found={}",
        img.width(),
        img.height(),
        quads.len()
    );

    if quads.is_empty() {
        return Vec::new();
    }

    dedup_quads(&mut quads, 5.0);

    // Уточнение углов: сначала через линии контура, потом cornerSubPix
    for quad in &mut quads {
        // 1. Уточнение через аппроксимацию линий (использует весь контур)
        refine_corner_lines(&mut quad.corners, &quad.contour);

        // 2. Уточнение через cornerSubPix (локальное уточнение)
        for corner in quad.corners.iter_mut() {
            refine_corner(img, corner, 5, -1, 30, 0.1);
        }
    }

    let cells: Vec<MarkerCell> = quads
        .iter()
        .map(|q| MarkerCell {
            gc: GridCoords { i: 0, j: 0 },
            corners_img: q.corners,
        })
        .collect();

    let view = GrayImageView {
        width: img.width() as usize,
        height: img.height() as usize,
        data: img.as_raw(),
    };
    let matcher = Matcher::new(*dict, dict.max_correction_bits());

    let mut perimeters: Vec<f32> = quads
        .iter()
        .map(|q| {
            let d01 = (q.corners[1] - q.corners[0]).norm();
            let d12 = (q.corners[2] - q.corners[1]).norm();
            let d23 = (q.corners[3] - q.corners[2]).norm();
            let d30 = (q.corners[0] - q.corners[3]).norm();
            d01 + d12 + d23 + d30
        })
        .collect();
    perimeters.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let px_per_square = perimeters[perimeters.len() / 2] / 4.0;

    // Фильтруем слишком мелкие quads
    let min_perimeter = px_per_square * 4.0 * 0.6;
    let filtered_cells: Vec<MarkerCell> = cells
        .into_iter()
        .filter(|cell| {
            let q = &cell.corners_img;
            let d01 = (q[1] - q[0]).norm();
            let d12 = (q[2] - q[1]).norm();
            let d23 = (q[3] - q[2]).norm();
            let d30 = (q[0] - q[3]).norm();
            (d01 + d12 + d23 + d30) >= min_perimeter
        })
        .collect();
    debug!(
        "detect_aruco_markers: {} cells after size filter (min_perimeter={:.0})",
        filtered_cells.len(),
        min_perimeter
    );

    if filtered_cells.is_empty() {
        return Vec::new();
    }

    let cfg = ScanDecodeConfig {
        border_bits: 1,
        inset_frac: 0.04,
        marker_size_rel: 1.0,
        min_border_score: 0.5,
        dedup_by_id: true,
        multi_threshold: true,
    };

    let markers: Vec<_> = filtered_cells
        .par_chunks(32)
        .flat_map(|chunk| scan_decode_markers_in_cells(&view, chunk, px_per_square, &cfg, &matcher))
        .collect();
    markers
}

pub fn build_marker_homographies(
    board: &CharucoBoard,
    markers: &[MarkerDetection],
) -> HashMap<u32, Homography> {
    let spec = board.spec();
    let cell_size = spec.cell_size;
    let marker_size = cell_size * spec.marker_size_rel;
    let half = marker_size / 2.0;

    let mut transforms = HashMap::new();

    for marker in markers {
        let (sx, sy) = match board.marker_cell(marker.id as i32) {
            Some(cell) => cell,
            None => {
                debug!(
                    "build_marker_homographies: marker id={} not on board",
                    marker.id
                );
                continue;
            }
        };

        let cx = (sx as f32 + 0.5) * cell_size;
        let cy = (sy as f32 + 0.5) * cell_size;
        let src: [Point2<f32>; 4] = [
            Point2::new(cx - half, cy - half), // TL
            Point2::new(cx + half, cy - half), // TR
            Point2::new(cx + half, cy + half), // BR
            Point2::new(cx - half, cy + half), // BL
        ];
        let dst = match marker.corners_img {
            Some(img_corners) => img_corners,
            None => {
                debug!(
                    "build_marker_homographies: marker id={} has no corners_img",
                    marker.id
                );
                continue;
            }
        };

        if let Some(h) = homography_from_4pt(&src, &dst) {
            let det = h.h.determinant().abs();
            if det > 1e-6 {
                transforms.insert(marker.id, h);
            } else {
                debug!(
                    "build_marker_homographies: marker id={} singular homography (det={:.2e})",
                    marker.id, det
                );
            }
        } else {
            debug!(
                "build_marker_homographies: marker id={} homography_from_4pt returned None",
                marker.id
            );
        }
    }
    debug!(
        "build_marker_homographies: {} valid transforms from {} markers",
        transforms.len(),
        markers.len()
    );
    transforms
}

pub fn interpolate_charuco_corners(
    board: &CharucoBoard,
    transforms: &HashMap<u32, Homography>,
    min_markers: usize,
) -> Vec<(usize, Point2<f32>)> {
    let mut corners = Vec::new();

    let mut corner_to_homographies: HashMap<usize, Vec<&Homography>> = HashMap::new();
    for (marker_id, h) in transforms {
        if let Some(ids) = board.marker_surrounding_charuco_corners(*marker_id as i32) {
            for c_id in ids {
                corner_to_homographies.entry(c_id).or_default().push(h);
            }
        }
    }

    for i in 0..board.expected_inner_rows() as i32 {
        for j in 0..board.expected_inner_cols() as i32 {
            let corner_id = match board.charuco_corner_id_from_board_corner(i, j) {
                Some(id) => id as usize,
                None => continue,
            };

            let obj_xy = match board.charuco_object_xy(corner_id as u32) {
                Some(pt) => Point2::new(pt.x, pt.y),
                None => continue,
            };

            let projections: Vec<_> = corner_to_homographies
                .get(&corner_id)
                .map(|hs| hs.iter().map(|h| h.apply(obj_xy)).collect())
                .unwrap_or_default();

            if projections.len() >= min_markers {
                let sum: Point2<f32> = projections
                    .iter()
                    .fold(Point2::origin(), |acc, p| acc + p.coords);
                let avg = Point2::new(
                    sum.x / projections.len() as f32,
                    sum.y / projections.len() as f32,
                );
                corners.push((corner_id, avg));
            }
        }
    }
    debug!(
        "interpolate_charuco_corners: {} corners (from {} transforms)",
        corners.len(),
        transforms.len()
    );
    corners
}

/// Уточнение углов через аппроксимацию линий контура (OpenCV _refineCandidateLines).
/// Использует все точки контура для МНК-аппроксимации 4 сторон и находит пересечения.
fn refine_corner_lines(quad: &mut [Point2<f32>; 4], contour: &[Point2<i32>]) {
    if contour.len() < 8 {
        return; // Недостаточно точек
    }

    // Конвертируем контур в f32
    let contour_f32: Vec<Point2<f32>> = contour
        .iter()
        .map(|p| Point2::new(p.x as f32, p.y as f32))
        .collect();

    // Группы точек для каждой стороны (4 стороны + временная группа)
    let mut side_points: [Vec<Point2<f32>>; 5] =
        [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];

    // Находим индексы углов в контуре
    let mut corner_indices: [usize; 4] = [0; 4];
    for j in 0..4 {
        let corner = quad[j];
        // Находим ближайшую точку контура к углу
        let mut min_dist = f32::MAX;
        let mut min_idx = 0;
        for (i, pt) in contour_f32.iter().enumerate() {
            let d = (pt.x - corner.x).powi(2) + (pt.y - corner.y).powi(2);
            if d < min_dist {
                min_dist = d;
                min_idx = i;
            }
        }
        corner_indices[j] = min_idx;
    }

    // Группируем точки контура по сторонам
    let mut current_side = 0;
    for i in 0..contour_f32.len() {
        // Проверяем, не является ли текущая точка углом
        for j in 0..4 {
            if i == corner_indices[j] {
                current_side = j;
                break;
            }
        }
        side_points[current_side].push(contour_f32[i]);
    }

    // Переносим временную группу (4) в соответствующую сторону
    if !side_points[4].is_empty() {
        let target_side = current_side;
        let temp_points: Vec<Point2<f32>> = side_points[4].drain(..).collect();
        side_points[target_side].extend(temp_points);
    }

    // Определяем направление контура (по порядку индексов углов)
    let inc = if corner_indices[0] > corner_indices[1] && corner_indices[3] > corner_indices[0] {
        -1
    } else if corner_indices[2] > corner_indices[3] && corner_indices[1] > corner_indices[2] {
        -1
    } else {
        1
    };

    // Аппроксимируем линии для каждой стороны методом наименьших квадратов
    let mut lines: [(f32, f32, f32); 4] = [(0.0, 0.0, 0.0); 4];
    for i in 0..4 {
        lines[i] = fit_line_lsq(&side_points[i]);
    }

    // Находим пересечения соседних линий
    for i in 0..4 {
        let line1 = lines[i];
        let line2 = if inc < 0 {
            lines[(i + 1) % 4] // следующая линия
        } else {
            lines[(i + 3) % 4] // предыдущая линия
        };

        if let Some(intersection) = line_intersection(line1, line2) {
            quad[i] = intersection;
        }
    }
}

/// МНК-аппроксимация линии ax + by + c = 0.
/// Возвращает (a, b, c) нормализованные.
fn fit_line_lsq(points: &[Point2<f32>]) -> (f32, f32, f32) {
    if points.len() < 2 {
        return (1.0, 0.0, 0.0);
    }

    // Находим границы
    let mut min_x = points[0].x;
    let mut max_x = points[0].x;
    let mut min_y = points[0].y;
    let mut max_y = points[0].y;

    for p in points {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }

    let dx = max_x - min_x;
    let dy = max_y - min_y;

    if dx > dy {
        // Линия ближе к горизонтальной: y = k*x + b
        // Решаем МНК для y = k*x + b
        let n = points.len() as f32;
        let sum_x: f32 = points.iter().map(|p| p.x).sum();
        let sum_y: f32 = points.iter().map(|p| p.y).sum();
        let sum_x2: f32 = points.iter().map(|p| p.x * p.x).sum();
        let sum_xy: f32 = points.iter().map(|p| p.x * p.y).sum();

        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() < 1e-6 {
            return (0.0, 1.0, -sum_y / n); // Вертикальная линия примерно
        }

        let k = (n * sum_xy - sum_x * sum_y) / denom;
        let b = (sum_y - k * sum_x) / n;

        // Преобразуем y = kx + b в ax + by + c = 0  =>  kx - y + b = 0
        (k, -1.0, b)
    } else {
        // Линия ближе к вертикальной: x = k*y + b
        let n = points.len() as f32;
        let sum_x: f32 = points.iter().map(|p| p.x).sum();
        let sum_y: f32 = points.iter().map(|p| p.y).sum();
        let sum_y2: f32 = points.iter().map(|p| p.y * p.y).sum();
        let sum_xy: f32 = points.iter().map(|p| p.x * p.y).sum();

        let denom = n * sum_y2 - sum_y * sum_y;
        if denom.abs() < 1e-6 {
            return (1.0, 0.0, -sum_x / n); // Горизонтальная линия примерно
        }

        let k = (n * sum_xy - sum_x * sum_y) / denom;
        let b = (sum_x - k * sum_y) / n;

        // Преобразуем x = ky + b в ax + by + c = 0  =>  x - ky - b = 0
        (1.0, -k, -b)
    }
}

/// Находит пересечение двух линий ax + by + c = 0.
fn line_intersection(line1: (f32, f32, f32), line2: (f32, f32, f32)) -> Option<Point2<f32>> {
    let (a1, b1, c1) = line1;
    let (a2, b2, c2) = line2;

    let det = a1 * b2 - a2 * b1;
    if det.abs() < 1e-6 {
        return None; // Параллельны
    }

    let x = (b1 * c2 - b2 * c1) / det;
    let y = (a2 * c1 - a1 * c2) / det;

    Some(Point2::new(x, y))
}

pub fn find_marker_quads(gray: &GrayImage) -> Vec<QuadWithContour> {
    let mut all_quads = Vec::new();
    let max_dim = gray.width().max(gray.height()) as f64;
    let min_perimeter = (0.03 * max_dim) as usize;
    let max_perimeter = (4.0 * max_dim) as usize;

    for win_size in [13, 23] {
        let binary = adaptive_threshold(gray, win_size, 7);

        let contours: Vec<imageproc::contours::Contour<i32>> = find_contours(&binary);
        debug!(
            "find_marker_quads: win={}, {} raw contours",
            win_size,
            contours.len()
        );
        let mut kept = 0usize;

        for c in &contours {
            let n = c.points.len();
            if n < min_perimeter || n > max_perimeter {
                continue;
            }

            // Конвертация imageproc::Point<i32> → Vec<imageproc::point::Point<i32>>
            let pts: Vec<imageproc::point::Point<i32>> = c
                .points
                .iter()
                .map(|p| imageproc::point::Point::new(p.x, p.y))
                .collect();

            let epsilon = n as f64 * 0.05;
            let approx = approximate_polygon_dp(&pts, epsilon, true);

            if approx.len() != 4 || !is_contour_convex(&approx) {
                continue;
            }

            // Мин. расстояние между углами
            let min_dist = min_corner_distance_sq(&approx);
            let threshold = (n as f64 * 0.05).powi(2);
            if min_dist < threshold {
                continue;
            }

            // → [Point2<f32>; 4]
            let corners: [Point2<f32>; 4] = reorder_corners(&[
                Point2::new(approx[0].x as f32, approx[0].y as f32),
                Point2::new(approx[1].x as f32, approx[1].y as f32),
                Point2::new(approx[2].x as f32, approx[2].y as f32),
                Point2::new(approx[3].x as f32, approx[3].y as f32),
            ]);

            // Сохраняем оригинальный контур
            let contour: Vec<Point2<i32>> =
                c.points.iter().map(|p| Point2::new(p.x, p.y)).collect();

            all_quads.push(QuadWithContour { corners, contour });
            kept += 1;
        }
        debug!("find_marker_quads: win={}, kept={} quads", win_size, kept);
    }
    debug!("find_marker_quads: total quads={}", all_quads.len());
    all_quads
}

/// Переупорядочить 4 угла в порядок TL, TR, BR, BL (как OpenCV _reorderCandidatesCorners).
/// Работает для произвольной ориентации доски.
fn reorder_corners(c: &[Point2<f32>; 4]) -> [Point2<f32>; 4] {
    let mut pts = *c;
    // Центр масс
    let cx: f32 = pts.iter().map(|p| p.x).sum::<f32>() / 4.0;
    let cy: f32 = pts.iter().map(|p| p.y).sum::<f32>() / 4.0;
    // Сортируем по углу относительно центра
    pts.sort_by(|a, b| {
        let aa = (a.y - cy).atan2(a.x - cx);
        let ab = (b.y - cy).atan2(b.x - cx);
        aa.partial_cmp(&ab).unwrap_or(std::cmp::Ordering::Equal)
    });
    // TL = минимальная сумма x+y (ближе к началу координат)
    let tl_idx = (0..4)
        .min_by(|&i, &j| {
            let si = pts[i].x + pts[i].y;
            let sj = pts[j].x + pts[j].y;
            si.partial_cmp(&sj).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    [
        pts[tl_idx],
        pts[(tl_idx + 1) % 4],
        pts[(tl_idx + 2) % 4],
        pts[(tl_idx + 3) % 4],
    ]
}

fn is_contour_convex(poly: &[imageproc::point::Point<i32>]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let mut sign = 0i64;
    let n = poly.len();
    for i in 0..n {
        let p0 = poly[i];
        let p1 = poly[(i + 1) % n];
        let p2 = poly[(i + 2) % n];
        let dx1 = (p1.x - p0.x) as i64;
        let dy1 = (p1.y - p0.y) as i64;
        let dx2 = (p2.x - p1.x) as i64;
        let dy2 = (p2.y - p1.y) as i64;
        let cross = dx1 * dy2 - dy1 * dx2;
        if cross != 0 {
            if sign == 0 {
                sign = cross;
            } else if sign.signum() != cross.signum() {
                return false;
            }
        }
    }
    true
}

fn min_corner_distance_sq(poly: &[imageproc::point::Point<i32>]) -> f64 {
    let n = poly.len();
    let mut min_d = f64::MAX;
    for i in 0..n {
        let p0 = poly[i];
        let p1 = poly[(i + 1) % n];
        let dx = (p1.x - p0.x) as f64;
        let dy = (p1.y - p0.y) as f64;
        min_d = min_d.min(dx * dx + dy * dy);
    }
    min_d
}
