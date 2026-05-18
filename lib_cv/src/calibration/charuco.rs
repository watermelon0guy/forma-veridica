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
// После find_marker_quads: удалить quads с почти совпадающими центрами
fn dedup_quads(quads: &mut Vec<[Point2<f32>; 4]>, min_dist: f32) {
    let mut kept = Vec::new();
    for quad in quads.clone() {
        let center = quad.iter().fold(Point2::origin(), |a, p| a + p.coords) / 4.0;
        if kept.iter().all(|k: &[Point2<f32>; 4]| {
            let kc = k.iter().fold(Point2::origin(), |a, p| a + p.coords) / 4.0;
            (center - kc).norm() >= min_dist
        }) {
            kept.push(quad);
        }
    }
    *quads = kept;
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

    let cells: Vec<MarkerCell> = quads
        .iter()
        .map(|q| MarkerCell {
            gc: GridCoords { i: 0, j: 0 },
            corners_img: *q,
        })
        .collect();

    let view = GrayImageView {
        width: img.width() as usize,
        height: img.height() as usize,
        data: img.as_raw(),
    };
    let matcher = Matcher::new(*dict, dict.max_correction_bits);

    let mut perimeters: Vec<f32> = quads
        .iter()
        .map(|q| {
            let d01 = (q[1] - q[0]).norm();
            let d12 = (q[2] - q[1]).norm();
            let d23 = (q[3] - q[2]).norm();
            let d30 = (q[0] - q[3]).norm();
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

/// Инвертировать GrayImage (255 - pixel)
fn invert_binary(img: &GrayImage) -> GrayImage {
    image::ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
        image::Luma([255 - img.get_pixel(x, y).0[0]])
    })
}

pub fn find_marker_quads(gray: &GrayImage) -> Vec<[Point2<f32>; 4]> {
    let mut all_quads = Vec::new();
    let max_dim = gray.width().max(gray.height()) as f64;
    let min_perimeter = (0.03 * max_dim) as usize;
    let max_perimeter = (4.0 * max_dim) as usize;

    for win_size in [13, 23] {
        let binary = adaptive_threshold(gray, win_size, 7);

        // Ищем на прямом И инвертированном (как OpenCV — оба варианта)
        for (label, bin_img) in [("direct", &binary), ("inverted", &invert_binary(&binary))] {
            let contours: Vec<imageproc::contours::Contour<i32>> = find_contours(bin_img);
            debug!(
                "find_marker_quads: win={}, {}: {} raw contours",
                win_size,
                label,
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
                let quad: [Point2<f32>; 4] = reorder_corners(&[
                    Point2::new(approx[0].x as f32, approx[0].y as f32),
                    Point2::new(approx[1].x as f32, approx[1].y as f32),
                    Point2::new(approx[2].x as f32, approx[2].y as f32),
                    Point2::new(approx[3].x as f32, approx[3].y as f32),
                ]);

                all_quads.push(quad);
                kept += 1;
            }
            debug!(
                "find_marker_quads: win={}, {}: kept={} quads",
                win_size, label, kept
            );
        }
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
