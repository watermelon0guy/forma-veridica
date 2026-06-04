use kiddo::SquaredEuclidean;
use kiddo::immutable::float::kdtree::ImmutableKdTree;
use log::{debug, error, info, warn};
use nalgebra::{DMatrix, Matrix3x4, Point2, Point3, SVD};
use sift::{KeyPoint, Sift};
use std::num::NonZero;
use vision_calibration::core::{Iso3, PinholeCamera};
use vision_calibration::rig_extrinsics::RigExtrinsicsExport;

#[derive(Debug, Clone)]
pub struct PointContainer {
    pub p: Point3<f64>,
    pub color: Option<(u8, u8, u8)>, // RGB цвет точки
    pub track_id: Option<usize>,     // ID для отслеживания точки во времени
    pub confidence: f32,             // Уверенность в позиции точки
}

impl PointContainer {
    pub fn new(p: Point3<f64>, confidence: f32) -> Self {
        Self {
            p,
            color: None,
            track_id: None,
            confidence,
        }
    }
}

/// Структура для хранения облака точек
#[derive(Debug, Clone)]
pub struct PointCloud {
    pub points: Vec<PointContainer>,
    pub timestamp: usize, // Временная метка кадра
}

/// Собирает проекционные матрицы P_i = K_i · [R_i | t_i] для всех камер.
pub fn build_projection_matrices(export: &RigExtrinsicsExport) -> Vec<Matrix3x4<f64>> {
    let n = export.cameras.len();
    let mut projections: Vec<Matrix3x4<f64>> = Vec::with_capacity(n);

    for i in 0..n {
        if i == 0 && export.cam_se3_rig[0] != Iso3::identity() {
            warn!(
                "Первая камера должна быть референсной, \
                 но её rig->camera transform не identity"
            );
        }

        let k = export.cameras[i].k.k_matrix();
        let rt: Matrix3x4<f64> = export.cam_se3_rig[i]
            .to_homogeneous()
            .fixed_view::<3, 4>(0, 0)
            .into_owned();
        projections.push(k * rt);
    }

    projections
}

/// Убирает дисторсию: пиксельные координаты → неискажённые пиксельные.
/// Обратный ход: K⁻¹ → undistort → K (без дисторсии).
pub fn undistort_points(
    points_px: &[Point2<f64>], // N искажённых пикселей
    camera: &PinholeCamera,
) -> Vec<Point2<f64>> {
    // N неискажённых пикселей
    let k = camera.k.k_matrix();
    points_px
        .iter()
        .map(|px| {
            let ray = camera.backproject_pixel(px);
            // ray.point = (x_norm, y_norm, 1.0) — неискажённые нормализованные
            let pixel = k * ray.point; // K · (x, y, 1) → (u_undist, v_undist, 1)
            Point2::new(pixel.x, pixel.y)
        })
        .collect()
}

pub fn triangulate_points_multiple(
    points_2d: &[Vec<Point2<f64>>],
    camera_params: &RigExtrinsicsExport,
) -> Result<Vec<PointContainer>, Box<dyn std::error::Error>> {
    if points_2d.len() < 2 || camera_params.cameras.len() < 2 {
        error!("Недостаточно камер или наборов точек");
        return Err("Требуется минимум 2 камеры для триангуляции".into());
    }
    if points_2d.len() != camera_params.cameras.len() {
        error!("Количество наборов точек не соответствует количеству камер");
        return Err("Количество списков точек должно совпадать с количеством камер".into());
    }

    let num_points = points_2d[0].len();
    debug!("Количество точек для триангуляции: {}", num_points);

    let projections = build_projection_matrices(camera_params);

    // Диагностика: логируем параметры камер
    for (i, cam) in camera_params.cameras.iter().enumerate() {
        let k = cam.k.k_matrix();
        let iso = camera_params.cam_se3_rig[i];
        debug!(
            "Камера {i}: fx={:.1} fy={:.1} cx={:.1} cy={:.1} dist=({:.4},{:.4},{:.4},{:.4},{:.4})",
            k[(0, 0)],
            k[(1, 1)],
            k[(0, 2)],
            k[(1, 2)],
            cam.dist.k1,
            cam.dist.k2,
            cam.dist.k3,
            cam.dist.p1,
            cam.dist.p2
        );
        debug!(
            "  rig->cam: t=({:.3},{:.3},{:.3}) R_angle={:.1}°",
            iso.translation.x,
            iso.translation.y,
            iso.translation.z,
            iso.rotation.angle().to_degrees()
        );
        debug!(
            "  R matrix: [{:.3},{:.3},{:.3}; {:.3},{:.3},{:.3}; {:.3},{:.3},{:.3}]",
            iso.rotation.to_rotation_matrix()[(0, 0)],
            iso.rotation.to_rotation_matrix()[(0, 1)],
            iso.rotation.to_rotation_matrix()[(0, 2)],
            iso.rotation.to_rotation_matrix()[(1, 0)],
            iso.rotation.to_rotation_matrix()[(1, 1)],
            iso.rotation.to_rotation_matrix()[(1, 2)],
            iso.rotation.to_rotation_matrix()[(2, 0)],
            iso.rotation.to_rotation_matrix()[(2, 1)],
            iso.rotation.to_rotation_matrix()[(2, 2)],
        );
    }

    let result = triangulate_points(points_2d, &projections);

    // Статистика по confidence
    let num_bad = result.iter().filter(|p| p.confidence < 0.25).count();
    info!(
        "Триангулировано {} точек, из них {} с низкой уверенностью (< 0.25)",
        result.len(),
        num_bad
    );

    Ok(result)
}

/// Порог reprojection error для confidence (пиксели).
/// Точки с ошибкой выше этого получают confidence = 0.
const REPROJ_THRESHOLD_PX: f64 = 25.0;

fn triangulate_points(
    points_2d: &[Vec<Point2<f64>>],
    projection_matrices: &[Matrix3x4<f64>],
) -> Vec<PointContainer> {
    let num_points = points_2d[0].len();
    let num_cameras = projection_matrices.len();
    let mut points_3d = Vec::with_capacity(num_points);

    for pt_i in 0..num_points {
        // Строим матрицу A (2N × 4) для DLT
        let mut a = DMatrix::zeros(2 * num_cameras, 4);

        for cam_i in 0..num_cameras {
            let pt = points_2d[cam_i][pt_i];
            let p = &projection_matrices[cam_i];

            // Строки: p.row(0) - u * p.row(2),  p.row(1) - v * p.row(2)
            for j in 0..4 {
                a[(2 * cam_i, j)] = p[(0, j)] - pt.x * p[(2, j)];
                a[(2 * cam_i + 1, j)] = p[(1, j)] - pt.y * p[(2, j)];
            }
        }

        // SVD: A = U·Σ·V^T. Последняя строка V^T -> искомая точка
        let svd = SVD::new(a, true, true);
        let v_t = svd.v_t.unwrap();
        let x_homog = v_t.row(3); // строка с минимальным сингулярным числом

        let w = x_homog[3];
        if w.abs() < 1e-12 {
            warn!("Точка {} на бесконечности (w ≈ 0), пропускаем", pt_i);
            continue;
        }

        let x = x_homog[0] / w;
        let y = x_homog[1] / w;
        let z = x_homog[2] / w;

        // Reprojection error -> confidence
        let mut total_error = 0.0f64;
        for cam_i in 0..num_cameras {
            let p = &projection_matrices[cam_i];
            let orig = points_2d[cam_i][pt_i];

            // P * [x, y, z, 1]^T -> (proj_x, proj_y, proj_w)
            let proj_x = p[(0, 0)] * x + p[(0, 1)] * y + p[(0, 2)] * z + p[(0, 3)];
            let proj_y = p[(1, 0)] * x + p[(1, 1)] * y + p[(1, 2)] * z + p[(1, 3)];
            let proj_w = p[(2, 0)] * x + p[(2, 1)] * y + p[(2, 2)] * z + p[(2, 3)];

            let err =
                ((proj_x / proj_w - orig.x).powi(2) + (proj_y / proj_w - orig.y).powi(2)).sqrt();
            total_error += err;
        }

        let avg_error = total_error / num_cameras as f64;
        let confidence = (1.0 - (avg_error / REPROJ_THRESHOLD_PX).min(1.0)) as f32;

        // Логируем первые 5 точек для диагностики
        if pt_i < 5 {
            debug!(
                "Точка {pt_i}: 3D=({x:.2},{y:.2},{z:.2}) reproj_err={avg_error:.2}px conf={confidence:.2}"
            );
        }

        points_3d.push(PointContainer::new(Point3::new(x, y, z), confidence));
    }

    points_3d
}

/// Размерность SIFT-дескриптора (всегда 128 чисел).
const SIFT_DIM: usize = 128;

/// Сопоставление: ключевая точка референсной камеры <-> точка другой камеры.
#[derive(Debug, Clone)]
pub struct FeatureMatch {
    pub ref_idx: usize, // индекс keypoint в камере 0
    pub cam_idx: usize, // индекс keypoint в целевой камере (i >= 1)
    pub distance: f32,  // расстояние между дескрипторами
}

/// Преобразует срез f32 в массив фиксированной длины для kiddo.
fn to_fixed_array(v: &[f32]) -> [f32; SIFT_DIM] {
    let mut arr = [0.0f32; SIFT_DIM];
    arr.copy_from_slice(&v[..SIFT_DIM]);
    arr
}

/// Детектирует SIFT-признаки на всех камерах и сопоставляет камеру 0 с каждой
/// из остальных через KNN-поиск (k-d дерево) + Lowe's ratio test + взаимная проверка.
pub fn match_first_camera_features_to_all(
    images: &[image::DynamicImage],
) -> (
    Vec<Vec<FeatureMatch>>, // matches[other_cam][match_idx]
    Vec<Vec<KeyPoint>>,     // keypoints[cam]
    Vec<Vec<Vec<f32>>>,     // descriptors[cam][kp_idx]
) {
    let num_cameras = images.len();
    if num_cameras < 2 {
        warn!("Нужно минимум 2 изображения для сопоставления");
        return (vec![], vec![], vec![]);
    }

    // Максимальное качество: низкий порог контраста, больше октав
    let sift = Sift::new(
        1.6,  // sigma
        6,    // num_octaves — больше для лучшего покрытия масштабов
        3,    // num_intervals
        0.5,  // assumed_blur
        0.01, // contrast_threshold — ниже = больше точек
        15.0, // edge_threshold — выше = не отбрасываем точки на гранях
    );

    // 1. SIFT на всех камерах
    let mut all_keypoints: Vec<Vec<KeyPoint>> = Vec::with_capacity(num_cameras);
    let mut all_descriptors: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_cameras);

    for (i, img) in images.iter().enumerate() {
        let (kp, desc) = sift.detect_and_compute(img);
        info!("Камера {i}: найдено {} ключевых точек", kp.len());
        all_keypoints.push(kp);
        all_descriptors.push(desc);
    }

    // 2. Строим ImmutableKdTree по дескрипторам камеры 0
    let ref_arrays: Vec<[f32; SIFT_DIM]> = all_descriptors[0]
        .iter()
        .map(|d| to_fixed_array(d))
        .collect();
    let ref_tree: ImmutableKdTree<f32, usize, SIFT_DIM, 32> =
        ImmutableKdTree::new_from_slice(&ref_arrays);

    let ratio_threshold: f32 = 0.6; // жёстче фильтр

    // 3. Для каждой камеры i >= 1: двунаправленный matching
    let mut all_matches: Vec<Vec<FeatureMatch>> = Vec::with_capacity(num_cameras - 1);

    for cam_i in 1..num_cameras {
        let cam_descriptors = &all_descriptors[cam_i];
        let mut cam_matches: Vec<FeatureMatch> = Vec::new();

        // Прямой поиск: cam_i -> camera 0
        for (cam_idx, desc) in cam_descriptors.iter().enumerate() {
            let neighbors = ref_tree
                .nearest_n::<SquaredEuclidean>(&to_fixed_array(desc), NonZero::new(2).unwrap());

            if neighbors.len() == 2
                && neighbors[0].distance < ratio_threshold * neighbors[1].distance
            {
                cam_matches.push(FeatureMatch {
                    ref_idx: neighbors[0].item as usize,
                    cam_idx,
                    distance: neighbors[0].distance,
                });
            }
        }

        // Обратный поиск: camera 0 -> cam_i (взаимная проверка)
        let cam_arrays: Vec<[f32; SIFT_DIM]> =
            cam_descriptors.iter().map(|d| to_fixed_array(d)).collect();
        let cam_tree: ImmutableKdTree<f32, usize, SIFT_DIM, 32> =
            ImmutableKdTree::new_from_slice(&cam_arrays);

        // Взаимная проверка: для каждого прямого матча проверяем обратный
        let mut reciprocal_matches: Vec<FeatureMatch> = Vec::new();
        let ref_desc_all = &all_descriptors[0];
        for m in &cam_matches {
            let ref_desc = &ref_desc_all[m.ref_idx];
            let back_neighbors = cam_tree
                .nearest_n::<SquaredEuclidean>(&to_fixed_array(ref_desc), NonZero::new(1).unwrap());

            if !back_neighbors.is_empty() && back_neighbors[0].item as usize == m.cam_idx {
                reciprocal_matches.push(m.clone());
            }
        }

        info!(
            "Камера 0 ↔ камера {cam_i}: {}/{} совпадений (взаимных), до проверки: {}",
            reciprocal_matches.len(),
            cam_descriptors.len(),
            cam_matches.len()
        );
        all_matches.push(reciprocal_matches);
    }

    (all_matches, all_keypoints, all_descriptors)
}

/// Оставляет только сопоставления, где точка видна ВО ВСЕХ камерах.
pub fn min_visible_match_set(
    all_matches: &[Vec<FeatureMatch>],
    ref_num_keypoints: usize,
) -> Vec<Vec<FeatureMatch>> {
    if all_matches.is_empty() {
        return vec![];
    }

    // Какие keypoints камеры 0 есть во всех остальных камерах
    let common_ref_indices: Vec<usize> = (0..ref_num_keypoints)
        .filter(|ref_idx| {
            all_matches
                .iter()
                .all(|cam| cam.iter().any(|m| m.ref_idx == *ref_idx))
        })
        .collect();

    info!(
        "Точек, видимых во всех камерах: {} из {}",
        common_ref_indices.len(),
        ref_num_keypoints
    );

    // Фильтруем, оставляя только общие точки
    all_matches
        .iter()
        .map(|cam| {
            cam.iter()
                .filter(|m| common_ref_indices.contains(&m.ref_idx))
                .cloned()
                .collect()
        })
        .collect()
}

/// Извлекает 2D-координаты из отфильтрованных матчей.
/// Возвращает points[cam][point_idx].
pub fn gather_points_2d_from_matches(
    all_matches: &[Vec<FeatureMatch>],
    all_keypoints: &[Vec<KeyPoint>],
) -> Vec<Vec<Point2<f64>>> {
    let num_cameras = all_keypoints.len();
    let num_matches = all_matches[0].len();

    let mut points_2d: Vec<Vec<Point2<f64>>> = Vec::with_capacity(num_cameras);

    // Камера 0 — референсная: координаты по ref_idx
    let mut cam0_points: Vec<Point2<f64>> = Vec::with_capacity(num_matches);
    for m in &all_matches[0] {
        let kp = &all_keypoints[0][m.ref_idx];
        cam0_points.push(Point2::new(kp.x as f64, kp.y as f64));
    }
    points_2d.push(cam0_points);

    // Камеры 1..N: координаты по cam_idx
    for (cam_i, cam_matches) in all_matches.iter().enumerate() {
        let actual_cam = cam_i + 1;
        let mut cam_points: Vec<Point2<f64>> = Vec::with_capacity(num_matches);
        for m in cam_matches {
            let kp = &all_keypoints[actual_cam][m.cam_idx];
            cam_points.push(Point2::new(kp.x as f64, kp.y as f64));
        }
        points_2d.push(cam_points);
    }

    points_2d
}

/// Копирует цвет из референсного изображения (камера 0) в облако точек.
/// Использует исходные (искажённые) пиксельные координаты для lookup.
pub fn add_color_to_point_cloud(
    cloud: &mut PointCloud,
    points_2d_cam0: &[Point2<f64>],
    ref_image: &image::DynamicImage,
) {
    let rgb = ref_image.to_rgb8();
    let (w, h) = rgb.dimensions();

    for (i, point) in cloud.points.iter_mut().enumerate() {
        let x = points_2d_cam0[i].x as u32;
        let y = points_2d_cam0[i].y as u32;
        if x < w && y < h {
            let pixel = rgb.get_pixel(x, y);
            point.color = Some((pixel[0], pixel[1], pixel[2]));
        }
    }
}

/// Удаляет точки с уверенностью ниже порога.
pub fn filter_point_cloud_by_confidence(cloud: &mut PointCloud, threshold: f32) {
    cloud.points.retain(|p| p.confidence >= threshold);
}

/// Сохраняет облако точек в формате PLY (Point Cloud Library).
pub fn save_point_cloud<P: AsRef<std::path::Path>>(
    cloud: &PointCloud,
    path: P,
) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;

    let has_color = cloud.points.iter().any(|p| p.color.is_some());

    writeln!(file, "ply")?;
    writeln!(file, "format ascii 1.0")?;
    writeln!(file, "element vertex {}", cloud.points.len())?;
    writeln!(file, "property float x")?;
    writeln!(file, "property float y")?;
    writeln!(file, "property float z")?;
    if has_color {
        writeln!(file, "property uchar red")?;
        writeln!(file, "property uchar green")?;
        writeln!(file, "property uchar blue")?;
    }
    writeln!(file, "property float confidence")?;
    writeln!(file, "end_header")?;

    for point in &cloud.points {
        if has_color {
            let (r, g, b) = point.color.unwrap_or((128, 128, 128));
            writeln!(
                file,
                "{} {} {} {} {} {} {}",
                point.p.x, point.p.y, point.p.z, r, g, b, point.confidence
            )?;
        } else {
            writeln!(
                file,
                "{} {} {} {}",
                point.p.x, point.p.y, point.p.z, point.confidence
            )?;
        }
    }

    Ok(())
}

/// Трекинг точек между двумя кадрами для всех камер через Lucas-Kanade.
/// Для каждой камеры независимо: строит пирамиды, запускает optical flow,
/// возвращает новые координаты. Не фильтрует «потерянные» точки —
/// плохие будут отсеяны на этапе триангуляции по reprojection error.
pub fn track_points_optical_flow_all(
    prev_frames: &[image::GrayImage],
    curr_frames: &[image::GrayImage],
    prev_points: &[Vec<Point2<f64>>],
    window_size: usize,
    max_iterations: usize,
    pyramid_levels: usize,
) -> Vec<Vec<Point2<f64>>> {
    use optical_flow_lk::{build_pyramid, calc_optical_flow};

    let num_cameras = prev_frames.len();
    let mut all_new_points: Vec<Vec<Point2<f64>>> = Vec::with_capacity(num_cameras);

    for cam_i in 0..num_cameras {
        let prev_pyramid = build_pyramid(&prev_frames[cam_i], pyramid_levels);
        let curr_pyramid = build_pyramid(&curr_frames[cam_i], pyramid_levels);

        // Point2<f64> -> (f32, f32) для optical flow
        let prev_f32: Vec<(f32, f32)> = prev_points[cam_i]
            .iter()
            .map(|p| (p.x as f32, p.y as f32))
            .collect();

        let new_f32 = calc_optical_flow(
            &prev_pyramid,
            &curr_pyramid,
            &prev_f32,
            window_size,
            max_iterations,
        );

        // (f32, f32) -> Point2<f64>
        let new_points: Vec<Point2<f64>> = new_f32
            .into_iter()
            .map(|(x, y)| Point2::new(x as f64, y as f64))
            .collect();

        let (w, h) = curr_frames[cam_i].dimensions();
        let in_bounds = new_points
            .iter()
            .filter(|p| p.x >= 0.0 && p.y >= 0.0 && p.x < w as f64 && p.y < h as f64)
            .count();
        debug!(
            "Камера {cam_i}: отслежено {}/{} точек в границах кадра",
            in_bounds,
            new_points.len()
        );

        all_new_points.push(new_points);
    }

    all_new_points
}
