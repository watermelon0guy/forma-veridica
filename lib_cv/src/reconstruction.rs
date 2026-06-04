use kiddo::KdTree;
use kiddo::SquaredEuclidean;
use log::{debug, error, info, warn};
use nalgebra::{DMatrix, Matrix3x4, Point2, Point3, SVD};
use sift::{KeyPoint, Sift};
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

/// Убирает дисторсию: пиксельные координаты -> нормализованные (z = 1).
/// Использует полный обратный ход модели камеры.
pub fn undistort_points(
    points_px: &[Point2<f64>], // N искажённых пикселей
    camera: &PinholeCamera,
) -> Vec<Point2<f64>> {
    // N неискажённых нормализованных
    points_px
        .iter()
        .map(|px| {
            let ray = camera.backproject_pixel(px);
            Point2::new(ray.point.x, ray.point.y)
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
        let confidence = (1.0 - (avg_error / 5.0).min(1.0)) as f32;

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
/// из остальных через KNN-поиск (k-d дерево) + Lowe's ratio test.
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

    let sift = Sift::new(1.6, 4, 3, 0.5, 0.04, 10.0);

    // 1. SIFT на всех камерах
    let mut all_keypoints: Vec<Vec<KeyPoint>> = Vec::with_capacity(num_cameras);
    let mut all_descriptors: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_cameras);

    for (i, img) in images.iter().enumerate() {
        let (kp, desc) = sift.detect_and_compute(img);
        info!("Камера {i}: найдено {} ключевых точек", kp.len());
        all_keypoints.push(kp);
        all_descriptors.push(desc);
    }

    // 2. Строим k-d дерево по дескрипторам камеры 0 (референсной)
    let ref_descriptors = &all_descriptors[0];
    let mut tree: KdTree<f32, SIFT_DIM> = KdTree::new();
    for (idx, desc) in ref_descriptors.iter().enumerate() {
        tree.add(&to_fixed_array(desc), idx as u64);
    }

    // 3. Для каждой камеры i >= 1: 2 ближайших соседа + ratio test
    let mut all_matches: Vec<Vec<FeatureMatch>> = Vec::with_capacity(num_cameras - 1);

    for cam_i in 1..num_cameras {
        let cam_descriptors = &all_descriptors[cam_i];
        let mut cam_matches = Vec::new();

        for (cam_idx, desc) in cam_descriptors.iter().enumerate() {
            let neighbors = tree.nearest_n::<SquaredEuclidean>(&to_fixed_array(desc), 2);

            // Lowe's ratio test: d_best < 0.7 * d_second
            if neighbors.len() == 2 && neighbors[0].distance < 0.7 * neighbors[1].distance {
                cam_matches.push(FeatureMatch {
                    ref_idx: neighbors[0].item as usize,
                    cam_idx,
                    distance: neighbors[0].distance,
                });
            }
        }

        info!(
            "Камера 0 ↔ камера {cam_i}: {} совпадений (из {})",
            cam_matches.len(),
            cam_descriptors.len()
        );
        all_matches.push(cam_matches);
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
