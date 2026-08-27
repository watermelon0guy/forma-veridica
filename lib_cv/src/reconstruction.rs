use kiddo::SquaredEuclidean;
use kiddo::immutable::float::kdtree::ImmutableKdTree;
use log::{debug, info, trace, warn};
use nalgebra::{DMatrix, Matrix3, Matrix3x4, Point2, Point3, SVD, Vector3};
use sift::{KeyPoint, Sift};
use std::num::NonZero;
use std::path::Path;
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

/// Вычисляет фундаментальную матрицу F из калибровки двух камер.
/// F = K1^{-T} * [t]_x * R * K0^{-1}
pub fn compute_fundamental_matrix(export: &RigExtrinsicsExport) -> Matrix3<f64> {
    assert!(
        export.cameras.len() == 2,
        "Фундаментальная матрица определена только для 2 камер"
    );
    let k0 = export.cameras[0].k.k_matrix();
    let k1 = export.cameras[1].k.k_matrix();

    // cam_se3_rig[i] = T_Ci_R (rig -> camera_i)
    // T_C1_C0 = T_C1_R * T_R_C0 = cam_se3_rig[1] * cam_se3_rig[0]^{-1}
    let t_c1_c0 = &export.cam_se3_rig[1] * export.cam_se3_rig[0].inverse();

    let r = t_c1_c0.rotation.to_rotation_matrix().into_inner();
    let t = t_c1_c0.translation.vector;

    // Матрица векторного произведения [t]_x
    let t_x = Matrix3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0);

    let k0_inv = k0.try_inverse().expect("K0 не обратима");
    let k1_inv_t = k1.try_inverse().expect("K1 не обратима").transpose();

    k1_inv_t * t_x * r * k0_inv
}

/// Расстояние от точки p1 до эпиполярной линии, соответствующей точке p0.
pub fn epipolar_distance(f: &Matrix3<f64>, p0: &Point2<f64>, p1: &Point2<f64>) -> f64 {
    let x0 = Vector3::new(p0.x, p0.y, 1.0);
    let x1 = Vector3::new(p1.x, p1.y, 1.0);
    let l = f * x0;
    let numerator = x1.dot(&l).abs();
    let denominator = (l[0] * l[0] + l[1] * l[1]).sqrt();
    numerator / denominator
}

/// Фильтрует соответствия по эпиполярному ограничению.
/// Оставляет только те пары, где точка камеры 1 лежит вблизи эпиполярной линии.
/// ВАЖНО: точки должны быть неискажёнными (undistorted) —
/// фундаментальная матрица строится из pinhole-модели без учёта дисторсии.
///
/// Возвращает (отфильтрованные_точки, индексы_прошедших).
pub fn filter_matches_by_epipolar(
    points_2d: &[Vec<Point2<f64>>],
    export: &RigExtrinsicsExport,
    threshold_px: f64,
) -> (Vec<Vec<Point2<f64>>>, Vec<usize>) {
    assert_eq!(
        points_2d.len(),
        2,
        "Эпиполярная фильтрация поддерживает ровно 2 камеры"
    );
    assert_eq!(
        points_2d[0].len(),
        points_2d[1].len(),
        "Количество точек должно совпадать для обеих камер"
    );

    let f = compute_fundamental_matrix(export);
    let num_points = points_2d[0].len();

    let mut good_indices: Vec<usize> = Vec::new();
    for i in 0..num_points {
        let d = epipolar_distance(&f, &points_2d[0][i], &points_2d[1][i]);
        if d < threshold_px {
            good_indices.push(i);
        }
    }

    trace!(
        "Эпиполярная фильтрация: {} из {} соответствий прошли порог {:.1} px",
        good_indices.len(),
        num_points,
        threshold_px
    );

    let filtered = points_2d
        .iter()
        .map(|cam_points| good_indices.iter().map(|&i| cam_points[i]).collect())
        .collect();

    (filtered, good_indices)
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
) -> Result<(Vec<PointContainer>, Vec<bool>), Box<dyn std::error::Error>> {
    if points_2d.len() < 2 || camera_params.cameras.len() < 2 {
        return Err("Требуется минимум 2 камеры для триангуляции".into());
    }
    if points_2d.len() != camera_params.cameras.len() {
        return Err("Количество списков точек должно совпадать с количеством камер".into());
    }

    let num_points = points_2d[0].len();
    trace!("Количество точек для триангуляции: {}", num_points);

    let projections = build_projection_matrices(camera_params);

    // Диагностика: логируем параметры камер
    for (i, cam) in camera_params.cameras.iter().enumerate() {
        let k = cam.k.k_matrix();
        let iso = camera_params.cam_se3_rig[i];
        trace!(
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
        trace!(
            "  rig->cam: t=({:.3},{:.3},{:.3}) R_angle={:.1}°",
            iso.translation.x,
            iso.translation.y,
            iso.translation.z,
            iso.rotation.angle().to_degrees()
        );
        trace!(
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

    // Центры камер в системе рига: C_i = (T_C_R)^{-1} · origin
    let centers: Vec<Vector3<f64>> = camera_params
        .cam_se3_rig
        .iter()
        .map(|iso| iso.inverse().translation.vector)
        .collect();

    let (points_3d, binary_mask) =
        triangulate_points(points_2d, &projections, &centers, MIN_PARALLAX_ANGLE_DEG);

    // Статистика по confidence
    let num_bad = points_3d.iter().filter(|p| p.confidence < 0.25).count();
    trace!(
        "Триангулировано {} точек, из них {} с низкой уверенностью (< 0.25)",
        points_3d.len(),
        num_bad
    );

    Ok((points_3d, binary_mask))
}

/// Порог reprojection error для confidence (пиксели).
/// Точки с ошибкой выше этого получают confidence = 0.
const REPROJ_THRESHOLD_PX: f64 = 5.0;

// ХАРДКОД (13.2): порог угла триангуляции в градусах; по плану переедет в конфиг.
// Для рига exp_2 (встречные камеры, B≈1.4 м, узкий FOV) честные точки имеют
// угол >= ~8.6°, порог отсекает только пересечения ложных матчей
const MIN_PARALLAX_ANGLE_DEG: f64 = 2.0;

/// Проверяет, что угол между лучами «камера → точка» не меньше порога.
/// Лучи почти параллельны => ошибка глубины ~ 1/sin(θ) => глубина — лотерея.
/// Точка в центре камеры (луч нулевой длины) — тоже отбраковка.
fn has_min_parallax(point: &Point3<f64>, centers: &[Vector3<f64>], min_angle_deg: f64) -> bool {
    let cos_min = min_angle_deg.to_radians().cos();
    let v_ref = point.coords - centers[0];
    let d_ref = v_ref.norm();
    if d_ref <= 1e-9 {
        return false;
    }
    for center in &centers[1..] {
        let v = point.coords - center;
        let d = v.norm();
        if d <= 1e-9 {
            return false;
        }
        if v_ref.dot(&v) / (d_ref * d) > cos_min {
            return false;
        }
    }
    true
}

fn triangulate_points(
    points_2d: &[Vec<Point2<f64>>],
    projection_matrices: &[Matrix3x4<f64>],
    centers: &[Vector3<f64>],
    min_parallax_angle_deg: f64,
) -> (Vec<PointContainer>, Vec<bool>) {
    let num_points = points_2d[0].len();
    let num_cameras = projection_matrices.len();
    let mut points_3d = Vec::with_capacity(num_points);
    let mut binary_masks: Vec<bool> = vec![false; num_points];
    let mut cheirality_rejected = 0usize;
    let mut parallax_rejected = 0usize;

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
            trace!("Точка {} на бесконечности (w ≈ 0), пропускаем", pt_i);
            continue;
        }

        let x = x_homog[0] / w;
        let y = x_homog[1] / w;
        let z = x_homog[2] / w;

        // Reprojection error -> confidence
        let mut total_error = 0.0f64;
        let mut behind = false;
        for cam_i in 0..num_cameras {
            let p = &projection_matrices[cam_i];
            let orig = points_2d[cam_i][pt_i];

            // P * [x, y, z, 1]^T -> (proj_x, proj_y, proj_w)
            let proj_x = p[(0, 0)] * x + p[(0, 1)] * y + p[(0, 2)] * z + p[(0, 3)];
            let proj_y = p[(1, 0)] * x + p[(1, 1)] * y + p[(1, 2)] * z + p[(1, 3)];
            // proj_w — глубина точки в системе камеры (третья строка K всегда (0,0,1))
            let proj_w = p[(2, 0)] * x + p[(2, 1)] * y + p[(2, 2)] * z + p[(2, 3)];

            // Cheirality: пиксель не различает точки перед/за камерой (знак
            // сокращается при делении), поэтому после триангуляции проверяем явно (13.1)
            if proj_w <= 0.0 {
                behind = true;
                break;
            }

            let err =
                ((proj_x / proj_w - orig.x).powi(2) + (proj_y / proj_w - orig.y).powi(2)).sqrt();
            total_error += err;
        }

        if behind {
            cheirality_rejected += 1;
            trace!("Точка {pt_i} за камерой (cheirality), пропускаем");
            continue;
        }

        if !has_min_parallax(&Point3::new(x, y, z), centers, min_parallax_angle_deg) {
            parallax_rejected += 1;
            trace!("Точка {pt_i}: угол триангуляции < {min_parallax_angle_deg}°, пропускаем");
            continue;
        }

        let avg_error = total_error / num_cameras as f64;
        let confidence = (1.0 - (avg_error / REPROJ_THRESHOLD_PX).min(1.0)) as f32;

        // Логируем первые 5 точек для диагностики
        if pt_i < 5 {
            trace!(
                "Точка {pt_i}: 3D=({x:.2},{y:.2},{z:.2}) reproj_err={avg_error:.2}px conf={confidence:.2}"
            );
        }

        points_3d.push(PointContainer::new(Point3::new(x, y, z), confidence));
        binary_masks[pt_i] = true;
    }

    trace!(
        "Гейты триангуляции: cheirality {cheirality_rejected}, parallax {parallax_rejected}, из {num_points} точек"
    );

    (points_3d, binary_masks)
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

/// RootSIFT: L1-нормализация + поэлементный sqrt.
/// Преобразует Евклидово расстояние в расстояние Хеллингера,
/// что значительно улучшает качество сопоставления на повторяющихся текстурах.
fn root_sift_normalize(descriptors: &mut [Vec<f32>]) {
    for desc in descriptors.iter_mut() {
        let l1: f32 = desc.iter().map(|v| v.abs()).sum();
        if l1 > 1e-10 {
            for v in desc.iter_mut() {
                *v /= l1;
                *v = v.sqrt();
            }
        }
    }
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

    // RootSIFT нормализация
    for desc in all_descriptors.iter_mut() {
        root_sift_normalize(desc);
    }

    // 2. Строим ImmutableKdTree по дескрипторам камеры 0
    let ref_arrays: Vec<[f32; SIFT_DIM]> = all_descriptors[0]
        .iter()
        .map(|d| to_fixed_array(d))
        .collect();
    let ref_tree: ImmutableKdTree<f32, usize, SIFT_DIM, 32> =
        ImmutableKdTree::new_from_slice(&ref_arrays);

    let ratio_threshold: f32 = 0.75;

    // 3. Для каждой камеры i >= 1: прямой matching
    let mut all_matches: Vec<Vec<FeatureMatch>> = Vec::with_capacity(num_cameras - 1);

    for cam_i in 1..num_cameras {
        let cam_descriptors = &all_descriptors[cam_i];
        let mut cam_matches: Vec<FeatureMatch> = Vec::new();

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

        info!(
            "Камера 0 <-> камера {cam_i}: {} совпадений (из {})",
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

    debug!(
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
/// Возвращает points[cam][point_idx], где все списки упорядочены
/// одинаково - по возрастанию ref_idx.
pub fn gather_points_2d_from_matches(
    all_matches: &[Vec<FeatureMatch>],
    all_keypoints: &[Vec<KeyPoint>],
) -> Vec<Vec<Point2<f64>>> {
    let num_cameras = all_keypoints.len();

    // Собираем уникальные ref_idx в порядке возрастания.
    let mut ref_indices: Vec<usize> = all_matches[0].iter().map(|m| m.ref_idx).collect();
    ref_indices.sort();
    ref_indices.dedup();

    let num_matches = ref_indices.len();

    // Строим маппинг ref_idx -> cam_idx для каждой камеры.
    let mut cam_idx_maps: Vec<std::collections::HashMap<usize, usize>> =
        Vec::with_capacity(num_cameras - 1);
    for cam_matches in all_matches.iter() {
        let mut map = std::collections::HashMap::new();
        for m in cam_matches {
            map.entry(m.ref_idx).or_insert(m.cam_idx);
        }
        cam_idx_maps.push(map);
    }

    let mut points_2d: Vec<Vec<Point2<f64>>> = Vec::with_capacity(num_cameras);

    // Камера 0 — референсная: координаты по ref_idx (одинаковый порядок).
    let mut cam0_points: Vec<Point2<f64>> = Vec::with_capacity(num_matches);
    for &ref_idx in &ref_indices {
        let kp = &all_keypoints[0][ref_idx];
        cam0_points.push(Point2::new(kp.x as f64, kp.y as f64));
    }
    points_2d.push(cam0_points);

    // Камеры 1..N: координаты по cam_idx, в том же порядке ref_idx.
    for (cam_i, cam_map) in cam_idx_maps.iter().enumerate() {
        let actual_cam = cam_i + 1;
        let mut cam_points: Vec<Point2<f64>> = Vec::with_capacity(num_matches);
        for &ref_idx in &ref_indices {
            let cam_idx = cam_map[&ref_idx];
            let kp = &all_keypoints[actual_cam][cam_idx];
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
pub fn save_point_cloud(cloud: &PointCloud, path: &Path) -> std::io::Result<()> {
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
    use optical_flow_lk::{build_pyramid, calc_optical_flow_ex};

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

        let results = calc_optical_flow_ex(
            &prev_pyramid,
            &curr_pyramid,
            &prev_f32,
            None,
            window_size,
            max_iterations,
            0.001,
        );

        // TrackResult -> Point2<f64>
        let new_points: Vec<Point2<f64>> = results
            .into_iter()
            .map(|r| Point2::new(r.pos.0 as f64, r.pos.1 as f64))
            .collect();

        let (w, h) = curr_frames[cam_i].dimensions();
        let in_bounds = new_points
            .iter()
            .filter(|p| p.x >= 0.0 && p.y >= 0.0 && p.x < w as f64 && p.y < h as f64)
            .count();
        trace!(
            "Камера {cam_i}: отслежено {}/{} точек в границах кадра",
            in_bounds,
            new_points.len()
        );

        all_new_points.push(new_points);
    }

    all_new_points
}

/// Epipolar-constrained SIFT matching: KNN by descriptor,
/// then filter by epipolar distance, then Lowe ratio test.
pub fn match_with_epipolar_constraint(
    images: &[image::DynamicImage],
    export: &RigExtrinsicsExport,
    epipolar_threshold_px: f64,
) -> (Vec<Vec<FeatureMatch>>, Vec<Vec<KeyPoint>>) {
    let num_cameras = images.len();
    if num_cameras < 2 {
        warn!("Need at least 2 images for matching");
        return (vec![], vec![]);
    }

    let sift = Sift::new(1.6, 6, 3, 0.5, 0.01, 15.0);

    // 1. SIFT + RootSIFT
    let mut all_keypoints: Vec<Vec<KeyPoint>> = Vec::with_capacity(num_cameras);
    let mut all_descriptors: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_cameras);

    for (i, img) in images.iter().enumerate() {
        let (kp, desc) = sift.detect_and_compute(img);
        debug!("Camera {i}: {} keypoints", kp.len());
        all_keypoints.push(kp);
        all_descriptors.push(desc);
    }

    for desc in all_descriptors.iter_mut() {
        root_sift_normalize(desc);
    }

    // 2. Undistort all keypoints
    let undistorted_kps: Vec<Vec<Point2<f64>>> = all_keypoints
        .iter()
        .enumerate()
        .map(|(cam_i, kps)| {
            let pixels: Vec<Point2<f64>> = kps
                .iter()
                .map(|kp| Point2::new(kp.x as f64, kp.y as f64))
                .collect();
            undistort_points(&pixels, &export.cameras[cam_i])
        })
        .collect();

    // 3. Fundamental matrix
    let f = compute_fundamental_matrix(export);

    // 4. K-d tree on camera 0 descriptors
    let ref_arrays: Vec<[f32; SIFT_DIM]> = all_descriptors[0]
        .iter()
        .map(|d| to_fixed_array(d))
        .collect();
    let ref_tree: ImmutableKdTree<f32, usize, SIFT_DIM, 32> =
        ImmutableKdTree::new_from_slice(&ref_arrays);

    let knn_k: usize = 10;
    let ratio_threshold: f32 = 0.75;

    let mut all_matches: Vec<Vec<FeatureMatch>> = Vec::with_capacity(num_cameras - 1);

    for cam_i in 1..num_cameras {
        let cam_descriptors = &all_descriptors[cam_i];
        let mut cam_matches: Vec<FeatureMatch> = Vec::new();

        for (cam_idx, desc) in cam_descriptors.iter().enumerate() {
            let neighbors = ref_tree
                .nearest_n::<SquaredEuclidean>(&to_fixed_array(desc), NonZero::new(knn_k).unwrap());

            // Filter by epipolar distance
            let p1 = undistorted_kps[cam_i][cam_idx];
            let mut candidates: Vec<(usize, f32)> = Vec::new();
            for n in &neighbors {
                let ref_idx = n.item as usize;
                let p0 = undistorted_kps[0][ref_idx];
                let d = epipolar_distance(&f, &p0, &p1);
                if d < epipolar_threshold_px {
                    candidates.push((ref_idx, n.distance));
                }
            }

            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            if candidates.len() >= 2 {
                if candidates[0].1 < ratio_threshold * candidates[1].1 {
                    cam_matches.push(FeatureMatch {
                        ref_idx: candidates[0].0,
                        cam_idx,
                        distance: candidates[0].1,
                    });
                }
            } else if candidates.len() == 1 {
                cam_matches.push(FeatureMatch {
                    ref_idx: candidates[0].0,
                    cam_idx,
                    distance: candidates[0].1,
                });
            }
        }

        debug!(
            "Camera 0 to camera {cam_i}: {} epipolar matches (out of {})",
            cam_matches.len(),
            cam_descriptors.len()
        );
        all_matches.push(cam_matches);
    }

    (all_matches, all_keypoints)
}

#[cfg(test)]
mod triangulation_tests {
    use super::*;

    /// Синтетическая калибровка: fx=fy=1000, (cx,cy)=(640,400), нулевая дисторсия.
    /// Камера 0 — референс в начале координат, камера 1 — центр в (baseline, 0, 0),
    /// оптические оси параллельны +Z. Собирается через YAML, потому что
    /// RigExtrinsicsExport #[non_exhaustive] — как и в проде (load_calibration_from_yaml)
    fn synthetic_export(baseline: f64) -> RigExtrinsicsExport {
        let yaml = format!(
            r#"kind: rig_extrinsics
cameras:
  - proj: null
    dist: {{k1: 0.0, k2: 0.0, k3: 0.0, p1: 0.0, p2: 0.0, iters: 8}}
    sensor: null
    k: {{fx: 1000.0, fy: 1000.0, cx: 640.0, cy: 400.0, skew: 0.0}}
    _phantom: null
  - proj: null
    dist: {{k1: 0.0, k2: 0.0, k3: 0.0, p1: 0.0, p2: 0.0, iters: 8}}
    sensor: null
    k: {{fx: 1000.0, fy: 1000.0, cx: 640.0, cy: 400.0, skew: 0.0}}
    _phantom: null
sensors: null
cam_se3_rig:
  - rotation: [0.0, 0.0, 0.0, 1.0]
    translation: [0.0, 0.0, 0.0]
  - rotation: [0.0, 0.0, 0.0, 1.0]
    translation: [-{baseline}, 0.0, 0.0]
rig_se3_target: []
mean_reproj_error: 0.0
per_cam_reproj_errors: [0.0, 0.0]
per_feature_residuals: {{}}
"#
        );
        serde_yml::from_str(&yaml).expect("синтетическая калибровка должна десериализоваться")
    }

    /// Проекция точки из координат камеры через K: u = cx + fx·x/z, v = cy + fy·y/z
    fn pixel(x_cam: f64, y_cam: f64, z_cam: f64) -> Point2<f64> {
        Point2::new(
            640.0 + 1000.0 * x_cam / z_cam,
            400.0 + 1000.0 * y_cam / z_cam,
        )
    }

    #[test]
    fn triangulates_known_point_and_keeps_mask() {
        let export = synthetic_export(0.5);
        // Точка (0, 0, 2): камера 0 видит её в центре, камера 1 — смещённой влево
        let p0 = pixel(0.0, 0.0, 2.0);
        let p1 = pixel(-0.5, 0.0, 2.0);
        let (points, mask) = triangulate_points_multiple(&[vec![p0], vec![p1]], &export).unwrap();

        assert_eq!(points.len(), 1);
        assert_eq!(mask, vec![true]);
        assert!((points[0].p.x - 0.0).abs() < 1e-6);
        assert!((points[0].p.y - 0.0).abs() < 1e-6);
        assert!((points[0].p.z - 2.0).abs() < 1e-6);
        assert!(points[0].confidence > 0.99);
    }

    #[test]
    fn rejects_point_behind_camera() {
        let export = synthetic_export(0.5);
        // Зеркальная точка (0, 0, -2): камера 0 проецирует её в тот же пиксель,
        // что и (0, 0, 2) — репроекция знаконезависима. До 13.1 проходила с conf=1.0
        let p0 = pixel(0.0, 0.0, -2.0);
        let p1 = pixel(-0.5, 0.0, -2.0);
        let (points, mask) = triangulate_points_multiple(&[vec![p0], vec![p1]], &export).unwrap();

        assert!(points.is_empty());
        assert_eq!(mask, vec![false]);
    }

    #[test]
    fn rejects_low_parallax_but_keeps_wide_baseline() {
        // Близкие камеры + далёкая точка: угол ~0.06° — глубина не определяется
        let near = synthetic_export(0.1);
        let p0 = pixel(0.0, 0.0, 100.0);
        let p1 = pixel(-0.1, 0.0, 100.0);
        let (points, mask) = triangulate_points_multiple(&[vec![p0], vec![p1]], &near).unwrap();
        assert!(points.is_empty());
        assert_eq!(mask, vec![false]);

        // Та же точка при широкой базе: угол ~5.7° — проходит
        let wide = synthetic_export(10.0);
        let p1w = pixel(-10.0, 0.0, 100.0);
        let (points, mask) = triangulate_points_multiple(&[vec![p0], vec![p1w]], &wide).unwrap();
        assert_eq!(points.len(), 1);
        assert_eq!(mask, vec![true]);
        assert!((points[0].p.z - 100.0).abs() < 1e-6);
    }

    #[test]
    fn mask_stays_aligned_with_cloud() {
        let export = synthetic_export(0.5);
        // Три точки: валидная, за камерой, с малым параллаксом
        let p0 = vec![
            pixel(0.0, 0.0, 2.0),
            pixel(0.0, 0.0, -2.0),
            pixel(0.0, 0.0, 1000.0),
        ];
        let p1 = vec![
            pixel(-0.5, 0.0, 2.0),
            pixel(-0.5, 0.0, -2.0),
            pixel(-0.5, 0.0, 1000.0),
        ];
        let (points, mask) = triangulate_points_multiple(&[p0, p1], &export).unwrap();

        assert_eq!(points.len(), mask.iter().filter(|m| **m).count());
        assert_eq!(mask, vec![true, false, false]);
        // Единственная выжившая — первая: порядок «облако ↔ маска» не поехал
        assert!((points[0].p.z - 2.0).abs() < 1e-6);
    }
}
