use std::path::PathBuf;

use calib_targets::{
    Coord,
    aruco::MarkerDetection,
    charuco::{CharucoBoard, CharucoCorner, CharucoDetectionResult, CharucoParams},
    core::GridAlignment,
    detect::detect_charuco_best,
};
use image::GrayImage;
use log::{debug, error, info, warn};
use nalgebra::{Point2, Point3};
use vision_calibration::{
    core::{
        BrownConrady5, CorrespondenceView, FxFyCxCySkew, NoMeta, PlanarDataset, RigView,
        RigViewObs, View,
    },
    planar_intrinsics::{self, FilterOptions, PlanarIntrinsicsExport},
    prelude::PlanarIntrinsicsProblem,
    rig_extrinsics::{
        self, RigExtrinsicsExport, RigExtrinsicsInput, RigExtrinsicsProblem,
        RigIntrinsicsManualInit, run_calibration, step_intrinsics_init_all_with_seed,
        step_intrinsics_optimize_all, step_rig_init, step_rig_optimize,
    },
    session::CalibrationSession,
};

use crate::calibration::charuco::{
    build_marker_homographies, detect_aruco_markers, filter_bad_charuco_corners,
    interpolate_charuco_corners, refine_charuco_corners,
};

pub mod charuco;
pub mod params;

pub use params::{DatasetParams, DetectionParams, SolverParams};

pub fn get_charuco_grid_first(
    charuco_board: &CharucoBoard,
    img: &GrayImage,
) -> Option<CharucoDetectionResult> {
    let board_spec = charuco_board.spec();
    let params_sweep = CharucoParams::sweep_for_board(&board_spec);
    match detect_charuco_best(img, &params_sweep) {
        Ok(result) => {
            debug!(
                "get_charuco_grid_first: {} corners, {} markers",
                result.corners.len(),
                result.markers.len()
            );
            Some(result)
        }
        Err(e) => {
            debug!("get_charuco_grid_first: detection error — {e}");
            None
        }
    }
}

pub fn get_charuco_marker_first(
    charuco_board: &CharucoBoard,
    img: &GrayImage,
    detection: &DetectionParams,
) -> Option<CharucoDetectionResult> {
    let markers = detect_aruco_markers(img, &charuco_board.spec().dictionary, detection);

    let transforms = build_marker_homographies(&charuco_board, &markers);
    if transforms.is_empty() {
        return None;
    }
    let corners =
        interpolate_charuco_corners(charuco_board, &transforms, detection.min_markers_per_corner);
    let mut corners = refine_charuco_corners(charuco_board, corners, &markers, img, detection);
    filter_bad_charuco_corners(charuco_board, &mut corners, &markers);
    if corners.is_empty() {
        None
    } else {
        Some(convert_to_charuco_result(charuco_board, &corners, markers))
    }
}

fn convert_to_charuco_result(
    board: &CharucoBoard,
    corners: &[(usize, Point2<f32>)],
    markers: Vec<MarkerDetection>,
) -> CharucoDetectionResult {
    // Строим обратный маппинг corner_id -> Coord
    // charuco_corner_id_from_board_corner(col, row): 1-based, col ∈ 1..cols, row ∈ 1..rows
    let mut id_to_grid: std::collections::HashMap<u32, Coord> = std::collections::HashMap::new();
    for row in 1..=board.expected_inner_rows() as i32 {
        for col in 1..=board.expected_inner_cols() as i32 {
            if let Some(cid) = board.charuco_corner_id_from_board_corner(col, row) {
                id_to_grid.insert(cid, Coord::new(col, row));
            }
        }
    }

    let charuco_corners: Vec<CharucoCorner> = corners
        .iter()
        .filter_map(|(corner_id, pixel_pos)| {
            let corner_id = *corner_id as u32;
            let target_position = board.charuco_object_xy(corner_id)?;
            let grid = id_to_grid.get(&corner_id)?;
            Some(CharucoCorner::new(
                *pixel_pos,
                *grid,
                corner_id,
                target_position,
                1.0, // score
            ))
        })
        .collect();

    CharucoDetectionResult::new(charuco_corners, markers, GridAlignment::IDENTITY)
}

pub fn calibrate_camera(
    correspondence_views: Vec<CorrespondenceView>,
    solver: &SolverParams,
) -> Result<PlanarIntrinsicsExport, Box<dyn std::error::Error>> {
    debug!("Калибровка камеры: {} кадров", correspondence_views.len());
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    let views = corr_views_to_view_no_meta(correspondence_views);
    let dataset = PlanarDataset::new(views)?;
    session.set_input(dataset)?;

    let mut filter_option = FilterOptions::default();
    filter_option.max_reproj_error = solver.max_reproj_error;
    filter_option.min_points_per_view = solver.min_points_per_view;
    filter_option.remove_sparse_views = solver.remove_sparse_views;
    planar_intrinsics::run_calibration_with_filtering(&mut session, filter_option)?;

    let intrinsics = session.export()?;

    Ok(intrinsics)
}

pub fn correspondence_view_from_charuco(
    charuco_board: &CharucoBoard,
    img: &image::ImageBuffer<image::Luma<u8>, Vec<u8>>,
    detection: &DetectionParams,
) -> Option<CorrespondenceView> {
    let charuco_detection = match get_charuco_marker_first(charuco_board, img, detection) {
        Some(charuco_det) => charuco_det,
        None => {
            return None;
        }
    };
    if charuco_detection.corners.is_empty() {
        return None;
    }
    let mut points_3d: Vec<Point3<f64>> = Vec::new();
    let mut points_2d: Vec<Point2<f64>> = Vec::new();
    for corner in &charuco_detection.corners {
        let target_position = corner.target_position;
        points_3d.push(Point3::new(
            target_position.x as f64,
            target_position.y as f64,
            0.0,
        ));
        points_2d.push(Point2::new(
            corner.position.x as f64,
            corner.position.y as f64,
        ));
    }
    debug!("ChArUco: {} углов найдено", points_3d.len());
    match CorrespondenceView::new(points_3d, points_2d) {
        Ok(view) => Some(view),
        Err(e) => {
            warn!("Ошибка CorrespondenceView: {e}");
            None
        }
    }
}

pub fn corr_views_to_view_no_meta(corr_views: Vec<CorrespondenceView>) -> Vec<View<NoMeta>> {
    corr_views
        .into_iter()
        .map(|corr_view| View {
            obs: corr_view,
            meta: NoMeta,
        })
        .collect()
}

pub fn calibrate_multiple_with_charuco_from_images(
    imgs_sets: &Vec<Vec<GrayImage>>,
    charuco_board: &CharucoBoard,
    detection: &DetectionParams,
) -> Result<RigExtrinsicsExport, Box<dyn std::error::Error>> {
    debug!("Start multiple cameras calibration");
    let num_cameras = imgs_sets.len();
    debug!(
        "Количество наборов изображений для калибровки: {}.",
        imgs_sets.len()
    );
    if imgs_sets.len() < 2 {
        error!("Ошибка: для калибровки требуется как минимум 2 набора изображений.");
        return Err("Недостаточно камер".into());
    }

    let num_frames = imgs_sets[0].len();

    let mut rigs = Vec::new();

    for set_num in 0..num_frames {
        let mut correspondences = Vec::new();
        for cam_idx in 0..num_cameras {
            correspondences.push(correspondence_view_from_charuco(
                charuco_board,
                &imgs_sets[cam_idx][set_num],
                detection,
            ));
        }

        let rig_view = RigView {
            obs: RigViewObs {
                cameras: correspondences,
            },
            meta: NoMeta,
        };

        rigs.push(rig_view);
    }

    let rig_dataset = RigExtrinsicsInput::new(rigs, num_cameras)?;

    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_input(rig_dataset)?;
    run_calibration(&mut session)?;
    let result = session.export()?;

    Ok(result)
}

pub fn update_rigs(
    rigs: &mut Vec<RigView<NoMeta>>,
    cams_imgs: &Vec<GrayImage>,
    charuco_board: &CharucoBoard,
    detection: &DetectionParams,
    dataset: &DatasetParams,
) {
    let mut correspondences = Vec::new();
    let num_cameras = cams_imgs.len();
    for cam_idx in 0..num_cameras {
        correspondences.push(correspondence_view_from_charuco(
            charuco_board,
            &cams_imgs[cam_idx],
            detection,
        ));
    }

    for opt_cv in &correspondences {
        if let Some(cv) = opt_cv {
            if cv.len() < dataset.min_corners_per_view {
                return;
            }
        }
    }

    // Добавляем риг только если есть валидные детекции минимум с
    // `min_cameras_per_frame` камер
    let valid_cams = correspondences.iter().filter(|c| c.is_some()).count();
    if valid_cams >= dataset.min_cameras_per_frame {
        let rig_view = RigView {
            obs: RigViewObs {
                cameras: correspondences,
            },
            meta: NoMeta,
        };
        rigs.push(rig_view);
    }
}

pub fn update_correspondes_views(
    correspondence_views: &mut Vec<Vec<Option<CorrespondenceView>>>,
    cams_imgs: &Vec<GrayImage>,
    charuco_board: &CharucoBoard,
    detection: &DetectionParams,
    dataset: &DatasetParams,
) {
    let mut correspondences = Vec::new();
    let num_cameras = cams_imgs.len();
    for cam_idx in 0..num_cameras {
        let correspondence =
            correspondence_view_from_charuco(charuco_board, &cams_imgs[cam_idx], detection)
                .filter(|c| c.points_2d.len() >= dataset.min_corners_per_view);
        correspondences.push(correspondence);
    }

    correspondence_views.push(correspondences);
}

pub fn calibrate_multiple(
    rigs: Vec<RigView<NoMeta>>,
) -> Result<RigExtrinsicsExport, Box<dyn std::error::Error>> {
    debug!("Start multiple cameras calibration");
    let num_cameras = match rigs.get(0) {
        Some(it) => it,
        None => return Err("Ошибка получения количества камер из наборов изображений".into()),
    }
    .obs
    .cameras
    .len();

    let num_frames = rigs.len();
    debug!("Камер: {num_cameras}, кадров: {num_frames}");

    if num_cameras < 2 {
        error!("Ошибка: для калибровки требуется как минимум 2 набора изображений.");
        return Err("Недостаточно камер".into());
    }

    let rig_dataset = RigExtrinsicsInput::new(rigs, num_cameras)?;

    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_input(rig_dataset)?;
    rig_extrinsics::run_calibration(&mut session)?;
    let result = session.export()?;

    // Диагностика
    for (i, cam) in result.cameras.iter().enumerate() {
        let k = cam.k.k_matrix();
        info!(
            "Камера {i}: fx={:.1} fy={:.1} cx={:.1} cy={:.1} k1={:.4} k2={:.4}",
            k[(0, 0)],
            k[(1, 1)],
            k[(0, 2)],
            k[(1, 2)],
            cam.dist.k1,
            cam.dist.k2
        );
    }
    info!("mean_reproj_error: {:.2} px", result.mean_reproj_error);

    Ok(result)
}

pub fn calibrate_multiple_with_inrinsics(
    rigs: Vec<RigView<NoMeta>>,
    intrinsics: Vec<PlanarIntrinsicsExport>,
) -> Result<RigExtrinsicsExport, Box<dyn std::error::Error>> {
    debug!("Start multiple cameras calibration");
    let num_cameras = match rigs.get(0) {
        Some(it) => it,
        None => return Err("Ошибка получения количества камер из наборов изображений".into()),
    }
    .obs
    .cameras
    .len();

    let num_frames = rigs.len();
    debug!("Камер: {num_cameras}, кадров: {num_frames}");

    if num_cameras < 2 {
        error!("Ошибка: для калибровки требуется как минимум 2 набора изображений.");
        return Err("Недостаточно камер".into());
    }

    let rig_dataset = RigExtrinsicsInput::new(rigs, num_cameras)?;
    let k_vec: Vec<FxFyCxCySkew<f64>> = intrinsics
        .iter()
        .map(|r| {
            let vision_calibration::core::IntrinsicsParams::FxFyCxCySkew { params } =
                r.params.camera.intrinsics.clone();
            params
        })
        .collect();
    let d_vec: Vec<BrownConrady5<f64>> = intrinsics
        .iter()
        .map(|r| {
            let vision_calibration::core::DistortionParams::BrownConrady5 { params } =
                r.params.camera.distortion.clone()
            else {
                panic!("Expected BrownConrady5 distortion");
            };
            params
        })
        .collect();

    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_input(rig_dataset)?;

    let mut manual = RigIntrinsicsManualInit::default();
    manual.per_cam_intrinsics = Some(k_vec);
    manual.per_cam_distortion = Some(d_vec);

    step_intrinsics_init_all_with_seed(&mut session, manual, None)?;
    step_intrinsics_optimize_all(&mut session, None)?; // опционально
    step_rig_init(&mut session)?;
    step_rig_optimize(&mut session, None)?;
    let result: RigExtrinsicsExport = session.export()?;

    Ok(result)
}

/// Сохраняет данные, необходимые для реконструкции, без огромного списка
/// остатков каждой отдельной точки. Полная диагностика может содержать больше
/// 65 536 элементов, что превышает лимит последовательности `serde_yml`.
pub fn save_calibration_to_yaml(
    path: &PathBuf,
    calibration: &RigExtrinsicsExport,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut compact = calibration.clone();
    compact.per_feature_residuals = Default::default();
    compact.image_manifest = None;
    let yaml = serde_yml::to_string(&compact)?;
    std::fs::write(path, yaml)?;
    Ok(())
}

pub fn load_calibration_from_yaml(
    path: &PathBuf,
) -> Result<RigExtrinsicsExport, Box<dyn std::error::Error>> {
    let yaml = std::fs::read_to_string(path)?;
    // Старые файлы могли быть сохранены с десятками тысяч подробных
    // per-feature residuals. Они не нужны реконструкции и не помещаются в
    // жёсткий лимит YAML-парсера, поэтому удаляем блок до десериализации.
    let compact_yaml = remove_top_level_yaml_field(&yaml, "per_feature_residuals");
    let export: RigExtrinsicsExport = serde_yml::from_str(&compact_yaml)?;
    Ok(export)
}

fn remove_top_level_yaml_field(yaml: &str, field: &str) -> String {
    let key = format!("{field}:");
    let mut output = String::with_capacity(yaml.len());
    let mut skipping = false;

    for line in yaml.split_inclusive('\n') {
        let content = line.trim_end_matches(['\r', '\n']);
        let is_top_level = !content.starts_with([' ', '\t']);

        if !skipping && is_top_level && content.trim_end() == key {
            skipping = true;
            continue;
        }

        if skipping {
            if content.is_empty() || !is_top_level {
                continue;
            }
            skipping = false;
        }

        output.push_str(line);
    }

    output
}

#[cfg(test)]
mod yaml_tests {
    use super::remove_top_level_yaml_field;

    #[test]
    fn removes_only_requested_top_level_block() {
        let yaml = "kind: rig_extrinsics\nper_feature_residuals:\n  target:\n    - error_px: 1.0\n  laser: []\nimage_manifest: null\n";
        assert_eq!(
            remove_top_level_yaml_field(yaml, "per_feature_residuals"),
            "kind: rig_extrinsics\nimage_manifest: null\n"
        );
    }

    #[test]
    fn preserves_nested_fields_with_the_same_name() {
        let yaml = "outer:\n  per_feature_residuals:\n    target: []\nnext: 1\n";
        assert_eq!(
            remove_top_level_yaml_field(yaml, "per_feature_residuals"),
            yaml
        );
    }
}
