use std::path::PathBuf;

use calib_targets::{
    GridCoords,
    aruco::MarkerDetection,
    charuco::{CharucoBoard, CharucoCorner, CharucoDetectionResult, CharucoParams},
    core::GridAlignment,
    detect::detect_charuco_best,
};
use image::GrayImage;
use log::{debug, error, info, warn};
use nalgebra::{Point2, Point3};
use vision_calibration::{
    core::{CorrespondenceView, NoMeta, PlanarDataset, RigView, RigViewObs, View},
    planar_intrinsics::{self, FilterOptions, PlanarIntrinsicsExport},
    prelude::PlanarIntrinsicsProblem,
    rig_extrinsics::{
        self, RigExtrinsicsExport, RigExtrinsicsInput, RigExtrinsicsProblem, run_calibration,
    },
    session::CalibrationSession,
};

use crate::calibration::charuco::{
    build_marker_homographies, detect_aruco_markers, interpolate_charuco_corners,
};

pub mod charuco;

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
) -> Option<CharucoDetectionResult> {
    let markers = detect_aruco_markers(img, &charuco_board.spec().dictionary);

    let transforms = build_marker_homographies(&charuco_board, &markers);
    if transforms.is_empty() {
        return None;
    }
    let corners = interpolate_charuco_corners(charuco_board, &transforms, 2);
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
    // Строим обратный маппинг corner_id -> (i, j) для GridCoords
    let mut id_to_grid: std::collections::HashMap<u32, GridCoords> =
        std::collections::HashMap::new();
    for i in 0..board.expected_inner_rows() as i32 {
        for j in 0..board.expected_inner_cols() as i32 {
            if let Some(cid) = board.charuco_corner_id_from_board_corner(i, j) {
                id_to_grid.insert(cid, GridCoords { i, j });
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

pub fn calibrate_with_charuco(
    imgs: &Vec<GrayImage>,
    charuco_board: &CharucoBoard,
) -> Result<PlanarIntrinsicsExport, Box<dyn std::error::Error>> {
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    let mut views = Vec::new();
    for img in imgs {
        if let Some(correspondence_view) = correspondence_view_from_charuco(charuco_board, img) {
            views.push(View::without_meta(correspondence_view));
        };
    }

    let dataset = PlanarDataset::new(views)?;
    let _ = session.set_input(dataset);

    let mut filter_option = FilterOptions::default();
    filter_option.max_reproj_error = 2.0; // почти не фильтруем при плохой начальной калибровке
    let _ = planar_intrinsics::run_calibration_with_filtering(&mut session, filter_option);

    let intrinsics = session.export()?;

    Ok(intrinsics)
}

fn correspondence_view_from_charuco(
    charuco_board: &CharucoBoard,
    img: &image::ImageBuffer<image::Luma<u8>, Vec<u8>>,
) -> Option<CorrespondenceView> {
    let charuco_detection = match get_charuco_marker_first(charuco_board, img) {
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

pub fn calibrate_multiple_with_charuco_from_images(
    imgs_sets: &Vec<Vec<GrayImage>>,
    charuco_board: &CharucoBoard,
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
    cams_imgs: Vec<GrayImage>,
    charuco_board: &CharucoBoard,
    min_correspondences: usize,
    min_point_in_correspondence: usize,
) {
    let mut correspondences = Vec::new();
    let num_cameras = cams_imgs.len();
    for cam_idx in 0..num_cameras {
        correspondences.push(correspondence_view_from_charuco(
            charuco_board,
            &cams_imgs[cam_idx],
        ));
    }

    for opt_cv in &correspondences {
        if let Some(cv) = opt_cv {
            if cv.len() < min_point_in_correspondence {
                return;
            }
        }
    }

    // Добавляем риг только если есть данные минимум с 2 камер
    let valid_cams = correspondences.iter().filter(|c| c.is_some()).count();
    if valid_cams >= min_correspondences {
        let rig_view = RigView {
            obs: RigViewObs {
                cameras: correspondences,
            },
            meta: NoMeta,
        };
        rigs.push(rig_view);
    }
}

pub fn calibrate_multiple_with_charuco_from_rigs(
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

pub fn load_calibration_from_yaml(
    path: &PathBuf,
) -> Result<RigExtrinsicsExport, Box<dyn std::error::Error>> {
    let yaml_str = std::fs::read_to_string(path)?;
    let export: RigExtrinsicsExport = serde_yml::from_str(&yaml_str)?;
    Ok(export)
}
