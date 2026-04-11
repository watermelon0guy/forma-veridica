use calib_targets::{
    charuco::{CharucoBoard, CharucoDetectionResult, CharucoParams},
    detect::{self},
};
use image::GrayImage;
use log::{debug, error, warn};
use nalgebra::{Point2, Point3};
use rayon::prelude::*;
use vision_calibration::{
    core::{CorrespondenceView, NoMeta, PlanarDataset, RigView, RigViewObs, View},
    optim::{PlanarIntrinsicsEstimate, RigExtrinsicsDataset},
    planar_intrinsics::{FilterOptions, run_calibration_with_filtering},
    prelude::{PlanarIntrinsicsProblem, RigExtrinsicsProblem, run_rig_extrinsics},
    rig_extrinsics::RigExtrinsicsExport,
    session::CalibrationSession,
};

pub fn get_charuco(
    charuco_board: &CharucoBoard,
    // detector_params: CharucoParams,
    img: &GrayImage,
) -> Option<CharucoDetectionResult> {
    let mut detector_params = CharucoParams::for_board(&charuco_board.spec());
    detector_params.chessboard.min_corners = 6; // вместо 32
    detector_params.min_marker_inliers = 2; // вместо 8
    detector_params.min_secondary_marker_inliers = 1;
    detector_params.chessboard.graph.min_spacing_pix = 1.0;
    detector_params.chessboard.graph.max_spacing_pix = 300.0;
    detector_params.max_hamming = 4;
    // detector_params.scan.min_border_score = 0.5;
    // detector_params.scan.inset_frac = 0.03;

    let px_values: Vec<f32> = (30..=200).step_by(20).map(|x| x as f32).collect();

    let result = px_values.into_par_iter().find_map_any(|px| {
        let mut params = detector_params.clone();
        params.px_per_square = px;

        detect::detect_charuco(img, &params).ok()
    });

    result
}

pub fn calibrate_with_charuco(
    imgs: &Vec<GrayImage>,
    charuco_board: &CharucoBoard,
) -> Result<PlanarIntrinsicsEstimate, Box<dyn std::error::Error>> {
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    let mut views = Vec::new();
    for img in imgs {
        if let Some(correspondence_view) = correspondence_view_from_charuco(charuco_board, img) {
            views.push(View::without_meta(correspondence_view));
        };
    }

    let dataset = PlanarDataset::new(views)?;
    let _ = session.set_input(dataset);

    let filter_option = FilterOptions::default();
    let _ = run_calibration_with_filtering(&mut session, filter_option);

    let intrinsics = session.export()?;

    Ok(intrinsics)
}

fn correspondence_view_from_charuco(
    charuco_board: &CharucoBoard,
    img: &image::ImageBuffer<image::Luma<u8>, Vec<u8>>,
) -> Option<CorrespondenceView> {
    let charuco_detection = match get_charuco(charuco_board, img) {
        Some(charuco_det) => charuco_det,
        None => {
            warn!("Error in charuco detection");
            return None;
        }
    };
    if charuco_detection.detection.corners.is_empty() {
        return None;
    }
    let mut points_3d: Vec<Point3<f64>> = Vec::new();
    let mut points_2d: Vec<Point2<f64>> = Vec::new();
    for corner in &charuco_detection.detection.corners {
        if let Some(target_position) = corner.target_position {
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
    }
    match CorrespondenceView::new(points_3d, points_2d) {
        Ok(view) => return Some(view),
        Err(e) => {
            warn!("Error creating correspondence view: {e}");
            return None;
        }
    };
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

    let rig_dataset = RigExtrinsicsDataset::new(rigs, num_cameras)?;

    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_input(rig_dataset)?;
    run_rig_extrinsics(&mut session)?;
    let result = session.export()?;

    Ok(result)
}

pub fn update_rigs(
    rigs: &mut Vec<RigView<NoMeta>>,
    cams_imgs: Vec<GrayImage>,
    charuco_board: &CharucoBoard,
) {
    let mut correspondences = Vec::new();
    let num_cameras = cams_imgs.len();
    for cam_idx in 0..num_cameras {
        correspondences.push(correspondence_view_from_charuco(
            charuco_board,
            &cams_imgs[cam_idx],
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
    debug!(
        "Количество камер изображений для калибровки: {}.",
        num_cameras
    );
    debug!(
        "Количество наборов изображений для калибровки: {}.",
        num_frames
    );

    if num_cameras < 2 {
        error!("Ошибка: для калибровки требуется как минимум 2 набора изображений.");
        return Err("Недостаточно камер".into());
    }

    let rig_dataset = RigExtrinsicsDataset::new(rigs, num_cameras)?;

    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_input(rig_dataset)?;
    run_rig_extrinsics(&mut session)?;
    let result = session.export()?;

    Ok(result)
}
/*

// Функция для нахождения общих точек
pub fn find_common_points(frames: &[Vector<i32>]) -> HashSet<i32> {
    if frames.is_empty() {
        return HashSet::new();
    }

    // Первый набор - копируем значения
    let mut common_ids: HashSet<i32> = frames.get(0).unwrap().iter().collect();

    for frame in frames.iter().skip(1) {
        // Временный HashSet для сравнения
        let current_ids: HashSet<_> = frame.iter().collect();
        common_ids = common_ids.intersection(&current_ids).cloned().collect();
    }

    common_ids
}

pub fn perform_calibration(
    image_path: &str,
    cameras_params_path: &Path,
    charuco_board: &CharucoBoard,
    num_cameras: usize,
) {
    debug!("Поиск калибровочных изображений в: {}", image_path);

    // Собираем все файлы в директории
    let dir_entries = match fs::read_dir(image_path) {
        Ok(entries) => entries,
        Err(e) => {
            error!("Ошибка чтения директории: {}", e);
            return;
        }
    };

    // Группируем изображения по камерам и кадрам
    let mut frame_numbers = Vec::new();
    let mut camera_images: Vec<Vector<Mat>> = vec![Vector::<Mat>::new(); num_cameras];

    for entry in dir_entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        debug!("Загружаю {}", file_name);

        if file_name.starts_with("img_") && file_name.ends_with(".png") {
            let parts: Vec<&str> = file_name.split('_').collect();
            if parts.len() == 3 {
                if let Ok(cam_num) = parts[1].parse::<usize>() {
                    if let Ok(frame_num) = parts[2].trim_end_matches(".png").parse::<usize>() {
                        if let Ok(img) = imread(&entry.path().to_string_lossy(), IMREAD_COLOR) {
                            camera_images[cam_num - 1].push(img);
                            frame_numbers.push(frame_num);
                        }
                    }
                }
            }
        }
    }

    // Удаляем дубликаты frame_numbers и сортируем
    frame_numbers.sort();
    frame_numbers.dedup();

    info!("Найдено {} наборов(сцен) изображений", frame_numbers.len());

    // Выполняем калибровку
    match calibrate_multiple_with_charuco(&camera_images, charuco_board) {
        Ok(cameras) => {
            info!(
                "Калибровка успешно завершена. Получено {} камер:",
                cameras.len()
            );
            for (i, cam) in cameras.iter().enumerate() {
                if i > 0 {
                    debug!(
                        "Дистанция от основной камеры: {:.2} мм",
                        norm(&cam.translation, NORM_L2, &Mat::default()).unwrap()
                    );
                }
            }

            // Сохранение параметров в файл (опционально)
            if let Err(e) = save_camera_parameters(
                &cameras,
                &format!(
                    "{}/calibration_params.yml",
                    cameras_params_path.to_str().unwrap()
                ),
            ) {
                error!("Ошибка при сохранении параметров: {}", e);
            }
        }
        Err(e) => error!("Ошибка при калибровке: {:?}", e),
    }
}

fn save_camera_parameters(cameras: &[CameraParameters], path: &str) -> opencv::Result<()> {
    let mut fs = FileStorage::new(path, FileStorage_Mode::WRITE as i32, "")?;

    for (i, cam) in cameras.iter().enumerate() {
        // Для матриц используем специальные методы записи
        fs.write_mat(&format!("camera_{}_intrinsic", i), &cam.intrinsic)?;
        fs.write_mat(&format!("camera_{}_distortion", i), &cam.distortion)?;

        if i > 0 {
            fs.write_mat(&format!("camera_{}_rotation", i), &cam.rotation)?;
            fs.write_mat(&format!("camera_{}_translation", i), &cam.translation)?;
        }
    }

    fs.release()?;
    Ok(())
}

pub fn load_camera_parameters(path: &str) -> opencv::Result<Vec<CameraParameters>> {
    let mut fs = FileStorage::new(path, FileStorage_Mode::READ as i32, "")?;

    let mut cameras = Vec::new();
    let mut i = 0;

    loop {
        let intrinsic_name = format!("camera_{}_intrinsic", i);
        debug!("Попытка считать данные для камеры {}", i);
        if fs.get_node(&intrinsic_name)?.empty()? {
            break;
        }

        let mut cam_params = CameraParameters::new()?;

        cam_params.intrinsic = fs.get_node(&intrinsic_name)?.mat()?;
        cam_params.distortion = fs.get_node(&format!("camera_{}_distortion", i))?.mat()?;

        if i > 0 {
            cam_params.rotation = fs.get_node(&format!("camera_{}_rotation", i))?.mat()?;
            cam_params.translation = fs.get_node(&format!("camera_{}_translation", i))?.mat()?;
        }

        cameras.push(cam_params);
        i += 1;
    }

    fs.release()?;

    if cameras.is_empty() {
        return Err(opencv::Error::new(
            opencv::core::StsError as i32,
            "Не удалось загрузить параметры ни одной камеры".to_string(),
        ));
    }

    Ok(cameras)
}
 */
