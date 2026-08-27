use calib_targets::charuco::CharucoBoard;
use lib_cv::calibration::{
    calibrate_camera, calibrate_multiple_with_inrinsics, update_correspondes_views, update_rigs,
};
use lib_cv::reconstruction::{
    PointCloud, add_color_to_point_cloud, filter_matches_by_epipolar,
    filter_point_cloud_by_confidence, gather_points_2d_from_matches,
    match_with_epipolar_constraint, min_visible_match_set, save_point_cloud,
    track_points_optical_flow_all, triangulate_points_multiple, undistort_points,
};
use lib_cv::video::VideoPlayer;
use log::{error, info, trace};
use vision_calibration::core::{CorrespondenceView, NoMeta, RigView};
use vision_calibration::rig_extrinsics::RigExtrinsicsExport;

use crate::config::{CalibrationConfig, ReconstructionConfig};

/// Полный прогон калибровки по конфигу: чтение видео, детекция ChArUco,
/// покамерная калибровка интринсиков и финальная калибровка рига.
pub fn run_calibration(
    config: &CalibrationConfig,
    on_progress: &mut dyn FnMut(f32),
) -> Result<RigExtrinsicsExport, String> {
    let charuco_board = CharucoBoard::new(config.charuco_board.to_board_spec())
        .map_err(|e| format!("Неправильные данные Charuco-доски в конфиге: {e}"))?;

    let mut img_rigs: Vec<RigView<NoMeta>> = Vec::new();
    // это набор обнаруженных точек на каждой камере ОТДЕЛЬНО
    // они не сопоставлены между собой в отличии от img_rigs
    let mut correspondence_views: Vec<Vec<Option<CorrespondenceView>>> = Vec::new();

    let mut video_players: Vec<VideoPlayer> = Vec::new();
    for camera in &config.cameras {
        match VideoPlayer::new(&camera.video_path) {
            Ok(vp) => video_players.push(vp),
            Err(e) => return Err(format!("Проблема при создании плеера: {e}")),
        }
    }

    // Переходим к офсетам выбранным на шаге с выравниванием видео
    for (i, player) in video_players.iter_mut().enumerate() {
        if let Err(e) = player.seek_to_time(config.cameras[i].start_time_in_seconds) {
            error!("Ошибка перехода к офсету: {}", e);
            return Err(format!("Проблема при переходе к офсету камеры {i}: {e}"));
        }
    }

    let total_frames = video_players
        .iter()
        .map(|p| p.total_frames())
        .max()
        .unwrap_or(1);

    let mut reading_vids = true;
    while reading_vids {
        let mut cams_imgs = Vec::new();
        for player in &mut video_players {
            trace!(
                "Кадр:{}, время: {}",
                player.current_frame(),
                player.current_time_in_seconds
            );
            if reading_vids {
                on_progress(player.current_frame() as f32 / total_frames as f32);
            }
            cams_imgs.push(player.dynamic_image().to_luma8());
            if player.rewind_forward(config.frame_step).is_err() {
                info!("Видео закончилось");
                reading_vids = false;
            }
        }
        if reading_vids {
            update_rigs(
                &mut img_rigs,
                &cams_imgs,
                &charuco_board,
                &config.detection,
                &config.dataset,
            );
            update_correspondes_views(
                &mut correspondence_views,
                &cams_imgs,
                &charuco_board,
                &config.detection,
                &config.dataset,
            )
        }
    }

    // Транспонируем [frame][cam] -> [cam][frame]
    let num_cameras = config.cameras.len();
    let mut cameras_intrinsics = Vec::new();
    for cam_idx in 0..num_cameras {
        let ccv: Vec<CorrespondenceView> = correspondence_views
            .iter()
            .filter_map(|frame| frame[cam_idx].clone())
            .collect();
        if ccv.is_empty() {
            return Err(format!(
                "Для камеры {cam_idx} нет данных: все обнаружения пусты"
            ));
        }
        let intrinsic = match calibrate_camera(ccv, &config.solver) {
            Ok(it) => it,
            Err(err) => return Err(format!("Ошибка калибровки для камеры {cam_idx}: {err}")),
        };
        cameras_intrinsics.push(intrinsic);
    }

    calibrate_multiple_with_inrinsics(img_rigs, cameras_intrinsics).map_err(|e| e.to_string())
}

pub fn run_reconstruction(
    config: &ReconstructionConfig,
    calibration_data: &RigExtrinsicsExport,
    on_progress: &mut dyn FnMut(f32),
) -> Result<(), String> {
    if calibration_data.cameras.len() != config.cameras.len() {
        return Err("Кол-во камер не совпадает в данных о калибровке и в настройках реконструкции, так быть не должно...".into());
    }
    let num_cameras = calibration_data.cameras.len();

    if num_cameras < 2 {
        return Err("Требуется минимум 2 камеры".into());
    }

    let mut players: Vec<VideoPlayer> = config
        .cameras
        .iter()
        .map(|cam| VideoPlayer::new(&cam.video_path).map_err(|err| err.to_string()))
        .collect::<Result<_, _>>()?;

    // Применяем смещения времени
    for (player, cam) in players.iter_mut().zip(&config.cameras) {
        player
            .seek_to_time(cam.start_time_in_seconds)
            .map_err(|e| e.to_string())?;
    }

    // --- Первый кадр: SIFT + matching ---
    let first_frames: Vec<_> = players.iter().map(|p| p.dynamic_image().clone()).collect();

    let (all_matches, all_keypoints) = match_with_epipolar_constraint(
        &first_frames,
        &calibration_data,
        config.params.epipolar_threshold_px,
    );

    let filtered_matches = min_visible_match_set(&all_matches, all_keypoints[0].len());

    let points_2d_raw = gather_points_2d_from_matches(&filtered_matches, &all_keypoints);

    // Matching already did epipolar filtering internally,
    // go straight to undistort + triangulate
    let mut undistorted: Vec<Vec<_>> = Vec::with_capacity(num_cameras);
    for cam_i in 0..num_cameras {
        let undist = undistort_points(&points_2d_raw[cam_i], &calibration_data.cameras[cam_i]);
        undistorted.push(undist);
    }

    let (points_3d, binary_mask) =
        triangulate_points_multiple(&undistorted, &calibration_data).map_err(|e| e.to_string())?;

    let current_frame: usize = 0;
    let mut cloud = PointCloud {
        points: points_3d,
        timestamp: current_frame,
    };

    let filtered_points = apply_mask(&points_2d_raw[0], &binary_mask);

    add_color_to_point_cloud(&mut cloud, &filtered_points, &first_frames[0]);

    let before = cloud.points.len();
    filter_point_cloud_by_confidence(&mut cloud, config.params.min_confidence);
    info!(
        "Кадр 0: отфильтровано {} точек, оставлено {}",
        before - cloud.points.len(),
        cloud.points.len()
    );

    std::fs::create_dir_all(&config.output_dir).map_err(|e| e.to_string())?;
    save_point_cloud(
        &cloud,
        &config
            .output_dir
            .join(format!("frame_{current_frame:04}.ply")),
    )
    .map_err(|e| e.to_string())?;

    // --- Готовимся к optical flow ---
    let mut prev_frames = first_frames;
    // Для optical flow нужны ИСКАЖЁННЫЕ координаты (трекинг по сырым кадрам)
    let mut prev_points = points_2d_raw;

    // --- Цикл по оставшимся кадрам ---
    let total_frames = players.iter().map(|p| p.total_frames()).max().unwrap_or(1);
    let mut frame_idx: usize = 1;

    loop {
        let mut eof = false;
        for p in &mut players {
            if p.rewind_forward(config.params.frame_step).is_err() {
                eof = true;
                break;
            }
        }
        if eof {
            break;
        }

        let curr_frames: Vec<_> = players.iter().map(|p| p.dynamic_image().clone()).collect();

        let prev_gray: Vec<_> = prev_frames.iter().map(|f| f.to_luma8()).collect();
        let curr_gray: Vec<_> = curr_frames.iter().map(|f| f.to_luma8()).collect();

        let new_points_raw = track_points_optical_flow_all(
            &prev_gray,
            &curr_gray,
            &prev_points,
            config.params.lk_window,
            config.params.lk_max_iterations,
            config.params.lk_pyramid_levels,
        );

        // Undistortion до эпиполярной фильтрации
        let mut new_undistorted_raw: Vec<Vec<_>> = Vec::with_capacity(num_cameras);
        for cam_i in 0..num_cameras {
            let undist = undistort_points(&new_points_raw[cam_i], &calibration_data.cameras[cam_i]);
            new_undistorted_raw.push(undist);
        }

        // Эпиполярная фильтрация на неискажённых точках
        let (new_undistorted, good_indices) = filter_matches_by_epipolar(
            &new_undistorted_raw,
            calibration_data,
            config.params.epipolar_threshold_px,
        );

        // Фильтруем искажённые координаты для lookup цвета и для следующей итерации optical flow
        let new_points: Vec<Vec<_>> = new_points_raw
            .iter()
            .map(|cam| good_indices.iter().map(|&i| cam[i]).collect())
            .collect();

        let (points_3d, binary_mask) =
            triangulate_points_multiple(&new_undistorted, &calibration_data)
                .map_err(|e| e.to_string())?;

        let mut cloud = PointCloud {
            points: points_3d,
            timestamp: frame_idx,
        };

        let filtered_points = apply_mask(&new_points[0], &binary_mask);
        add_color_to_point_cloud(&mut cloud, &filtered_points, &curr_frames[0]);

        filter_point_cloud_by_confidence(&mut cloud, config.params.min_confidence);

        save_point_cloud(
            &cloud,
            &config.output_dir.join(format!("frame_{frame_idx:04}.ply")),
        )
        .map_err(|e| e.to_string())?;

        prev_frames = curr_frames;
        prev_points = new_points;
        frame_idx += 1;

        on_progress(players[0].current_frame() as f32 / total_frames as f32);

        if players[0].current_frame() >= total_frames.saturating_sub(config.params.frame_step) {
            break;
        }
    }

    info!("Пайплайн завершён. Обработано {} кадров", frame_idx);
    Ok(())
}

fn apply_mask<T: Copy>(list: &[T], mask: &[bool]) -> Vec<T> {
    list.iter()
        .enumerate()
        .filter(|(i, _)| mask[*i])
        .map(|(_, p)| *p)
        .collect()
}
