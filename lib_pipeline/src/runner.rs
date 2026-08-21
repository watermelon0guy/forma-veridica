use std::path::Path;

use calib_targets::charuco::CharucoBoard;
use lib_cv::calibration::{
    calibrate_camera, calibrate_multiple_with_inrinsics, update_correspondes_views, update_rigs,
};
use lib_cv::video::VideoPlayer;
use log::{debug, error, info};
use vision_calibration::core::{CorrespondenceView, NoMeta, RigView};
use vision_calibration::rig_extrinsics::RigExtrinsicsExport;

use crate::config::CalibrationConfig;

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
            debug!(
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

/// Сохранение результата калибровки в YAML по пути из конфига.
pub fn save_result(
    result: &RigExtrinsicsExport,
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let yaml = serde_yml::to_string(result)?;
    std::fs::write(output_path, yaml)?;
    Ok(())
}
