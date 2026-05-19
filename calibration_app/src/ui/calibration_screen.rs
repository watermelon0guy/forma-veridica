use eframe::egui::Ui;
use lib_cv::calibration::{calibrate_multiple_with_charuco_from_rigs, update_rigs};
use log::{debug, error, info};
use vision_calibration::core::{NoMeta, RigView};

use crate::app::CalibrationApp;

pub fn calibration_screen(app: &mut CalibrationApp, ui: &mut Ui) {
    let mut img_rigs: Vec<RigView<NoMeta>> = Vec::new();

    for (i, player) in app.video_players.iter_mut().enumerate() {
        if let Err(e) = player.seek_to_time(app.offset_in_seconds[i]) {
            error!("Ошибка перехода к офсету: {}", e);
            return;
        }
    }

    let mut reading_vids = true;
    while reading_vids {
        let mut cams_imgs = Vec::new();
        for player in &mut app.video_players {
            debug!(
                "Кадр:{}, время: {}",
                player.current_frame(),
                player.current_time_in_seconds
            );
            cams_imgs.push(player.dynamic_image().to_luma8());
            if let Err(_) = &player.rewind_forward(20) {
                info!("Видео закончилось");
                reading_vids = false;
            };
        }
        if reading_vids {
            update_rigs(&mut img_rigs, cams_imgs, &app.charuco_board, 2, 8);
        }
    }

    match calibrate_multiple_with_charuco_from_rigs(img_rigs) {
        Ok(res) => debug!("{res:?}"),
        Err(e) => error!("Неудаяная калибровка: {e}"),
    }
    todo!()
}
