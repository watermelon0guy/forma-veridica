use eframe::egui::Ui;

use crate::app::{CalibrationApp, CalibrationStep};

mod align_videos_screen;
mod charuco_board_screen;
mod pick_videos_screen;

pub(crate) fn render_content(app: &mut CalibrationApp, ui: &mut Ui) {
    match app.state {
        CalibrationStep::SetupCharucoBoard => charuco_board_screen::charuco_board_screen(app, ui),
        CalibrationStep::PickVideos => pick_videos_screen::pick_videos_screen(app, ui),
        CalibrationStep::AlignVideos => align_videos_screen::align_video_screen(app, ui),
        CalibrationStep::Calibration => todo!(),
    }
}
