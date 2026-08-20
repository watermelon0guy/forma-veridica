use eframe::egui::Ui;

use crate::{
    app::{CalibrationApp, CalibrationStep},
    ui::{
        align_videos_screen::align_video_screen, calibration_screen::calibration_screen,
        charuco_board_screen::charuco_board_screen, pick_videos_screen::pick_videos_screen,
    },
};

mod advanced_params;
mod align_videos_screen;
mod calibration_screen;
mod charuco_board_screen;
mod components;
mod pick_videos_screen;

const PADDING: f32 = 10.0;

pub(crate) fn render_content(app: &mut CalibrationApp, ui: &mut Ui) {
    match app.state {
        CalibrationStep::SetupCharucoBoard => charuco_board_screen(app, ui),
        CalibrationStep::PickVideos => pick_videos_screen(app, ui),
        CalibrationStep::AlignVideos => align_video_screen(app, ui),
        CalibrationStep::Calibration => calibration_screen(app, ui),
    }
}
