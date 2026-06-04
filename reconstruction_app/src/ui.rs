use crate::{
    app::{PipelineState, ReconstructionApp},
    ui::pick_calibration_data::pick_calibration_screen,
    ui::pick_videos_screen::pick_videos_screen,
    ui::process_screen::process_screen,
};
use eframe::egui::Ui;
mod pick_calibration_data;
mod pick_videos_screen;
mod process_screen;

pub(crate) fn render_content(app: &mut ReconstructionApp, ui: &mut Ui) {
    match app.state {
        PipelineState::PickCalibrationData => pick_calibration_screen(app, ui),
        PipelineState::PickVideos => pick_videos_screen(app, ui),
        PipelineState::ReadyToProcess => process_screen(app, ui),
    }
}
