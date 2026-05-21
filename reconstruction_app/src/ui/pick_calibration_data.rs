use eframe::egui::{Button, CentralPanel, RichText, Ui};
use lib_cv::calibration::load_calibration_from_yaml;
use log::error;

use crate::app::{PipelineState, ReconstructionApp};

fn pick_camera_parameters_file(
    app: &mut ReconstructionApp,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(file_path) = rfd::FileDialog::new()
        .set_title("Выбрать файл параметров")
        .pick_file()
    {
        match load_calibration_from_yaml(&file_path) {
            Ok(data) => app.calibration_data = Some(data),
            Err(e) => {
                error!("Не вышло загрузить данные калибровки: {e}");
                return Err(e);
            }
        }
    }
    Ok(())
}

pub fn pick_calibration_screen(app: &mut ReconstructionApp, ui: &mut Ui) {
    CentralPanel::default().show_inside(ui, |ui| {
        ui.vertical_centered(|ui| {
            ui.heading("Параметры камер");

            match &app.calibration_data {
                None => {
                    ui.label(RichText::new("Выберите файл с параметрами камер"));
                    let pick_button = Button::new(RichText::new("Выбрать"));

                    if ui.add(pick_button).clicked() {
                        let _ = pick_camera_parameters_file(app);
                    }
                }
                Some(calib_data) => {
                    let num_cameras = calib_data.cameras.len();
                    ui.label(format!("В параметрах найдено камер: {num_cameras}"));
                    let change_button = Button::new(RichText::new("Изменить параметры"));
                    if ui.add(change_button).clicked() {
                        let _ = pick_camera_parameters_file(app);
                    }

                    let continue_button = Button::new(RichText::new("Перейти к выбору видео"));
                    if ui.add(continue_button).clicked() {
                        app.state = PipelineState::PickVideos;
                    }
                }
            }
        });
    });
}
