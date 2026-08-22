use eframe::egui::{CentralPanel, ProgressBar, RichText, Ui};

use crate::app::CalibrationApp;
use lib_cv::calibration::save_calibration_to_yaml;

use std::sync::mpsc::TryRecvError;

pub fn calibration_screen(app: &mut CalibrationApp, ui: &mut Ui) {
    CentralPanel::default().show(ui, |ui| {
        if app.calibration_thread.is_none()
            && app.calibration_result_rx.is_none()
            && app.calibration_result.is_none()
            && app.calibration_error.is_none()
        {
            app.start_calibration_thread();
        }

        if let Some(ref rx) = app.calibration_result_rx {
            match rx.try_recv() {
                Ok(Ok(result)) => {
                    app.calibration_result = Some(result);
                    app.calibration_thread = None;
                    app.calibration_result_rx = None;
                }
                Ok(Err(err)) => {
                    log::error!("Ошибка калибровки: {err}");
                    app.calibration_error = Some(err);
                    app.calibration_thread = None;
                    app.calibration_result_rx = None;
                }
                Err(TryRecvError::Empty) => {
                    // Ещё работает
                }
                Err(TryRecvError::Disconnected) => {
                    app.calibration_thread = None;
                    app.calibration_result_rx = None;
                }
            }
        }

        ui.vertical_centered(|ui| {
            ui.add_space(50.0);

            if app.calibration_thread.is_some() || app.calibration_result_rx.is_some() {
                // Калибровка идёт
                ui.heading("Выполняется калибровка...");
                ui.add_space(30.0);

                let progress = app.calibration_progress.lock().unwrap();

                let progress_bar = ProgressBar::new(progress.percent)
                    .text(format!("{:.0}%", progress.percent * 100.0))
                    .animate(true) // анимация полоски
                    .desired_width(400.0);

                ui.add(progress_bar);
            } else if let Some(ref result) = app.calibration_result {
                // Калибровка завершена
                ui.heading("Калибровка завершена!");
                ui.add_space(30.0);

                if ui.button("Сохранить результат").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("yaml", &["yml", "yaml"])
                        .save_file()
                    {
                        let path = path.with_extension("yaml");
                        if let Err(error) = save_calibration_to_yaml(&path, result) {
                            log::error!("Не удалось сохранить калибровку: {error}");
                        }
                    }
                }
            } else if let Some(ref err) = app.calibration_error {
                ui.heading(RichText::new("Ошибка калибровки").color(eframe::egui::Color32::RED));
                ui.add_space(10.0);
                ui.label(err);
                ui.add_space(20.0);
                if ui.button("Вернуться к настройке").clicked() {
                    app.state = crate::app::CalibrationStep::SetupCharucoBoard;
                }
            } else {
                ui.heading("Подготовка...");
            }
        });
    });
}
