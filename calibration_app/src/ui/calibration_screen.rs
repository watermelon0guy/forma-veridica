use eframe::egui::{CentralPanel, ProgressBar, Ui};

use crate::app::CalibrationApp;

use std::sync::mpsc::TryRecvError;

pub fn calibration_screen(app: &mut CalibrationApp, ui: &mut Ui) {
    CentralPanel::default().show_inside(ui, |ui| {
        if app.calibration_thread.is_none()
            && app.calibration_result_rx.is_none()
            && app.calibration_result.is_none()
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
                Ok(Err(_)) => {
                    // При ошибке тоже чистим
                    app.calibration_thread = None;
                    app.calibration_result_rx = None;
                }
                Err(TryRecvError::Empty) => {
                    // Ещё работает - продолжаем ждать
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
            } else if let Some(ref _result) = app.calibration_result {
                // Калибровка завершена
                ui.heading("Калибровка завершена!");
                ui.add_space(30.0);

                if ui.button("Сохранить результат").clicked() {
                    if let Some(ref result) = app.calibration_result {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("yaml", &["yml", "yaml"])
                            .save_file()
                        {
                            if let Ok(yaml) = serde_yml::to_string(result) {
                                let _ = std::fs::write(&path.with_extension("yaml"), yaml);
                            }
                        }
                    }
                }
            } else {
                ui.heading("Подготовка...");
            }
        });
    });
}
