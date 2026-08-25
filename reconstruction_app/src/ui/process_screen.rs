use std::sync::mpsc::TryRecvError;

use eframe::egui::{CentralPanel, ProgressBar, Ui};

use crate::app::ReconstructionApp;

pub fn process_screen(app: &mut ReconstructionApp, ui: &mut Ui) {
    CentralPanel::default().show(ui, |ui| {
        // Запускаем поток, если ещё не запущен
        if app.pipeline_thread.is_none()
            && app.pipeline_result_rx.is_none()
            && app.pipeline_result.is_none()
        {
            app.start_pipeline_thread();
        }

        // Проверяем результат
        if let Some(ref rx) = app.pipeline_result_rx {
            match rx.try_recv() {
                Ok(Ok(())) => {
                    app.pipeline_result = Some(Ok(()));
                    app.pipeline_thread = None;
                    app.pipeline_result_rx = None;
                }
                Ok(Err(e)) => {
                    app.pipeline_result = Some(Err(e));
                    app.pipeline_thread = None;
                    app.pipeline_result_rx = None;
                }
                Err(TryRecvError::Empty) => {
                    // Ещё работает
                }
                Err(TryRecvError::Disconnected) => {
                    app.pipeline_thread = None;
                    app.pipeline_result_rx = None;
                }
            }
        }

        ui.vertical_centered(|ui| {
            ui.add_space(50.0);

            if app.pipeline_thread.is_some() || app.pipeline_result_rx.is_some() {
                ui.heading("Выполняется реконструкция...");
                ui.add_space(20.0);

                let progress = app.reconstruction_progress.lock().unwrap();

                let progress_bar = ProgressBar::new(progress.percent)
                    .text(format!("{:.0}%", progress.percent * 100.0))
                    .animate(true)
                    .desired_width(400.0);
                ui.add(progress_bar);
            } else if let Some(ref result) = app.pipeline_result {
                match result {
                    Ok(()) => {
                        ui.heading("Реконструкция завершена!");
                        ui.add_space(10.0);
                        ui.label("Облака точек сохранены в папку point_clouds/");
                    }
                    Err(e) => {
                        ui.heading("Ошибка реконструкции");
                        ui.add_space(10.0);
                        ui.colored_label(eframe::egui::Color32::RED, e);
                    }
                }
            } else {
                ui.heading("Подготовка...");
            }
        });
    });
}
