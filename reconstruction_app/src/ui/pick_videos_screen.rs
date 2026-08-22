use std::path::PathBuf;

use eframe::egui::{Align, Button, CentralPanel, Color32, Layout, RichText, Ui};
use lib_pipeline::config::CameraConfig;

use crate::app::{PipelineState, ReconstructionApp};

pub fn pick_videos_screen(app: &mut ReconstructionApp, ui: &mut Ui) {
    CentralPanel::default().show(ui, |ui| {
        ui.vertical_centered(|ui| {
            for cam in &app.reconstruction_config.cameras.clone() {
                render_video_path(app, ui, &cam.video_path);
            }

            if ui.button("Добавить видео").clicked() {
                select_videos(app);
            };

            let num_cameras = match app.num_cameras() {
                None => return,
                Some(num_cameras) => num_cameras,
            };

            if app.reconstruction_config.cameras.len() == num_cameras {
                let to_align_button = Button::new("Перейти к выравниванию");
                if ui.add(to_align_button).clicked() {
                    app.state = PipelineState::AlignVideos;
                }
            } else if app.reconstruction_config.cameras.len() < num_cameras {
                ui.label(
                    RichText::new(format!(
                        "Выбрано слишком мало видео: нужно {}, выбрано {}",
                        num_cameras,
                        app.reconstruction_config.cameras.len()
                    ))
                    .color(Color32::RED),
                );
            } else {
                ui.label(
                    RichText::new(format!(
                        "Выбрано слишком много видео: нужно {}, выбрано {}",
                        num_cameras,
                        app.reconstruction_config.cameras.len()
                    ))
                    .color(Color32::RED),
                );
            }
        });
    });
}

fn render_video_path(app: &mut ReconstructionApp, ui: &mut Ui, path: &PathBuf) {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("Неизвестный файл");

    ui.columns_const(|[col_1, col_2]| {
        col_1.with_layout(Layout::top_down(Align::Min), |ui| {
            ui.label(RichText::new(file_name).size(16.0))
                .on_hover_text(format!("Путь: {}", path.display()));
        });

        col_2.with_layout(Layout::top_down(Align::Max), |ui| {
            if ui.button("❌").clicked() {
                app.reconstruction_config
                    .cameras
                    .retain(|camera| &camera.video_path != path);
            }
        });
    });

    ui.separator();
}

fn select_videos(app: &mut ReconstructionApp) {
    match rfd::FileDialog::new()
        .set_title("Выбрать видео")
        .add_filter("Видео", &["mp4", "avi"])
        .pick_files()
    {
        Some(paths) => {
            app.reconstruction_config.cameras =
                paths.iter().map(|p| CameraConfig::new(p)).collect();
        }
        None => return,
    }
}
