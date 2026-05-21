use std::path::PathBuf;

use eframe::egui::{Align, Button, Layout, RichText, Ui};

use crate::app::{PipelineState, ReconstructionApp};

pub fn pick_videos_screen(app: &mut ReconstructionApp, ui: &mut Ui) {
    eframe::egui::CentralPanel::default().show_inside(ui, |ui| {
        ui.vertical_centered(|ui| {
            for vid in &app.video_paths.clone() {
                render_video_path(app, ui, vid);
            }

            if ui.button("Добавить видео").clicked() {
                select_videos(app);
            };

            let to_align_button = Button::new("Перейти к калибровке");
            if app.video_paths.len() >= 2 {
                if ui.add(to_align_button).clicked() {
                    app.state = PipelineState::ReadyToProcess;
                }
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
                app.video_paths.retain(|p| p != path);
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
        Some(p) => {
            app.video_paths = p;
        }
        None => return,
    }
}
