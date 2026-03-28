use std::path::PathBuf;

use eframe::egui::{
    Align, CentralPanel, Context, Frame, Grid, Image, Layout, RichText, Slider, Style, Ui, Vec2,
    vec2,
};
use log::error;
use video_rs::Time;

use crate::{
    app::{CalibrationApp, CalibrationStep},
    video::VideoPlayer,
};

const PADDING: f32 = 10.0;

pub(crate) fn render_content(app: &mut CalibrationApp, ctx: &Context) {
    CentralPanel::default()
        // .frame(Frame::NONE)
        .frame(Frame::central_panel(&Style::default()))
        .show(ctx, |ui| match app.state {
            CalibrationStep::PickVideos => pick_videos_screen(app, ctx, ui),
            CalibrationStep::AlignVideos => align_video_screen(app, ctx, ui),
            CalibrationStep::Calibration => todo!(),
        });
}

fn align_video_screen(app: &mut CalibrationApp, ctx: &Context, ui: &mut Ui) {
    if let Err(e) = app.init_videos(ctx) {
        ui.label(format!("Ошибка инициализации видео: {e}"));
        return;
    };

    if app.video_players.is_empty() {
        return;
    }
    let total = app.video_players.len();

    Frame::NONE.show(ui, |ui| {
        let num_columns = ((total as f32).sqrt().ceil() as usize).min(total);
        let num_rows = (total + num_columns - 1) / num_columns;
        let cell_width = ui.available_width() / num_columns as f32 - PADDING / 2.0;
        let cell_height = ui.available_height() / num_rows as f32 - PADDING / 2.0;
        let cell_size = vec2(cell_width, cell_height);
        Grid::new("video_grid")
            .spacing(vec2(PADDING, PADDING))
            .num_columns(num_columns)
            .min_col_width(cell_width)
            .min_row_height(cell_height)
            .max_col_width(cell_width)
            .show(ui, |ui| {
                for (i, player) in app.video_players.iter_mut().enumerate() {
                    render_video_card(player, ctx, ui, cell_size);
                    if (i + 1) % num_columns == 0 && i + 1 < total {
                        ui.end_row();
                    }
                }
            });
    });
}

fn render_video_card(video_player: &mut VideoPlayer, ctx: &Context, ui: &mut Ui, cell_size: Vec2) {
    ui.allocate_ui_with_layout(cell_size, Layout::top_down_justified(Align::Center), |ui| {
        Frame::group(ui.style())
            .outer_margin(0.0)
            .inner_margin(PADDING)
            .show(ui, |ui| {
                ui.vertical_centered(|ui| {
                    if let Some(texture) = video_player.texture() {
                        ui.add(Image::new(texture).shrink_to_fit());
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("<<").clicked() {
                        if let Err(e) = video_player.rewind_backward(ctx, 10) {
                            error!("Ошибка при перемотке назад: {e}");
                        }
                    }
                    if ui.button("<").clicked() {
                        if let Err(e) = video_player.rewind_backward(ctx, 1) {
                            error!("Ошибка при перемотке назад: {e}");
                        }
                    }
                    if ui.button(">").clicked() {
                        if let Err(e) = video_player.rewind_forward(ctx, 1) {
                            error!("Ошибка при перемотке вперед: {e}");
                        }
                    }
                    if ui.button(">>").clicked() {
                        if let Err(e) = video_player.rewind_forward(ctx, 10) {
                            error!("Ошибка при перемотке вперед: {e}");
                        }
                    }

                    let duration_in_seconds = video_player.length_in_seconds();

                    ui.scope(|ui| {
                        // Устанавливаем ширину слайдера равной доступной ширине
                        ui.spacing_mut().slider_width = ui.available_width()
                            - ui.spacing().interact_size.x
                            - ui.spacing().item_spacing.x;

                        let slider_response = ui.add(Slider::new(
                            &mut video_player.current_time_in_seconds,
                            0.0..=duration_in_seconds,
                        ));
                        if slider_response.changed() {
                            video_player.update_current_frame_from_time_in_seconds(
                                video_player.current_time_in_seconds,
                            );
                            if let Err(e) =
                                video_player.seek_to_time(ctx, video_player.current_time_in_seconds)
                            {
                                error!("Ошибка при поиске по времени: {e}");
                            }
                        }
                    });
                });
            });
    });
}

fn render_video_path(app: &mut CalibrationApp, _ctx: &Context, ui: &mut Ui, path: &PathBuf) {
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

fn pick_videos_screen(app: &mut CalibrationApp, ctx: &Context, ui: &mut Ui) {
    ui.vertical_centered(|ui| {
        if app.num_cameras() == 0 {
            ui.label("Выберите видео калибровок, чтобы начать");
        }

        for vid in &app.video_paths.clone() {
            render_video_path(app, ctx, ui, vid);
        }

        if ui.button("Добавить видео").clicked() {
            select_videos(app);
        };
        if ui.button("Перейти к синхронизации видео").clicked() {
            app.state = CalibrationStep::AlignVideos;
        }
    });
}

fn select_videos(app: &mut CalibrationApp) {
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
