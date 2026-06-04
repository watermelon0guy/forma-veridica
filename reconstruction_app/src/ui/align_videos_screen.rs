use eframe::egui::{Align, CentralPanel, Frame, Grid, Image, Layout, ScrollArea, Ui, vec2};
use lib_cv::video::{VideoPlayer, set_color_image_to_texture_handle};
use log::error;

use crate::app::{PipelineState, ReconstructionApp};

const PADDING: f32 = 10.0;

pub fn align_video_screen(app: &mut ReconstructionApp, ui: &mut Ui) {
    if let Err(e) = app.init_videos(ui.ctx()) {
        ui.label(format!("Ошибка инициализации видео: {e}"));
        return;
    }

    if app.video_players.is_empty() {
        return;
    }
    let total = app.video_players.len();

    // Обновляем текстуры
    for (i, vp) in app.video_players.iter().enumerate() {
        set_color_image_to_texture_handle(vp.dynamic_image(), &mut app.video_texture_handles[i]);
    }

    CentralPanel::default().show_inside(ui, |ui| {
        Frame::NONE.show(ui, |ui| {
            let num_columns = ((total as f32).sqrt().ceil() as usize).min(total);
            let num_rows = (total + num_columns - 1) / num_columns;
            let cell_width = ui.available_width() / num_columns as f32 - PADDING / 2.0;
            let cell_height = ui.available_height() / num_rows as f32 - PADDING / 2.0;
            let cell_size = vec2(cell_width, cell_height);

            ScrollArea::vertical().show(ui, |ui| {
                Grid::new("recon_video_grid")
                    .spacing(vec2(PADDING, PADDING))
                    .num_columns(num_columns)
                    .min_col_width(cell_width)
                    .min_row_height(cell_height)
                    .max_col_width(cell_width)
                    .show(ui, |ui| {
                        for (i, player) in app.video_players.iter_mut().enumerate() {
                            render_video_card(player, &app.video_texture_handles[i], ui, cell_size);
                            if (i + 1) % num_columns == 0 && i + 1 < total {
                                ui.end_row();
                            }
                        }
                    });

                ui.add_space(20.0);
                if ui.button("Готово").clicked() {
                    app.offsets = app
                        .video_players
                        .iter()
                        .map(|vid| vid.current_time_in_seconds)
                        .collect();
                    app.state = PipelineState::ReadyToProcess;
                }
            });
        });
    });
}

fn render_video_card(
    video_player: &mut VideoPlayer,
    texture_handle: &eframe::egui::TextureHandle,
    ui: &mut Ui,
    cell_size: eframe::egui::Vec2,
) {
    ui.allocate_ui_with_layout(cell_size, Layout::top_down_justified(Align::Center), |ui| {
        Frame::group(ui.style())
            .outer_margin(0.0)
            .inner_margin(PADDING)
            .show(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add(Image::new(texture_handle).shrink_to_fit());
                });
                ui.horizontal(|ui| {
                    if ui.button("<<").clicked() {
                        if let Err(e) = video_player.rewind_backward(10) {
                            error!("Ошибка при перемотке назад: {e}");
                        }
                    }
                    if ui.button("<").clicked() {
                        if let Err(e) = video_player.rewind_backward(1) {
                            error!("Ошибка при перемотке назад: {e}");
                        }
                    }
                    if ui.button(">").clicked() {
                        if let Err(e) = video_player.rewind_forward(1) {
                            error!("Ошибка при перемотке вперед: {e}");
                        }
                    }
                    if ui.button(">>").clicked() {
                        if let Err(e) = video_player.rewind_forward(10) {
                            error!("Ошибка при перемотке вперед: {e}");
                        }
                    }

                    let duration = video_player.length_in_seconds();
                    ui.scope(|ui| {
                        ui.spacing_mut().slider_width = ui.available_width()
                            - ui.spacing().interact_size.x
                            - ui.spacing().item_spacing.x;

                        let slider_response = ui.add(eframe::egui::Slider::new(
                            &mut video_player.current_time_in_seconds,
                            0.0..=duration,
                        ));
                        if slider_response.changed() {
                            video_player.update_current_frame_from_time_in_seconds(
                                video_player.current_time_in_seconds,
                            );
                            if let Err(e) =
                                video_player.seek_to_time(video_player.current_time_in_seconds)
                            {
                                error!("Ошибка при поиске по времени: {e}");
                            }
                        }
                    });
                });
            });
    });
}
