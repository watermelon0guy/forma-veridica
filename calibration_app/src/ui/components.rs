use crate::ui::PADDING;
use lib_cv::video::VideoPlayer;

use eframe::egui::{Align, Frame, Image, Layout, Slider, TextureHandle, Ui, Vec2};
use log::error;

pub fn render_video_card_with_buttons(
    video_player: &mut VideoPlayer,
    texture_handle: &TextureHandle,
    ui: &mut Ui,
    cell_size: Vec2,
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

pub fn _render_video_card(
    video_player: &mut VideoPlayer,
    texture_handle: &TextureHandle,
    ui: &mut Ui,
    cell_size: Vec2,
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
                    let duration_in_seconds = video_player.length_in_seconds();

                    ui.scope(|ui| {
                        // Устанавливаем ширину слайдера равной доступной ширине
                        ui.spacing_mut().slider_width = ui.available_width()
                            - ui.spacing().interact_size.x
                            - ui.spacing().item_spacing.x;

                        ui.add(Slider::new(
                            &mut video_player.current_time_in_seconds,
                            0.0..=duration_in_seconds,
                        ));
                        // if slider_response.changed() {
                        //     video_player.update_current_frame_from_time_in_seconds(
                        //         video_player.current_time_in_seconds,
                        //     );
                        //     if let Err(e) =
                        //         video_player.seek_to_time(video_player.current_time_in_seconds)
                        //     {
                        //         error!("Ошибка при поиске по времени: {e}");
                        //     }
                        // }
                    });
                });
            });
    });
}
