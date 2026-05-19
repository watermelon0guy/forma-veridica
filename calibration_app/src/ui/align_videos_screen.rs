use eframe::egui::{
    Align, CentralPanel, Frame, Grid, Image, Layout, ScrollArea, Slider, TextureHandle, Ui, Vec2,
    vec2,
};
use lib_cv::{calibration::get_charuco_marker_first, utils::draw_charuco_detection};
use log::{error, warn};

use crate::{
    app::{CalibrationApp, CalibrationStep, FrameWithCharucoData},
    ui::{PADDING, components::render_video_card_with_buttons},
    video::{VideoPlayer, set_color_image_to_texture_handle},
};

pub fn align_video_screen(app: &mut CalibrationApp, ui: &mut Ui) {
    if let Err(e) = app.init_videos(ui.ctx()) {
        ui.label(format!("Ошибка инициализации видео: {e}"));
        return;
    };

    if app.video_players.is_empty() {
        return;
    }
    let total = app.video_players.len();

    for (i, vp) in app.video_players.iter().enumerate() {
        if !app.draw_charuco_results {
            set_color_image_to_texture_handle(
                vp.dynamic_image(),
                &mut app.video_texture_handles[i],
            );
            continue;
        }

        let should_detect: bool = match &app.last_detected_frame_with_charuco[i] {
            Some(cached) if cached.frame == vp.current_frame() => match &cached.charuco_data {
                Some(detection_res) => {
                    let img_with_charuco =
                        draw_charuco_detection(vp.dynamic_image(), &detection_res);
                    set_color_image_to_texture_handle(
                        &img_with_charuco,
                        &mut app.video_texture_handles[i],
                    );
                    continue;
                }
                None => {
                    set_color_image_to_texture_handle(
                        vp.dynamic_image(),
                        &mut app.video_texture_handles[i],
                    );
                    continue;
                }
            },
            _ => true,
        };

        if should_detect {
            let gray_img = vp.dynamic_image().to_luma8();
            let detection_result = get_charuco_marker_first(&app.charuco_board, &gray_img);

            match &detection_result {
                Some(detection_res) => {
                    let img_with_charuco =
                        draw_charuco_detection(vp.dynamic_image(), &detection_res);
                    set_color_image_to_texture_handle(
                        &img_with_charuco,
                        &mut app.video_texture_handles[i],
                    );
                }
                None => {
                    set_color_image_to_texture_handle(
                        vp.dynamic_image(),
                        &mut app.video_texture_handles[i],
                    );
                    warn!("Ошибка при обнаружении Charuco в видео {}", i);
                }
            }

            app.last_detected_frame_with_charuco[i] = Some(FrameWithCharucoData {
                frame: vp.current_frame(),
                charuco_data: detection_result,
            });
        }
    }
    CentralPanel::default().show_inside(ui, |ui| {
        Frame::NONE.show(ui, |ui| {
            let num_columns = ((total as f32).sqrt().ceil() as usize).min(total);
            let num_rows = (total + num_columns - 1) / num_columns;
            let cell_width = ui.available_width() / num_columns as f32 - PADDING / 2.0;
            let cell_height = ui.available_height() / num_rows as f32 - PADDING / 2.0;
            let cell_size = vec2(cell_width, cell_height);
            ScrollArea::vertical().show(ui, |ui| {
                Grid::new("video_grid")
                    .spacing(vec2(PADDING, PADDING))
                    .num_columns(num_columns)
                    .min_col_width(cell_width)
                    .min_row_height(cell_height)
                    .max_col_width(cell_width)
                    .show(ui, |ui| {
                        for (i, player) in app.video_players.iter_mut().enumerate() {
                            render_video_card_with_buttons(
                                player,
                                &app.video_texture_handles[i],
                                ui,
                                cell_size,
                            );
                            if (i + 1) % num_columns == 0 && i + 1 < total {
                                ui.end_row();
                            }
                        }
                    });
                ui.checkbox(&mut app.draw_charuco_results, "Рисовать ChAruco маркеры");
                if ui.button("Начать калибровку").clicked() {
                    app.offset_in_seconds = app
                        .video_players
                        .iter()
                        .map(|vid| vid.current_time_in_seconds)
                        .collect();

                    app.state = CalibrationStep::Calibration;
                }
            });
        });
    });
}
