use std::{ops::RangeInclusive, path::PathBuf};

use calib_targets::{
    aruco::builtins::{BUILTIN_DICTIONARY_NAMES, builtin_dictionary},
    printable::{PageOrientation, PageSize, PageSpec},
};
use eframe::egui::{
    Align, Button, ComboBox, Frame, Grid, Image, Layout, Panel, RichText, Slider, SliderClamping,
    TextureHandle, TextureOptions, Ui, Vec2, vec2,
};
use image::DynamicImage;
use lib_cv::{
    calibration::{charuco::dump_board_info, get_charuco_grid_first, get_charuco_marker_first},
    utils::draw_charuco_detection,
};
use log::{error, warn};

use crate::{
    app::{
        CalibrationApp, CalibrationStep, FrameWithCharucoData, charuco_target_spec_to_dynamic_image,
    },
    video::{VideoPlayer, dynamic_image_to_color_image, set_color_image_to_texture_handle},
};

const PADDING: f32 = 10.0;

pub(crate) fn render_content(app: &mut CalibrationApp, ui: &mut Ui) {
    match app.state {
        CalibrationStep::SetupCharucoBoard => charuco_board_screen(app, ui),
        CalibrationStep::PickVideos => pick_videos_screen(app, ui),
        CalibrationStep::AlignVideos => align_video_screen(app, ui),
        CalibrationStep::Calibration => todo!(),
    }
}

fn charuco_board_screen(app: &mut CalibrationApp, ui: &mut Ui) {
    let dict_len = app.charuco_target_spec.dictionary.codes.len() as u32;
    Panel::left("parameters").show_inside(ui, |ui| {
        ui.add(
            Slider::new(
                &mut app.charuco_target_spec.cols,
                RangeInclusive::new(1, dict_len * 2 / app.charuco_target_spec.rows),
            )
            .update_while_editing(false)
            .text("Столбцы")
            .clamping(SliderClamping::Always),
        );
        ui.add(
            Slider::new(
                &mut app.charuco_target_spec.rows,
                RangeInclusive::new(1, dict_len * 2 / app.charuco_target_spec.cols),
            )
            .update_while_editing(false)
            .text("Строчки")
            .clamping(SliderClamping::Always),
        );
        ui.add(
            Slider::new(
                &mut app.charuco_target_spec.marker_size_rel,
                RangeInclusive::new(0.01, 0.99),
            )
            .update_while_editing(false)
            .text("Размер маркера")
            .clamping(SliderClamping::Always),
        );
        ui.add(
            Slider::new(
                &mut app.charuco_target_spec.square_size_mm,
                RangeInclusive::new(10.0, 100.0),
            )
            .update_while_editing(false)
            .text("Размер квадрата")
            .clamping(SliderClamping::Always),
        );
        ComboBox::from_label("Наборы маркеров")
            .selected_text(app.charuco_target_spec.dictionary.name)
            .show_ui(ui, |ui| {
                for d in BUILTIN_DICTIONARY_NAMES {
                    let dict_name = d.to_string();
                    if ui
                        .selectable_label(
                            app.charuco_target_spec.dictionary.name == dict_name,
                            &dict_name,
                        )
                        .clicked()
                    {
                        if let Some(new_dict) = builtin_dictionary(d) {
                            app.charuco_target_spec.dictionary = new_dict;
                        }
                    }
                }
            });

        if ui.button("Сохранить паттерн").clicked() {
            if let Ok(()) = app.update_board_from_spec() {
                app.state = CalibrationStep::PickVideos;

                dump_board_info(&app.charuco_board);
            }
        }
    });

    let page_margin_mm = 10.0;
    let page_size = PageSize::Custom {
        width_mm: app.charuco_target_spec.square_size_mm * app.charuco_target_spec.cols as f64
            + page_margin_mm * 2.0,
        height_mm: app.charuco_target_spec.square_size_mm * app.charuco_target_spec.rows as f64
            + page_margin_mm * 2.0,
    };

    let page_spec = PageSpec {
        size: page_size,
        orientation: PageOrientation::Portrait,
        margin_mm: page_margin_mm,
    };

    eframe::egui::CentralPanel::default().show_inside(ui, |ui| {
        match &mut app.charuco_board_texture_handle {
            Some(texture) => {
                let image =
                    charuco_target_spec_to_dynamic_image(&app.charuco_target_spec, 60, page_spec)
                        .unwrap_or(DynamicImage::default());
                set_color_image_to_texture_handle(&image, texture);
                let texture_ref = &*texture;
                ui.centered_and_justified(|ui| {
                    ui.add(eframe::egui::Image::from_texture(texture_ref).shrink_to_fit())
                });
            }
            None => {
                let image =
                    charuco_target_spec_to_dynamic_image(&app.charuco_target_spec, 60, page_spec)
                        .unwrap_or(DynamicImage::default());
                app.charuco_board_texture_handle = Some(ui.ctx().load_texture(
                    "charuco_board",
                    dynamic_image_to_color_image(&image),
                    TextureOptions::default(),
                ));
            }
        }
    });
}

fn align_video_screen(app: &mut CalibrationApp, ui: &mut Ui) {
    if let Err(e) = app.init_videos(ui.ctx()) {
        ui.label(format!("Ошибка инициализации видео: {e}"));
        return;
    };

    if app.video_players.is_empty() {
        return;
    }
    let total = app.video_players.len();

    for (i, vp) in app.video_players.iter().enumerate() {
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
            let rgba_img = vp.dynamic_image().to_rgba8();
            let detection_result = get_charuco_marker_first(&app.charuco_board, &rgba_img);

            log::debug!(
                "Видео {}: размер кадра {}x{}, grey {}x{}, aspect_ratio={:.2}",
                i,
                vp.dynamic_image().width(),
                vp.dynamic_image().height(),
                vp.dynamic_image().width(),
                vp.dynamic_image().height(),
                vp.dynamic_image().width() as f32 / vp.dynamic_image().height() as f32
            );

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
                    render_video_card(player, &app.video_texture_handles[i], ui, cell_size);
                    if (i + 1) % num_columns == 0 && i + 1 < total {
                        ui.end_row();
                    }
                }
            });
    });
    if ui.button("Начать калибровку").clicked() {
        app.offset_in_seconds = app
            .video_players
            .iter()
            .map(|vid| vid.current_time_in_seconds)
            .collect();

        app.state = CalibrationStep::Calibration;
    }
}

fn render_video_card(
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

fn render_video_path(app: &mut CalibrationApp, ui: &mut Ui, path: &PathBuf) {
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

fn pick_videos_screen(app: &mut CalibrationApp, ui: &mut Ui) {
    ui.vertical_centered(|ui| {
        if app.num_cameras() == 0 {
            ui.label("Выберите видео калибровок, чтобы начать");
        }

        for vid in &app.video_paths.clone() {
            render_video_path(app, ui, vid);
        }

        if ui.button("Добавить видео").clicked() {
            select_videos(app);
        };

        let to_align_button = Button::new("Перейти к синхронизации видео");
        if app.video_paths.len() >= 2 {
            if ui.add(to_align_button).clicked() {
                app.state = CalibrationStep::AlignVideos;
            }
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
