use std::{ops::RangeInclusive, path::PathBuf};

use calib_targets::{
    aruco::builtins::{BUILTIN_DICTIONARY_NAMES, builtin_dictionary},
    printable::{PageOrientation, PageSize, PageSpec},
};
use eframe::egui::{
    Align, CentralPanel, ComboBox, Context, Frame, Grid, Image, Layout, RichText, SidePanel,
    Slider, SliderClamping, Style, TextureHandle, TextureOptions, Ui, Vec2, vec2,
};
use image::DynamicImage;
use log::error;

use crate::{
    app::{CalibrationApp, CalibrationStep, charuco_target_spec_to_dynamic_image},
    video::{VideoPlayer, dynamic_image_to_color_image, set_color_image_to_texture_handle},
};

const PADDING: f32 = 10.0;

pub(crate) fn render_content(app: &mut CalibrationApp, ctx: &Context) {
    CentralPanel::default()
        // .frame(Frame::NONE)
        .frame(Frame::central_panel(&Style::default()))
        .show(ctx, |ui| match app.state {
            CalibrationStep::SetupCharucoBoard => charuco_board_screen(app, ctx, ui),
            CalibrationStep::PickVideos => pick_videos_screen(app, ctx, ui),
            CalibrationStep::AlignVideos => align_video_screen(app, ctx, ui),
            CalibrationStep::Calibration => todo!(),
        });
}

fn charuco_board_screen(app: &mut CalibrationApp, ctx: &Context, ui: &mut Ui) {
    let dict_len = app.charuco_target_spec.dictionary.codes.len() as u32;
    SidePanel::left("parameters").show(ctx, |ui| {
        ui.add(
            Slider::new(
                &mut app.charuco_target_spec.cols,
                RangeInclusive::new(1, dict_len * 2 / app.charuco_target_spec.rows),
            )
            .update_while_editing(false)
            .text("Длина")
            .clamping(SliderClamping::Always),
        );
        ui.add(
            Slider::new(
                &mut app.charuco_target_spec.rows,
                RangeInclusive::new(1, dict_len * 2 / app.charuco_target_spec.cols),
            )
            .update_while_editing(false)
            .text("Ширина")
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
        ui.add(
            Slider::new(
                &mut app.charuco_target_spec.cols,
                RangeInclusive::new(1, dict_len * 2 / app.charuco_target_spec.rows),
            )
            .update_while_editing(false)
            .text("Длина листа")
            .clamping(SliderClamping::Always),
        );
        if ui.button("Сохранить паттерн").clicked() {
            // let _ = self.save_pattern();
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

    eframe::egui::CentralPanel::default().show(ctx, |ui| {
        match &mut app.charuco_board_texture_handle {
            Some(texture) => {
                let image =
                    charuco_target_spec_to_dynamic_image(&app.charuco_target_spec, 30, page_spec)
                        .unwrap_or(DynamicImage::default());
                set_color_image_to_texture_handle(&image, texture);
                let texture_ref = &*texture;
                ui.centered_and_justified(|ui| {
                    ui.add(eframe::egui::Image::from_texture(texture_ref).shrink_to_fit())
                });
            }
            None => {
                let image =
                    charuco_target_spec_to_dynamic_image(&app.charuco_target_spec, 10, page_spec)
                        .unwrap_or(DynamicImage::default());
                app.charuco_board_texture_handle = Some(ctx.load_texture(
                    "charuco_board",
                    dynamic_image_to_color_image(&image),
                    TextureOptions::default(),
                ));
            }
        }
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

    for (i, vp) in app.video_players.iter().enumerate() {
        // let img = vp.color_image().;
        // let det_res = match app.charuco_board {
        //     Some(board) => get_charuco(&board),
        //     None => todo!(),
        // };
        // draw_charuco_detection(vp.color_image(), result)

        set_color_image_to_texture_handle(vp.color_image(), &mut app.video_texture_handles[i]);
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
