use std::ops::RangeInclusive;

use calib_targets::{
    aruco::builtins::{BUILTIN_DICTIONARY_NAMES, builtin_dictionary},
    printable::{PageOrientation, PageSize, PageSpec},
};
use eframe::egui::{ComboBox, Panel, Slider, SliderClamping, TextureOptions, Ui};
use image::ImageFormat;
use lib_ui::utils::{dynamic_image_to_color_image, set_color_image_to_texture_handle};

use crate::app::{CalibrationApp, CalibrationStep, charuco_target_spec_to_dynamic_image};

pub fn charuco_board_screen(app: &mut CalibrationApp, ui: &mut Ui) {
    let dict_len = app
        .calibration_config
        .charuco_board
        .dictionary
        .codes()
        .len() as u32;

    let page_margin_mm = 10.0;

    Panel::left("parameters").show(ui, |ui| {
        ui.add(
            Slider::new(
                &mut app.calibration_config.charuco_board.cols,
                RangeInclusive::new(1, dict_len * 2 / app.calibration_config.charuco_board.rows),
            )
            .update_while_editing(false)
            .text("Столбцы")
            .clamping(SliderClamping::Always),
        );
        ui.add(
            Slider::new(
                &mut app.calibration_config.charuco_board.rows,
                RangeInclusive::new(1, dict_len * 2 / app.calibration_config.charuco_board.cols),
            )
            .update_while_editing(false)
            .text("Строчки")
            .clamping(SliderClamping::Always),
        );
        ui.add(
            Slider::new(
                &mut app.calibration_config.charuco_board.marker_size_rel,
                RangeInclusive::new(0.01, 0.99),
            )
            .update_while_editing(false)
            .text("Размер маркера")
            .clamping(SliderClamping::Always),
        );
        ui.add(
            Slider::new(
                &mut app.calibration_config.charuco_board.square_size_mm,
                RangeInclusive::new(5.0, 100.0),
            )
            .update_while_editing(false)
            .text("Размер квадрата")
            .clamping(SliderClamping::Always),
        );
        ComboBox::from_label("Наборы маркеров")
            .selected_text(app.calibration_config.charuco_board.dictionary.name())
            .show_ui(ui, |ui| {
                for d in BUILTIN_DICTIONARY_NAMES {
                    let dict_name = d.to_string();
                    if ui
                        .selectable_label(
                            app.calibration_config.charuco_board.dictionary.name() == dict_name,
                            &dict_name,
                        )
                        .clicked()
                    {
                        if let Some(new_dict) = builtin_dictionary(d) {
                            app.calibration_config.charuco_board.dictionary = new_dict;
                        }
                    }
                }
            });

        if ui.button("Продолжить").clicked() {
            if let Ok(()) = app.update_board_from_spec() {
                app.state = CalibrationStep::PickVideos;
                return;
            }
        }

        ui.separator();

        if ui.button("Сохранить как PNG").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .set_title("Сохранить паттерн как PNG")
                .add_filter("PNG", &["png"])
                .set_file_name("charuco_board.png")
                .save_file()
            {
                let mut save_page_spec = PageSpec::default();
                save_page_spec.size = PageSize::Custom {
                    width_mm: app.calibration_config.charuco_board.square_size_mm
                        * app.calibration_config.charuco_board.cols as f64
                        + page_margin_mm * 2.0,
                    height_mm: app.calibration_config.charuco_board.square_size_mm
                        * app.calibration_config.charuco_board.rows as f64
                        + page_margin_mm * 2.0,
                };
                save_page_spec.orientation = PageOrientation::Portrait;
                save_page_spec.margin_mm = page_margin_mm;
                match charuco_target_spec_to_dynamic_image(
                    &app.calibration_config.charuco_board,
                    300,
                    save_page_spec,
                ) {
                    Ok(image) => {
                        if let Err(e) = image.save_with_format(&path, ImageFormat::Png) {
                            log::error!("Ошибка сохранения PNG: {e}");
                        }
                    }
                    Err(e) => {
                        log::error!("Ошибка генерации изображения: {e}");
                    }
                }
            }
        }
    });

    let mut page_spec = PageSpec::default();
    page_spec.size = PageSize::Custom {
        width_mm: app.calibration_config.charuco_board.square_size_mm
            * app.calibration_config.charuco_board.cols as f64
            + page_margin_mm * 2.0,
        height_mm: app.calibration_config.charuco_board.square_size_mm
            * app.calibration_config.charuco_board.rows as f64
            + page_margin_mm * 2.0,
    };
    page_spec.orientation = PageOrientation::Portrait;
    page_spec.margin_mm = page_margin_mm;

    eframe::egui::CentralPanel::default().show(ui, |ui| {
        if let Ok(image) = charuco_target_spec_to_dynamic_image(
            &app.calibration_config.charuco_board,
            60,
            page_spec,
        ) {
            match &mut app.charuco_board_texture_handle {
                Some(texture) => {
                    set_color_image_to_texture_handle(&image, texture);
                    let texture_ref = &*texture;
                    ui.centered_and_justified(|ui| {
                        ui.add(eframe::egui::Image::from_texture(texture_ref).shrink_to_fit())
                    });
                }
                None => {
                    app.charuco_board_texture_handle = Some(ui.ctx().load_texture(
                        "charuco_board",
                        dynamic_image_to_color_image(&image),
                        TextureOptions::default(),
                    ));
                }
            }
        }
    });
}
