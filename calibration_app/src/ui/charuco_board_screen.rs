use std::ops::RangeInclusive;

use calib_targets::{
    aruco::builtins::{BUILTIN_DICTIONARY_NAMES, builtin_dictionary},
    printable::{PageOrientation, PageSize, PageSpec},
};
use eframe::egui::{ComboBox, Panel, Slider, SliderClamping, TextureOptions, Ui};
use image::DynamicImage;

use crate::{
    app::{CalibrationApp, CalibrationStep, charuco_target_spec_to_dynamic_image},
    video::{dynamic_image_to_color_image, set_color_image_to_texture_handle},
};

pub fn charuco_board_screen(app: &mut CalibrationApp, ui: &mut Ui) {
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

        if ui.button("Продолжить").clicked() {
            if let Ok(()) = app.update_board_from_spec() {
                app.state = CalibrationStep::PickVideos;
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
