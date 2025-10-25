use std::ops::RangeInclusive;

use eframe::egui::{ColorImage, SliderClamping};
use opencv::{
    Error,
    core::Size,
    imgproc,
    objdetect::{self, PredefinedDictionaryType},
    prelude::*,
};

pub struct GenCalibPatternApp {
    texture_handle: Option<eframe::egui::TextureHandle>,
    size: Size,         // number of chessboard squares in x and y directions
    square_length: i32, // squareLength chessboard square side length (normally in meters)
    marker_length: i32, // marker side length (same unit than squareLength)
    dictionary: PredefinedDictionaryType, // dictionary of markers indicating the type of markers
}

impl Default for GenCalibPatternApp {
    fn default() -> Self {
        Self {
            texture_handle: None,
            size: Size::new(10, 7),
            square_length: 10,
            marker_length: 5,
            dictionary: objdetect::PredefinedDictionaryType::DICT_4X4_50,
        }
    }
}

impl GenCalibPatternApp {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn generate_pattern(&mut self) -> Result<ColorImage, Error> {
        let dictionary = opencv::objdetect::get_predefined_dictionary(self.dictionary).unwrap();
        let charuco_board = opencv::objdetect::CharucoBoard::new_def(
            self.size,
            self.square_length as f32,
            self.marker_length as f32,
            &dictionary,
        )?;
        let mut mat_image = Mat::default();
        charuco_board.generate_image(
            opencv::core::Size::new(
                self.size.width * self.square_length as i32 * 10,
                self.size.height * self.square_length as i32 * 10,
            ),
            &mut mat_image,
            20,
            1,
        )?;
        let frame_size = [mat_image.cols() as usize, mat_image.rows() as usize];
        let mut rgb_image = opencv::core::Mat::default();
        imgproc::cvt_color_def(&mat_image, &mut rgb_image, imgproc::COLOR_BGR2RGB).unwrap();
        let color_image = eframe::egui::ColorImage::from_rgb(frame_size, rgb_image.data_bytes()?);
        Ok(color_image)
    }

    pub fn set_texture_handler(&mut self, ctx: &eframe::egui::Context) {
        let color_image = match self.generate_pattern() {
            Ok(image) => image,
            Err(e) => {
                eprintln!("Ошибка генерации паттерна: {}", e);
                // Создаем пустое изображение в случае ошибки
                eframe::egui::ColorImage::new([10, 5], eframe::epaint::Color32::WHITE)
            }
        };

        if let Some(handle) = &mut self.texture_handle {
            // Если текстура УЖЕ существует - ОБНОВЛЯЕМ её содержимое
            handle.set(color_image, eframe::egui::TextureOptions::default());
        } else {
            // Если текстуры ЕЩЁ НЕТ - СОЗДАЁМ новую
            self.texture_handle = Some(ctx.load_texture(
                "pattern_texture",
                color_image,
                eframe::egui::TextureOptions::default(),
            ));
        }
    }
}

impl eframe::App for GenCalibPatternApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        eframe::egui::SidePanel::left("parameters").show(ctx, |ui| {
            ui.add(
                eframe::egui::Slider::new(&mut self.size.height, RangeInclusive::new(1, 30))
                    .text("Длина")
                    .clamping(SliderClamping::Never),
            );
            ui.add(
                eframe::egui::Slider::new(&mut self.size.width, RangeInclusive::new(1, 30))
                    .text("Ширина")
                    .clamping(SliderClamping::Never),
            );
            ui.add(
                eframe::egui::Slider::new(
                    &mut self.marker_length,
                    RangeInclusive::new(2, (self.square_length as f32 * 0.7) as i32),
                )
                .text("Размер маркера")
                .clamping(SliderClamping::Always),
            );
            ui.add(
                eframe::egui::Slider::new(&mut self.square_length, RangeInclusive::new(2, 30))
                    .text("Размер квадрата")
                    .clamping(SliderClamping::Always),
            );
            eframe::egui::ComboBox::from_label("Наборы маркеров")
                .selected_text(format!("{:?}", self.dictionary))
                .show_ui(ui, |ui| {
                    // TODO закешировать
                    for i in 0..=21 {
                        if let Ok(dict) = objdetect::PredefinedDictionaryType::try_from(i) {
                            ui.selectable_value(&mut self.dictionary, dict, format!("{:?}", dict));
                        }
                    }
                })
        });

        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            self.set_texture_handler(&ctx);
            if let Some(texture) = &self.texture_handle {
                ui.centered_and_justified(|ui| {
                    ui.add(eframe::egui::Image::from_texture(texture).shrink_to_fit())
                });
            } else {
                ui.label("Паттерн не сгенерирован");
            }
        });

        // opencv::imgcodecs::imwrite("charuco_board.png", &img, &opencv::core::Vector::new())
        //     .unwrap();
    }
}
