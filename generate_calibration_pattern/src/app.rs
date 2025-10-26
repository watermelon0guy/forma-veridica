use std::ops::RangeInclusive;

use eframe::egui::{ColorImage, SliderClamping};
use opencv::{Error, core::Size, imgproc, objdetect::PredefinedDictionaryType, prelude::*};

pub struct GenCalibPatternApp {
    texture_handle: Option<eframe::egui::TextureHandle>,
    size: Size,              // number of chessboard squares in x and y directions
    square_length: i32,      // squareLength chessboard square side length (normally in meters)
    marker_length: i32,      // marker side length (same unit than squareLength)
    dictionary: ChArUcoDict, // dictionary of markers indicating the type of markers
    dictionaries: Vec<ChArUcoDict>,
}

#[derive(Clone)]
struct ChArUcoDict {
    type_opencv: PredefinedDictionaryType,
    name: String,
    amount: i32,
}

impl ChArUcoDict {
    pub fn new(type_opencv: PredefinedDictionaryType) -> Result<Self, String> {
        let name = format!("{:?}", type_opencv);
        let amount: i32 = name
            .rsplit_once("_")
            .and_then(|(_, amount_str)| amount_str.parse().ok())
            .ok_or_else(|| format!("Не удалось извлечь размер словаря {}", name))?;
        Ok(Self {
            type_opencv,
            name,
            amount: amount,
        })
    }
}

impl Default for ChArUcoDict {
    fn default() -> Self {
        ChArUcoDict {
            type_opencv: PredefinedDictionaryType::DICT_4X4_50,
            name: "DICT_4X4_50".to_string(),
            amount: 50,
        }
    }
}

impl Default for GenCalibPatternApp {
    fn default() -> Self {
        let dictionaries: Vec<ChArUcoDict> = (0..=21)
            .filter_map(|i| {
                match PredefinedDictionaryType::try_from(i)
                    .ok()
                    .and_then(|dict_type| ChArUcoDict::new(dict_type).ok())
                {
                    Some(dict) => Some(dict),
                    None => {
                        eprintln!("Не удалось создать словарь для индекса {}", i);
                        None
                    }
                }
            })
            .collect();

        Self {
            texture_handle: None,
            size: Size::new(10, 7),
            square_length: 10,
            marker_length: 7,
            dictionary: ChArUcoDict::default(),
            dictionaries,
        }
    }
}

impl PartialEq for ChArUcoDict {
    fn eq(&self, other: &Self) -> bool {
        self.type_opencv == other.type_opencv
    }
}

impl GenCalibPatternApp {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn generate_pattern(&mut self) -> Result<ColorImage, Error> {
        let dictionary =
            opencv::objdetect::get_predefined_dictionary(self.dictionary.type_opencv).unwrap();
        let charuco_board = opencv::objdetect::CharucoBoard::new_def(
            self.size,
            self.square_length as f32,
            self.marker_length as f32,
            &dictionary,
        )?;
        let mut mat_image = Mat::default();
        charuco_board.generate_image(
            opencv::core::Size::new(
                self.size.width * self.square_length,
                self.size.height * self.square_length,
            ),
            &mut mat_image,
            0,
            1,
        )?;
        let frame_size = [mat_image.cols() as usize, mat_image.rows() as usize];
        let mut rgb_image = opencv::core::Mat::default();
        imgproc::cvt_color_def(&mat_image, &mut rgb_image, imgproc::COLOR_BGR2RGB).unwrap();
        let color_image = eframe::egui::ColorImage::from_rgb(frame_size, rgb_image.data_bytes()?);
        Ok(color_image)
    }

    pub fn set_texture_handler(
        &mut self,
        ctx: &eframe::egui::Context,
    ) -> Result<(), opencv::Error> {
        let color_image = match self.generate_pattern() {
            Ok(image) => image,
            Err(e) => return Err(e),
        };

        if let Some(handle) = &mut self.texture_handle {
            handle.set(color_image, eframe::egui::TextureOptions::NEAREST);
        } else {
            self.texture_handle = Some(ctx.load_texture(
                "pattern_texture",
                color_image,
                eframe::egui::TextureOptions::NEAREST,
            ));
        }
        Ok(())
    }
}

impl eframe::App for GenCalibPatternApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        eframe::egui::SidePanel::left("parameters").show(ctx, |ui| {
            ui.add(
                eframe::egui::Slider::new(
                    &mut self.size.height,
                    RangeInclusive::new(1, self.dictionary.amount * 2 / self.size.width),
                )
                .text("Длина")
                .clamping(SliderClamping::Never),
            );
            ui.add(
                eframe::egui::Slider::new(
                    &mut self.size.width,
                    RangeInclusive::new(1, self.dictionary.amount * 2 / self.size.height),
                )
                .text("Ширина")
                .clamping(SliderClamping::Never),
            );
            ui.add(
                eframe::egui::Slider::new(
                    &mut self.marker_length,
                    RangeInclusive::new(6, (self.square_length as f32 * 0.7) as i32),
                )
                .text("Размер маркера")
                .clamping(SliderClamping::Always),
            );
            ui.add(
                eframe::egui::Slider::new(&mut self.square_length, RangeInclusive::new(10, 60))
                    .text("Размер квадрата")
                    .clamping(SliderClamping::Always),
            );
            eframe::egui::ComboBox::from_label("Наборы маркеров")
                .selected_text(&self.dictionary.name)
                .show_ui(ui, |ui| {
                    // TODO закешировать
                    for d in &self.dictionaries {
                        ui.selectable_value(&mut self.dictionary, d.clone(), &d.name);
                    }
                })
        });

        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            let _ = self.set_texture_handler(&ctx);
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
