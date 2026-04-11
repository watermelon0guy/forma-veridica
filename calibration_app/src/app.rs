use std::path::PathBuf;

use calib_targets::{
    aruco::builtins::{DICT_4X4_100, DICT_6X6_100},
    charuco::{CharucoBoard, CharucoDetectionResult},
    printable::{CharucoTargetSpec, PageSpec, PrintableTargetDocument, render_target_bundle},
};
use eframe::{
    App,
    egui::{Context, TextureHandle, TextureOptions},
};
use image::load_from_memory;
use vision_calibration::core::{NoMeta, RigView};

use crate::{
    ui::render_content,
    video::{VideoPlayer, dynamic_image_to_color_image},
};

pub(crate) struct CalibrationApp {
    pub(crate) video_paths: Vec<PathBuf>,
    pub(crate) state: CalibrationStep,
    pub(crate) video_players: Vec<VideoPlayer>,
    pub(crate) video_texture_handles: Vec<TextureHandle>,
    pub(crate) offset_in_seconds: Vec<f64>,
    pub(crate) _rigs: Vec<RigView<NoMeta>>,
    pub(crate) charuco_target_spec: CharucoTargetSpec,
    pub(crate) charuco_board: CharucoBoard,
    pub(crate) charuco_board_texture_handle: Option<TextureHandle>,
    pub(crate) last_detected_frame_with_charuco: Vec<Option<FrameWithCharucoData>>,
}

pub(crate) struct FrameWithCharucoData {
    pub(crate) frame: u64,
    pub(crate) charuco_data: Option<CharucoDetectionResult>,
}

pub(crate) enum CalibrationStep {
    SetupCharucoBoard,
    PickVideos,
    AlignVideos,
    Calibration,
}

impl Default for CalibrationApp {
    fn default() -> Self {
        let charuco_target_spec = CharucoTargetSpec {
            rows: 11,
            cols: 8,
            square_size_mm: 20.0,
            marker_size_rel: 0.55,
            dictionary: DICT_6X6_100,
            marker_layout: calib_targets::charuco::MarkerLayout::OpenCvCharuco,
            border_bits: 3,
        };
        let charuco_board = CharucoBoard::new(charuco_target_spec.to_board_spec())
            .expect("Неправильные даненые по умолчанию для Charuco");

        Self {
            video_paths: Vec::new(),
            state: CalibrationStep::SetupCharucoBoard,
            video_players: Vec::new(),
            offset_in_seconds: Vec::new(),
            _rigs: Vec::new(),
            video_texture_handles: Vec::new(),
            charuco_board,
            charuco_board_texture_handle: None,
            charuco_target_spec,
            last_detected_frame_with_charuco: Vec::new(),
        }
    }
}

impl CalibrationApp {
    pub(crate) fn num_cameras(&self) -> usize {
        self.video_paths.len()
    }

    pub(crate) fn init_videos(&mut self, ctx: &Context) -> Result<(), Box<dyn std::error::Error>> {
        if !self.video_players.is_empty() {
            return Ok(());
        }

        let mut players = Vec::new();

        for path in &self.video_paths {
            match VideoPlayer::new(path) {
                Ok(vp) => players.push(vp),
                Err(e) => return Err(format!("Проблема при создании плеера: {e}").into()),
            }
        }

        self.video_players = players;
        self.last_detected_frame_with_charuco =
            (0..self.video_players.len()).map(|_| None).collect();

        self.video_texture_handles = self
            .video_players
            .iter()
            .enumerate()
            .map(|(i, vp)| {
                ctx.load_texture(
                    format!("video_frame_{i}"),
                    dynamic_image_to_color_image(vp.dynamic_image()),
                    TextureOptions::default(),
                )
            })
            .collect();

        Ok(())
    }

    pub(crate) fn update_board_from_spec(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.charuco_board = CharucoBoard::new(self.charuco_target_spec.to_board_spec())?;
        Ok(())
    }

    pub(crate) fn _perform_calibration() {}
}

impl App for CalibrationApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        render_content(self, ctx);
    }
}

pub(crate) fn charuco_target_spec_to_dynamic_image(
    charuco_target_spec: &CharucoTargetSpec,
    dpi: u32,
    page_spec: PageSpec,
) -> Result<image::DynamicImage, Box<dyn std::error::Error>> {
    // Создаём документ для печати (размеры в мм)
    let mut document =
        PrintableTargetDocument::from_charuco_board_spec_mm(&charuco_target_spec.to_board_spec());
    document.render.png_dpi = dpi;
    document.page = page_spec;
    // Рендерим - получаем PNG байты, SVG и JSON
    let bundle = render_target_bundle(&document)?;
    Ok(load_from_memory(&bundle.png_bytes)?)
}
