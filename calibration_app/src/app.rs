use std::path::PathBuf;

use calib_targets::charuco::CharucoBoard;
use eframe::{
    App,
    egui::{Context, TextureHandle, TextureOptions},
};
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
    pub(crate) charuco_board: Option<CharucoBoard>,
    pub(crate) charuco_board_texture_handle: Option<TextureHandle>,
}

pub(crate) enum CalibrationStep {
    SetupCharucoBoard,
    PickVideos,
    AlignVideos,
    Calibration,
}

impl Default for CalibrationApp {
    fn default() -> Self {
        Self {
            video_paths: Vec::new(),
            state: CalibrationStep::PickVideos,
            video_players: Vec::new(),
            offset_in_seconds: Vec::new(),
            _rigs: Vec::new(),
            video_texture_handles: Vec::new(),
            charuco_board: None,
            charuco_board_texture_handle: None,
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

        self.video_texture_handles = self
            .video_players
            .iter()
            .enumerate()
            .map(|(i, vp)| {
                ctx.load_texture(
                    format!("video_frame_{i}"),
                    dynamic_image_to_color_image(vp.color_image()),
                    TextureOptions::default(),
                )
            })
            .collect();

        Ok(())
    }

    pub(crate) fn _perform_calibration() {}
}

impl App for CalibrationApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        render_content(self, ctx);
    }
}
