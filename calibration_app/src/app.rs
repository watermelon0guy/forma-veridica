use std::path::PathBuf;

use eframe::{
    App,
    egui::{Context, TextureHandle},
};

use crate::{ui::render_content, video::VideoPlayer};

pub(crate) struct CalibrationApp {
    pub(crate) video_paths: Vec<PathBuf>,
    pub(crate) state: CalibrationStep,
    pub(crate) video_players: Vec<VideoPlayer>,
}

pub(crate) enum CalibrationStep {
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
            match VideoPlayer::new(ctx, path) {
                Ok(vp) => players.push(vp),
                Err(e) => return Err(format!("Проблема при создании плеера: {e}").into()),
            }
        }

        self.video_players = players;
        Ok(())
    }
}

impl App for CalibrationApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        render_content(self, ctx);
    }
}
