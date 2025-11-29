use eframe::egui;
use std::{default, path::PathBuf};

pub struct ReconstructionApp {
    project_path: PathBuf,
}

impl Default for ReconstructionApp {
    fn default() -> Self {
        Self {
            project_path: Default::default(),
        }
    }
}

impl eframe::App for ReconstructionApp {
    fn update(&mut self, ctx: &eframe::egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Реконструкция");
        });
    }
}

impl ReconstructionApp {
    pub fn new() -> Self {
        Self::default()
    }
}
