use eframe::App;
use eframe::egui::{TextureHandle, TextureOptions};
use lib_cv::video::VideoPlayer;
use lib_pipeline::config::ReconstructionConfig;
use lib_pipeline::runner::run_reconstruction;
use lib_ui::utils::dynamic_image_to_color_image;
use std::sync::{Arc, Mutex, mpsc};
use vision_calibration::rig_extrinsics::RigExtrinsicsExport;

use crate::ui::render_content;

pub(crate) struct ReconstructionApp {
    pub reconstruction_config: ReconstructionConfig,
    pub calibration_data: Option<RigExtrinsicsExport>,
    pub state: PipelineState,
    pub pipeline_thread: Option<std::thread::JoinHandle<()>>,
    pub pipeline_result_rx: Option<mpsc::Receiver<Result<(), String>>>,
    pub pipeline_result: Option<Result<(), String>>,
    pub reconstruction_progress: Arc<Mutex<ReconstructionProgress>>,
    // Для экрана выравнивания
    pub video_players: Vec<VideoPlayer>,
    pub video_texture_handles: Vec<TextureHandle>,
    pub offsets: Vec<f64>,
}

#[derive(Default)]
pub(crate) struct ReconstructionProgress {
    pub(crate) percent: f32,
    pub(crate) is_running: bool,
}

impl Default for ReconstructionApp {
    fn default() -> Self {
        Self {
            reconstruction_config: ReconstructionConfig::default(),
            calibration_data: None,
            state: Default::default(),
            pipeline_thread: None,
            pipeline_result_rx: None,
            pipeline_result: None,
            video_players: Vec::new(),
            video_texture_handles: Vec::new(),
            offsets: Vec::new(),
            reconstruction_progress: Default::default(),
        }
    }
}

#[derive(Default)]
pub(crate) enum PipelineState {
    #[default]
    PickCalibrationData,
    PickVideos,
    AlignVideos,
    ReadyToProcess,
}

impl App for ReconstructionApp {
    fn ui(&mut self, ui: &mut eframe::egui::Ui, _frame: &mut eframe::Frame) {
        render_content(self, ui);
    }
}

impl ReconstructionApp {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn num_cameras(&self) -> Option<usize> {
        match &self.calibration_data {
            Some(cd) => Some(cd.cameras.len()),
            None => None,
        }
    }

    pub(crate) fn init_videos(&mut self, ctx: &eframe::egui::Context) -> Result<(), String> {
        if !self.video_players.is_empty() {
            return Ok(());
        }
        for (i, cam) in self.reconstruction_config.cameras.iter().enumerate() {
            let vp = VideoPlayer::new(&cam.video_path).map_err(|e| e.to_string())?;
            let texture = ctx.load_texture(
                format!("recon_video_{i}"),
                dynamic_image_to_color_image(vp.dynamic_image()),
                TextureOptions::default(),
            );
            self.video_players.push(vp);
            self.video_texture_handles.push(texture);
        }
        Ok(())
    }

    pub(crate) fn start_pipeline_thread(&mut self) {
        let progress = Arc::clone(&self.reconstruction_progress);

        let (tx, rx) = mpsc::channel();
        self.pipeline_result_rx = Some(rx);

        let config = self.reconstruction_config.clone();
        let calib = match &self.calibration_data {
            Some(c) => c.clone(),
            None => {
                self.pipeline_result_rx = None;
                return;
            }
        };

        let handle = std::thread::spawn(move || {
            {
                let mut p = progress
                    .lock()
                    .expect("Вызвали почему то два раза lock. Ошибка в коде");
                p.is_running = true;
                p.percent = 0.0;
            }
            let mut on_progress = |percent: f32| {
                progress.lock().unwrap().percent = percent;
            };

            let result = run_reconstruction(&config, &calib, &mut on_progress);

            let _ = tx.send(result);

            {
                let mut p = progress.lock().unwrap();
                p.is_running = false;
            }
        });

        self.pipeline_thread = Some(handle);
    }
}
