use std::sync::{Arc, Mutex, mpsc};

use calib_targets::{
    aruco::builtins::DICT_6X6_100,
    charuco::{CharucoBoard, CharucoDetectionResult, MarkerLayout::OpenCvCharuco},
    printable::{CharucoTargetSpec, PageSpec, PrintableTargetDocument, render_target_bundle},
};
use eframe::{
    App,
    egui::{Context, TextureHandle, TextureOptions},
};
use image::load_from_memory;
use lib_pipeline::config::CalibrationConfig;
use lib_pipeline::runner::run_calibration;
use lib_ui::utils::dynamic_image_to_color_image;
use vision_calibration::{
    core::{NoMeta, RigView},
    rig_extrinsics::RigExtrinsicsExport,
};

use crate::ui::render_content;
use lib_cv::video::VideoPlayer;

pub(crate) struct CalibrationApp {
    pub(crate) calibration_config: CalibrationConfig,
    pub(crate) state: CalibrationStep,
    pub(crate) video_players: Vec<VideoPlayer>,
    pub(crate) video_texture_handles: Vec<TextureHandle>,
    pub(crate) _rigs: Vec<RigView<NoMeta>>,
    pub(crate) charuco_board: CharucoBoard,
    pub(crate) charuco_board_texture_handle: Option<TextureHandle>,
    pub(crate) last_detected_frame_with_charuco: Vec<Option<FrameWithCharucoData>>,
    pub(crate) draw_charuco_results: bool,

    pub(crate) calibration_progress: Arc<Mutex<CalibrationProgress>>,
    pub(crate) calibration_result_rx: Option<mpsc::Receiver<Result<RigExtrinsicsExport, String>>>,
    pub(crate) calibration_thread: Option<std::thread::JoinHandle<()>>,
    pub(crate) calibration_result: Option<RigExtrinsicsExport>,
    pub(crate) calibration_error: Option<String>,
    pub(crate) advanced_params_open: bool,
}

#[derive(Default)]
pub(crate) struct CalibrationProgress {
    pub(crate) percent: f32,
    pub(crate) is_running: bool,
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
        let charuco_target_spec = CharucoTargetSpec::new(11, 8, 20.0, 0.55, DICT_6X6_100)
            .with_marker_layout(OpenCvCharuco)
            .with_border_bits(3);
        let charuco_board = CharucoBoard::new(charuco_target_spec.to_board_spec())
            .expect("Неправильные даненые по умолчанию для Charuco");

        Self {
            calibration_config: CalibrationConfig::new(charuco_target_spec),
            state: CalibrationStep::SetupCharucoBoard,
            video_players: Vec::new(),
            _rigs: Vec::new(),
            video_texture_handles: Vec::new(),
            charuco_board,
            charuco_board_texture_handle: None,
            last_detected_frame_with_charuco: Vec::new(),
            draw_charuco_results: false,
            calibration_progress: Default::default(),
            calibration_result_rx: None,
            calibration_thread: None,
            calibration_result: None,
            calibration_error: None,
            advanced_params_open: false,
        }
    }
}

impl CalibrationApp {
    pub(crate) fn num_cameras(&self) -> usize {
        self.calibration_config.cameras.len()
    }

    pub(crate) fn init_videos(&mut self, ctx: &Context) -> Result<(), Box<dyn std::error::Error>> {
        if !self.video_players.is_empty() {
            return Ok(());
        }

        let mut players = Vec::new();

        for path in &self.calibration_config.cameras {
            match VideoPlayer::new(&path.video_path) {
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
        self.charuco_board =
            CharucoBoard::new(self.calibration_config.charuco_board.to_board_spec())?;
        Ok(())
    }

    pub(crate) fn start_calibration_thread(&mut self) {
        let progress = Arc::clone(&self.calibration_progress);

        // Создаём канал для результата
        let (tx, rx) = mpsc::channel();
        self.calibration_result_rx = Some(rx);

        // Берём данные для калибровки
        let config = self.calibration_config.clone();

        // Запускаем поток
        let handle = std::thread::spawn(move || {
            // Устанавливаем начальный статус
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

            // Вызываем тяжёлую функцию калибровки
            let result = run_calibration(&config, &mut on_progress);

            // Отправляем результат назад
            let _ = tx.send(result);

            // Отмечаем завершение
            {
                let mut p = progress.lock().unwrap();
                p.is_running = false;
            }
        });

        self.calibration_thread = Some(handle);
    }

    pub(crate) fn sync_offsets_from_players(&mut self) {
        for (camera, player) in self
            .calibration_config
            .cameras
            .iter_mut()
            .zip(self.video_players.iter())
        {
            camera.start_time_in_seconds = player.current_time_in_seconds;
        }
    }
}

impl App for CalibrationApp {
    fn ui(&mut self, ui: &mut eframe::egui::Ui, _frame: &mut eframe::Frame) {
        render_content(self, ui);
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
