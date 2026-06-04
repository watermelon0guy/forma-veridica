use std::{
    path::PathBuf,
    sync::{Arc, Mutex, mpsc},
};

use calib_targets::{
    aruco::builtins::DICT_6X6_100,
    charuco::{CharucoBoard, CharucoDetectionResult},
    printable::{CharucoTargetSpec, PageSpec, PrintableTargetDocument, render_target_bundle},
};
use eframe::{
    App,
    egui::{Context, TextureHandle, TextureOptions},
};
use image::load_from_memory;
use lib_cv::calibration::{calibrate_multiple_with_charuco_from_rigs, update_rigs};
use log::{debug, error, info};
use vision_calibration::{
    core::{NoMeta, RigView},
    rig_extrinsics::RigExtrinsicsExport,
};

use crate::ui::render_content;
use lib_cv::video::{VideoPlayer, dynamic_image_to_color_image};

pub(crate) struct CalibrationApp {
    pub(crate) video_paths: Vec<PathBuf>,
    pub(crate) state: CalibrationStep,
    pub(crate) video_players: Vec<VideoPlayer>,
    pub(crate) video_texture_handles: Vec<TextureHandle>,
    pub(crate) offset_in_seconds: Vec<f64>,
    pub(crate) _rigs: Vec<RigView<NoMeta>>,
    pub(crate) charuco_target_spec: CharucoTargetSpec,
    pub(crate) charuco_square_size: f64,
    pub(crate) charuco_board: CharucoBoard,
    pub(crate) charuco_board_texture_handle: Option<TextureHandle>,
    pub(crate) last_detected_frame_with_charuco: Vec<Option<FrameWithCharucoData>>,
    pub(crate) draw_charuco_results: bool,

    pub(crate) calibration_progress: Arc<Mutex<CalibrationProgress>>,
    pub(crate) calibration_result_rx: Option<mpsc::Receiver<Result<RigExtrinsicsExport, String>>>,
    pub(crate) calibration_thread: Option<std::thread::JoinHandle<()>>,
    pub(crate) calibration_result: Option<RigExtrinsicsExport>,
    pub(crate) calibration_error: Option<String>,
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
            draw_charuco_results: false,
            calibration_progress: Default::default(),
            calibration_result_rx: None,
            calibration_thread: None,
            calibration_result: None,
            calibration_error: None,
            charuco_square_size: 20.0,
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
        self.charuco_target_spec.square_size_mm = self.charuco_square_size;
        self.charuco_board = CharucoBoard::new(self.charuco_target_spec.to_board_spec())?;
        Ok(())
    }

    pub(crate) fn start_calibration_thread(&mut self) {
        let progress = Arc::clone(&self.calibration_progress);

        // Создаём канал для результата
        let (tx, rx) = mpsc::channel();
        self.calibration_result_rx = Some(rx);

        // Берём данные для калибровки
        let video_paths = self.video_paths.clone();
        let charuco_board = self.charuco_board.clone();
        let offsets = self.offset_in_seconds.clone();

        // Запускаем поток
        let handle = std::thread::spawn(move || {
            // Устанавливаем начальный статус
            {
                let mut p = progress.lock().unwrap();
                p.is_running = true;
                p.percent = 0.0;
            }

            // Вызываем тяжёлую функцию калибровки
            let result = run_calibration_in_thread(
                video_paths.clone(),
                charuco_board,
                offsets,
                Arc::clone(&progress),
                // ctx, // для request_repaint()
            );

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

fn run_calibration_in_thread(
    video_paths: Vec<PathBuf>,
    charuco_board: CharucoBoard,
    offsets: Vec<f64>,
    progress: Arc<Mutex<CalibrationProgress>>,
    // ctx: Context,
) -> Result<RigExtrinsicsExport, String> {
    let mut img_rigs: Vec<RigView<NoMeta>> = Vec::new();

    let mut video_players: Vec<VideoPlayer> = Vec::new();
    for path in &video_paths {
        match VideoPlayer::new(path) {
            Ok(vp) => video_players.push(vp),
            Err(e) => return Err(format!("Проблема при создании плеера: {e}").into()),
        }
    }

    for (i, player) in video_players.iter_mut().enumerate() {
        if let Err(e) = player.seek_to_time(offsets[i]) {
            error!("Ошибка перехода к офсету: {}", e);
            return Err(format!("Проблема при создании плеера: {e}").into());
        }
    }

    let total_frames = video_players[0].total_frames();

    let mut reading_vids = true;
    while reading_vids {
        let mut cams_imgs = Vec::new();
        for player in &mut video_players {
            debug!(
                "Кадр:{}, время: {}",
                player.current_frame(),
                player.current_time_in_seconds
            );
            if reading_vids {
                let mut p = progress.lock().unwrap();
                p.percent = player.current_frame() as f32 / total_frames as f32;
            }
            cams_imgs.push(player.dynamic_image().to_luma8());
            if let Err(_) = &player.rewind_forward(20) {
                info!("Видео закончилось");
                reading_vids = false;
            };
        }
        if reading_vids {
            update_rigs(&mut img_rigs, cams_imgs, &charuco_board, 2, 8);
        }
    }

    calibrate_multiple_with_charuco_from_rigs(img_rigs).map_err(|e| e.to_string())
}
