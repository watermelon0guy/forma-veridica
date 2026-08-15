use eframe::App;
use eframe::egui::{TextureHandle, TextureOptions};
use lib_cv::{
    reconstruction::{
        PointCloud, add_color_to_point_cloud, filter_matches_by_epipolar,
        filter_point_cloud_by_confidence, gather_points_2d_from_matches,
        match_with_epipolar_constraint, min_visible_match_set, save_point_cloud,
        track_points_optical_flow_all, triangulate_points_multiple, undistort_points,
    },
    video::VideoPlayer,
};
use lib_ui::utils::dynamic_image_to_color_image;
use log::info;
use std::sync::mpsc;
use vision_calibration::rig_extrinsics::RigExtrinsicsExport;

use std::path::PathBuf;

use crate::ui::render_content;

pub(crate) struct ReconstructionApp {
    pub calibration_data: Option<RigExtrinsicsExport>,
    pub video_paths: Vec<PathBuf>,
    pub state: PipelineState,
    pub pipeline_thread: Option<std::thread::JoinHandle<()>>,
    pub pipeline_result_rx: Option<mpsc::Receiver<Result<(), String>>>,
    pub pipeline_result: Option<Result<(), String>>,
    // Для экрана выравнивания
    pub video_players: Vec<VideoPlayer>,
    pub video_texture_handles: Vec<TextureHandle>,
    pub offsets: Vec<f64>,
}

impl Default for ReconstructionApp {
    fn default() -> Self {
        Self {
            calibration_data: None,
            video_paths: Vec::new(),
            state: Default::default(),
            pipeline_thread: None,
            pipeline_result_rx: None,
            pipeline_result: None,
            video_players: Vec::new(),
            video_texture_handles: Vec::new(),
            offsets: Vec::new(),
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
        for (i, path) in self.video_paths.iter().enumerate() {
            let vp = VideoPlayer::new(path).map_err(|e| e.to_string())?;
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
        let (tx, rx) = mpsc::channel();
        self.pipeline_result_rx = Some(rx);

        let video_paths = self.video_paths.clone();
        let calib = self.calibration_data.clone();
        let offsets = self.offsets.clone();

        let handle = std::thread::spawn(move || {
            let result = run_pipeline_in_thread(video_paths, calib, offsets);
            let _ = tx.send(result);
        });

        self.pipeline_thread = Some(handle);
    }

    pub(crate) fn _run_pipeline(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        run_pipeline_in_thread(
            self.video_paths.clone(),
            self.calibration_data.clone(),
            self.offsets.clone(),
        )
        .map_err(|e| e.into())
    }
}

/// Свободная функция для запуска в отдельном потоке (владеет данными).
fn run_pipeline_in_thread(
    video_paths: Vec<PathBuf>,
    calib_data: Option<RigExtrinsicsExport>,
    offsets: Vec<f64>,
) -> Result<(), String> {
    let calib = calib_data.ok_or("Нет данных калибровки")?;
    let num_cameras = calib.cameras.len();
    if num_cameras < 2 {
        return Err("Требуется минимум 2 камеры".into());
    }
    if video_paths.len() != num_cameras {
        return Err("Количество видео не совпадает с количеством камер".into());
    }

    let mut players: Vec<VideoPlayer> = video_paths
        .iter()
        .map(|p| VideoPlayer::new(p).map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;

    // Применяем смещения времени
    for (i, p) in players.iter_mut().enumerate() {
        if i < offsets.len() {
            p.seek_to_time(offsets[i]).map_err(|e| e.to_string())?;
        }
    }

    let frame_step = 5u64;

    // --- Первый кадр: SIFT + matching ---
    let first_frames: Vec<_> = players.iter().map(|p| p.dynamic_image().clone()).collect();

    let (all_matches, all_keypoints) = match_with_epipolar_constraint(&first_frames, &calib, 15.0);

    let filtered_matches = min_visible_match_set(&all_matches, all_keypoints[0].len());

    let points_2d_raw = gather_points_2d_from_matches(&filtered_matches, &all_keypoints);

    // Matching already did epipolar filtering internally,
    // go straight to undistort + triangulate
    let mut undistorted: Vec<Vec<_>> = Vec::with_capacity(num_cameras);
    for cam_i in 0..num_cameras {
        let undist = undistort_points(&points_2d_raw[cam_i], &calib.cameras[cam_i]);
        undistorted.push(undist);
    }

    let points_3d = triangulate_points_multiple(&undistorted, &calib).map_err(|e| e.to_string())?;

    let current_frame: usize = 0;
    let mut cloud = PointCloud {
        points: points_3d,
        timestamp: current_frame,
    };

    add_color_to_point_cloud(&mut cloud, &points_2d_raw[0], &first_frames[0]);

    let before = cloud.points.len();
    filter_point_cloud_by_confidence(&mut cloud, 0.05);
    info!(
        "Кадр 0: отфильтровано {} точек, оставлено {}",
        before - cloud.points.len(),
        cloud.points.len()
    );

    std::fs::create_dir_all("point_clouds").map_err(|e| e.to_string())?;
    save_point_cloud(&cloud, format!("point_clouds/frame_{current_frame:04}.ply"))
        .map_err(|e| e.to_string())?;

    // --- Готовимся к optical flow ---
    let mut prev_frames = first_frames;
    // Для optical flow нужны ИСКАЖЁННЫЕ координаты (трекинг по сырым кадрам)
    let mut prev_points = points_2d_raw;

    // --- Цикл по оставшимся кадрам ---
    let total_frames = players[0].total_frames();
    let mut frame_idx: usize = 1;

    loop {
        let mut eof = false;
        for p in &mut players {
            if p.rewind_forward(frame_step).is_err() {
                eof = true;
                break;
            }
        }
        if eof {
            break;
        }

        let curr_frames: Vec<_> = players.iter().map(|p| p.dynamic_image().clone()).collect();

        let prev_gray: Vec<_> = prev_frames.iter().map(|f| f.to_luma8()).collect();
        let curr_gray: Vec<_> = curr_frames.iter().map(|f| f.to_luma8()).collect();

        let new_points_raw =
            track_points_optical_flow_all(&prev_gray, &curr_gray, &prev_points, 13, 30, 3);

        // Undistortion до эпиполярной фильтрации
        let mut new_undistorted_raw: Vec<Vec<_>> = Vec::with_capacity(num_cameras);
        for cam_i in 0..num_cameras {
            let undist = undistort_points(&new_points_raw[cam_i], &calib.cameras[cam_i]);
            new_undistorted_raw.push(undist);
        }

        // Эпиполярная фильтрация на неискажённых точках
        let (new_undistorted, good_indices) =
            filter_matches_by_epipolar(&new_undistorted_raw, &calib, 15.0);

        // Фильтруем искажённые координаты для lookup цвета и для следующей итерации optical flow
        let new_points: Vec<Vec<_>> = new_points_raw
            .iter()
            .map(|cam| good_indices.iter().map(|&i| cam[i]).collect())
            .collect();

        let points_3d =
            triangulate_points_multiple(&new_undistorted, &calib).map_err(|e| e.to_string())?;

        let mut cloud = PointCloud {
            points: points_3d,
            timestamp: frame_idx,
        };

        add_color_to_point_cloud(&mut cloud, &new_points[0], &curr_frames[0]);

        filter_point_cloud_by_confidence(&mut cloud, 0.05);

        save_point_cloud(&cloud, format!("point_clouds/frame_{frame_idx:04}.ply"))
            .map_err(|e| e.to_string())?;

        prev_frames = curr_frames;
        prev_points = new_points;
        frame_idx += 1;

        if players[0].current_frame() >= total_frames - frame_step {
            break;
        }
    }

    info!("Пайплайн завершён. Обработано {} кадров", frame_idx);
    Ok(())
}
