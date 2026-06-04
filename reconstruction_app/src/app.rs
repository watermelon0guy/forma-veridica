use eframe::App;
use lib_cv::{
    reconstruction::{
        PointCloud, add_color_to_point_cloud, filter_point_cloud_by_confidence,
        gather_points_2d_from_matches, match_first_camera_features_to_all, min_visible_match_set,
        save_point_cloud, track_points_optical_flow_all, triangulate_points_multiple,
        undistort_points,
    },
    video::VideoPlayer,
};
use log::info;
use vision_calibration::rig_extrinsics::RigExtrinsicsExport;

use std::path::PathBuf;

use crate::ui::render_content;

pub(crate) struct ReconstructionApp {
    pub calibration_data: Option<RigExtrinsicsExport>,
    pub video_paths: Vec<PathBuf>,
    pub state: PipelineState,
}

impl Default for ReconstructionApp {
    fn default() -> Self {
        Self {
            calibration_data: None,
            video_paths: Vec::new(),
            state: Default::default(),
        }
    }
}

#[derive(Default)]
pub(crate) enum PipelineState {
    #[default]
    PickCalibrationData,
    PickVideos,
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

    pub(crate) fn run_pipeline(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let calib = self
            .calibration_data
            .as_ref()
            .ok_or("Нет данных калибровки")?;
        let num_cameras = calib.cameras.len();
        if self.video_paths.len() != num_cameras {
            return Err("Количество видео не совпадает с количеством камер".into());
        }

        // Открываем видео
        let mut players: Vec<VideoPlayer> = self
            .video_paths
            .iter()
            .map(|p| VideoPlayer::new(p).map_err(|e| e.to_string()))
            .collect::<Result<_, _>>()?;

        let frame_step = 20u64;

        // --- Первый кадр: SIFT + matching ---
        let first_frames: Vec<_> = players.iter().map(|p| p.dynamic_image().clone()).collect();

        let (all_matches, all_keypoints, _descriptors) =
            match_first_camera_features_to_all(&first_frames);

        let filtered_matches = min_visible_match_set(&all_matches, all_keypoints[0].len());

        let points_2d = gather_points_2d_from_matches(&filtered_matches, &all_keypoints);

        // --- Undistortion ---
        let mut undistorted: Vec<Vec<_>> = Vec::with_capacity(num_cameras);
        for cam_i in 0..num_cameras {
            let undist = undistort_points(&points_2d[cam_i], &calib.cameras[cam_i]);
            undistorted.push(undist);
        }

        // --- Триангуляция ---
        let points_3d = triangulate_points_multiple(&undistorted, calib)?;

        let current_frame: usize = 0;
        let mut cloud = PointCloud {
            points: points_3d,
            timestamp: current_frame,
        };

        add_color_to_point_cloud(&mut cloud, &points_2d[0], &first_frames[0]);

        let before = cloud.points.len();
        filter_point_cloud_by_confidence(&mut cloud, 0.25);
        info!(
            "Кадр 0: отфильтровано {} точек, оставлено {}",
            before - cloud.points.len(),
            cloud.points.len()
        );

        std::fs::create_dir_all("point_clouds")?;
        save_point_cloud(&cloud, format!("point_clouds/frame_{current_frame:04}.ply"))?;

        // --- Готовимся к optical flow ---
        let mut prev_frames = first_frames;
        let mut prev_points = points_2d.clone();

        // --- Цикл по оставшимся кадрам ---
        let total_frames = players[0].total_frames();
        let mut frame_idx: usize = 1;

        loop {
            // Шагаем вперёд на frame_step
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

            // Optical flow: трекинг точек на каждой камере
            let prev_gray: Vec<_> = prev_frames.iter().map(|f| f.to_luma8()).collect();
            let curr_gray: Vec<_> = curr_frames.iter().map(|f| f.to_luma8()).collect();

            let new_points =
                track_points_optical_flow_all(&prev_gray, &curr_gray, &prev_points, 13, 30, 3);

            // Undistortion новых точек
            let mut undistorted: Vec<Vec<_>> = Vec::with_capacity(num_cameras);
            for cam_i in 0..num_cameras {
                let undist = undistort_points(&new_points[cam_i], &calib.cameras[cam_i]);
                undistorted.push(undist);
            }

            // Триангуляция
            let points_3d = triangulate_points_multiple(&undistorted, calib)?;

            let mut cloud = PointCloud {
                points: points_3d,
                timestamp: frame_idx,
            };

            add_color_to_point_cloud(&mut cloud, &new_points[0], &curr_frames[0]);

            let before = cloud.points.len();
            filter_point_cloud_by_confidence(&mut cloud, 0.25);
            info!(
                "Кадр {frame_idx}: отфильтровано {} точек, оставлено {}",
                before - cloud.points.len(),
                cloud.points.len()
            );

            save_point_cloud(&cloud, format!("point_clouds/frame_{frame_idx:04}.ply"))?;

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
}
