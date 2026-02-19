use lib_cv::calibration::load_camera_parameters;
use lib_cv::correspondence::gather_points_2d_from_matches;
use lib_cv::reconstruction::{
    PointCloud, add_color_to_point_cloud, filter_point_cloud_by_confindence,
    match_first_camera_features_to_all, min_visible_match_set, save_point_cloud,
    undistort_points_single_camera,
};
use lib_cv::utils::{
    open_video_captures, read_frames, split_video_into_quadrants, vector_point2f_to_mat,
};
use log::{debug, error, info};
use opencv::core::{Point2f, Vector};
use opencv::video::calc_optical_flow_pyr_lk;
use opencv::videoio::VideoCapture;
use opencv::{Error, prelude::*};

use std::{fs::create_dir_all, path::PathBuf};

use crate::model::{CalibrationData, PipelineState, ProjectResources, VideoData};
use crate::ui::UiRenderer;

pub(crate) struct ReconstructionApp {
    pub resources: ProjectResources,
    pub pipeline_state: PipelineState,
}

impl Default for ReconstructionApp {
    fn default() -> Self {
        Self {
            resources: Default::default(),
            pipeline_state: Default::default(),
        }
    }
}

impl eframe::App for ReconstructionApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        UiRenderer::render_content(self, ctx);
    }
}

impl ReconstructionApp {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn set_project_folder(&mut self, p: std::path::PathBuf) {
        self.resources = ProjectResources {
            project_path: Some(p),
            calibration_data: None,
            video_data: None,
        };
        self.pipeline_state = PipelineState::FetchProject
    }

    pub(crate) fn load_camera_parameters(&mut self, path: PathBuf) {
        let project_path = self.resources.project_path.as_ref().unwrap();
        let dest_path = project_path.join("camera_parameters.yml");

        if let Err(_) = std::fs::copy(&path, &dest_path) {
            return;
        }

        let cam_params = match load_camera_parameters(dest_path.to_str().unwrap()) {
            Ok(c) => c,
            Err(_) => return,
        };
        self.resources.calibration_data = Some(CalibrationData::new(dest_path, cam_params));
    }

    pub(crate) fn pick_camera_video(&mut self, cam_num: usize) {
        // Не самая понятная функция конечно...
        match rfd::FileDialog::new()
            .add_filter("Видео", &["mp4"])
            .set_title("Выбрать видео")
            .pick_file()
        {
            Some(file_path) => {
                let project_path = self.resources.project_path.as_ref().unwrap();
                let dest_path = project_path.join("data/video");
                if let Err(_) = create_dir_all(&dest_path) {
                    return;
                }
                let dest_path = dest_path.join(format!("camera_{cam_num}.mp4"));

                if let Err(_) = std::fs::copy(&file_path, &dest_path) {
                    return;
                }
                match &mut self.resources.video_data {
                    Some(vd) => {
                        vd.video_files[cam_num] = Some(dest_path);
                    }
                    None => {
                        let num_cams = match &self.resources.calibration_data {
                            Some(cb) => cb.num_cameras,
                            None => return,
                        };
                        self.resources.video_data =
                            Some(match VideoData::new(&dest_path, cam_num, num_cams) {
                                Ok(vd) => vd,
                                Err(_) => return,
                            });
                    }
                }
            }
            None => return,
        }
    }

    pub(crate) fn pick_from_4_combined_video(&mut self) {
        if let Some(file_path) = rfd::FileDialog::new()
            .add_filter("Видео", &["mp4"])
            .set_title("Выбрать видео")
            .pick_file()
        {
            let project_path = self.resources.project_path.as_ref().unwrap();
            let dest_path = project_path.join("data/video");
            if let Err(_) = create_dir_all(&dest_path) {
                return;
            }

            if let Ok(paths) = split_video_into_quadrants(&file_path, &dest_path, "camera") {
                let paths: Vec<Option<PathBuf>> = paths.iter().map(|p| Some(p.clone())).collect();
                if let Ok(vd) = VideoData::from_vec(paths) {
                    self.resources.video_data = Some(vd);
                }
            }
        }
    }

    pub(crate) fn fetch_project(&mut self) {
        self.fetch_camera_params();
        self.fetch_video_data();
        self.pipeline_state = PipelineState::SetupMenu;
    }

    pub(crate) fn fetch_camera_params(&mut self) {
        let project_path = self.resources.project_path.as_ref().unwrap();
        let file_path = project_path.join("camera_parameters.yml");

        if file_path.exists() {
            let cam_params = match load_camera_parameters(file_path.to_str().unwrap()) {
                Ok(c) => c,
                Err(_) => return,
            };
            self.resources.calibration_data = Some(CalibrationData::new(file_path, cam_params));
        }
    }

    pub(crate) fn fetch_video_data(&mut self) {
        let project_path = self.resources.project_path.as_ref().unwrap();
        let video_files: Vec<Option<PathBuf>> = match project_path.join("data/video").read_dir() {
            Ok(read_dir) => read_dir
                .filter_map(|entry| entry.ok())
                .map(|entry| Some(entry.path()))
                .collect(),
            Err(_) => vec![],
        };
        if let Ok(video_data) = VideoData::from_vec(video_files) {
            self.resources.video_data = Some(video_data);
        }
    }

    pub(crate) fn run_pipeline(&self) -> Result<(), opencv::Error> {
        let mut caps: Vec<VideoCapture> = Vec::new();

        let video_data = self
            .resources
            .video_data
            .as_ref()
            .ok_or_else(|| Error::new(-1, "VideoData не загружена"))?;

        let calibration_data = self
            .resources
            .calibration_data
            .as_ref()
            .ok_or_else(|| Error::new(-1, "CalibrationData не загружена"))?;

        let project_path = self
            .resources
            .project_path
            .as_ref()
            .ok_or_else(|| Error::new(-1, "Нет пути проекта не загружена"))?;

        open_video_captures(&mut caps, &video_data.video_files)?;

        let mut frames = vec![Mat::default(); caps.len()];

        read_frames(&mut caps, &mut frames)?;

        let (mut all_matches, keypoints_list, _descriptors_list) =
            match_first_camera_features_to_all(&frames);

        all_matches = min_visible_match_set(&mut all_matches, &keypoints_list);

        let points_2d: Vector<Mat> =
            match gather_points_2d_from_matches(&all_matches, &keypoints_list) {
                Ok(p_2d) => {
                    debug!("Координаты извлечены из массива общих совпадений");
                    p_2d
                }
                Err(e) => {
                    error!(
                        "Ошибка извлечения координат из массива общих совпадений: {}",
                        e
                    );
                    return Err(Error::new(-1, "Не удалось извлечь 2D точки из совпадений"));
                }
            };
        let mut undistorted_points_2d = Vector::<Mat>::default();

        for (i, points) in points_2d.iter().enumerate() {
            let undistorted_nx2 =
                match undistort_points_single_camera(&points, &calibration_data.camera_params[i]) {
                    Ok(u_nx2) => u_nx2,
                    Err(e) => {
                        error!("Ошибка в undistort_points_single_camera: {}", e);
                        return Err(e);
                    }
                };

            undistorted_points_2d.push(undistorted_nx2);
        }

        let points_3d = match lib_cv::reconstruction::triangulate_points_multiple(
            &undistorted_points_2d,
            &calibration_data.camera_params,
        ) {
            Ok(points) => points,
            Err(e) => {
                error!("Ошибка при триангуляции точек: {:?}", e);
                return Err(e);
            }
        };

        let current_frame: usize = 0;

        let mut cloud = PointCloud {
            points: points_3d,
            timestamp: current_frame,
        };

        add_color_to_point_cloud(&mut cloud, &points_2d, &frames[0]);

        let initial_count = cloud.points.len();
        filter_point_cloud_by_confindence(&mut cloud, 0.25);
        info!(
            "Отфильтровано {} точек (оставлено {})",
            initial_count - cloud.points.len(),
            cloud.points.len()
        );
        let dest_path = project_path.join(format!("data/point_clouds"));
        let filename = dest_path.join(format!("point_cloud_{current_frame}.ply"));
        if let Err(e) = create_dir_all(&dest_path) {
            return Err(opencv::Error::new(
                -1,
                &format!("Не удалось создать директорию: {}", e),
            ));
        }

        match save_point_cloud(&cloud, &filename) {
            Ok(_) => info!(
                "Облако точек успешно сохранено в файл: {}",
                filename.display()
            ),
            Err(e) => error!("Ошибка при сохранении облака точек: {:?}", e),
        };

        let mut prev_images = frames.clone();

        let mut prev_points: Vec<Vector<Point2f>> =
            vec![Vector::<Point2f>::default(); calibration_data.num_cameras];
        for camera_i in 0..calibration_data.num_cameras {
            for j in 0..points_2d.get(camera_i).unwrap().rows() {
                let x = *points_2d
                    .get(camera_i as usize)
                    .unwrap()
                    .at_2d::<f64>(j, 0)
                    .unwrap() as f32;
                let y = *points_2d
                    .get(camera_i as usize)
                    .unwrap()
                    .at_2d::<f64>(j, 1)
                    .unwrap() as f32;
                prev_points[camera_i].push(opencv::core::Point2f::new(x, y));
            }
        }

        for current_frame in 1..video_data.total_frames {
            read_frames(&mut caps, &mut frames)?;
            let win_size = opencv::core::Size::new(13, 13);
            let max_level = 3;
            let criteria = opencv::core::TermCriteria::new(
                opencv::core::TermCriteria_EPS + opencv::core::TermCriteria_COUNT,
                1000_000,
                0.000_001,
            )
            .unwrap();
            let flags = 0;
            let min_eig_threshold = 1e-4;

            let mut undistorted_points_2d = Vector::<Mat>::default();

            for (camera_i, (prev, next)) in prev_images.iter().zip(frames.iter()).enumerate() {
                // Подготавливаем данные для оптического потока
                let mut next_points = Vector::<Point2f>::default();
                let mut status = Vector::<u8>::default();
                let mut err = Vector::<f32>::default();

                // Преобразуем points_2d в формат для оптического потока (используем точки первой камеры)
                calc_optical_flow_pyr_lk(
                    &prev,
                    &next,
                    &prev_points[camera_i],
                    &mut next_points,
                    &mut status,
                    &mut err,
                    win_size,
                    max_level,
                    criteria,
                    flags,
                    min_eig_threshold,
                )
                .unwrap();

                debug!(
                    "Потеряно треков: {}",
                    status.iter().filter(|&s| s == 0).count()
                );

                let points_mat = match vector_point2f_to_mat(&next_points) {
                    Ok(mat) => mat,
                    Err(e) => {
                        error!("Ошибка конвертации из vector в mat: {}", e);
                        return Err(e);
                    }
                };
                let undistorted_nx2 = match undistort_points_single_camera(
                    &points_mat,
                    &calibration_data.camera_params[camera_i],
                ) {
                    Ok(u_nx2) => u_nx2,
                    Err(e) => {
                        error!("Ошибка в undistort_points_single_camera: {}", e);
                        return Err(e);
                    }
                };
                undistorted_points_2d.push(undistorted_nx2);

                prev_points[camera_i] = next_points;
            }

            let points_3d = match lib_cv::reconstruction::triangulate_points_multiple(
                &undistorted_points_2d,
                &calibration_data.camera_params,
            ) {
                Ok(points) => {
                    info!(
                        "Триангуляция успешно выполнена. Получено {} 3D точек",
                        points.len()
                    );
                    points
                }
                Err(e) => {
                    error!("Ошибка при триангуляции точек: {:?}", e);
                    return Err(e);
                }
            };

            let mut cloud = PointCloud {
                points: points_3d,
                timestamp: current_frame,
            };

            add_color_to_point_cloud(&mut cloud, &points_2d, &frames[0]);

            // Фильтрация по уверенности
            let initial_count = cloud.points.len();
            filter_point_cloud_by_confindence(&mut cloud, 0.25);
            info!(
                "Отфильтровано {} точек (оставлено {})",
                initial_count - cloud.points.len(),
                cloud.points.len()
            );
            info!("Обработка облака точек завершена");

            let filename = dest_path.join(format!("point_cloud_{current_frame}.ply"));

            match save_point_cloud(&cloud, &filename) {
                Ok(_) => info!(
                    "Облако точек успешно сохранено в файл: {}",
                    filename.display()
                ),
                Err(e) => error!("Ошибка при сохранении облака точек: {:?}", e),
            };

            prev_images = frames.clone();
        }

        Ok(())
    }
}
