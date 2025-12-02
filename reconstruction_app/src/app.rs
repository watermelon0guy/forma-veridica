use eframe::egui;
use lib_cv::calibration::{CameraParameters, load_camera_parameters};
use lib_cv::correspondence::gather_points_2d_from_matches;
use lib_cv::reconstruction::{
    PointCloud, add_color_to_point_cloud, filter_point_cloud_by_confindence,
    match_first_camera_features_to_all, min_visible_match_set, save_point_cloud,
    undistort_points_single_camera,
};
use lib_cv::utils::{split_video_into_quadrants, vector_point2f_to_mat};
use log::{debug, error, info};
use opencv::core::{Point2f, Vector};
use opencv::video::calc_optical_flow_pyr_lk;
use opencv::videoio::{self, VideoCapture};
use opencv::{Error, prelude::*};

use std::{fs::create_dir_all, path::PathBuf};

pub struct ReconstructionApp {
    resources: ProjectResources,
    pipeline_state: PipelineState,
}

impl Default for ReconstructionApp {
    fn default() -> Self {
        Self {
            resources: Default::default(),
            pipeline_state: Default::default(),
        }
    }
}

#[derive(Default)]
struct ProjectResources {
    project_path: Option<PathBuf>,
    calibration_data: Option<CalibrationData>,
    video_data: Option<VideoData>,
}

struct CalibrationData {
    calibration_file: PathBuf,
    camera_params: Vec<CameraParameters>,
    num_cameras: usize,
}

impl CalibrationData {
    fn new(calibration_file: PathBuf, camera_params: Vec<CameraParameters>) -> Self {
        let num_cameras = camera_params.len();
        Self {
            calibration_file,
            camera_params,
            num_cameras,
        }
    }
}

struct VideoData {
    video_files: Vec<Option<PathBuf>>,
    total_frames: usize,
}

impl VideoData {
    fn new(video_file: &PathBuf, cam_i: usize, num_cams: usize) -> Result<Self, opencv::Error> {
        let mut video_files = vec![None; num_cams];
        video_files[cam_i] = Some(video_file.clone());
        let total_frames = {
            let cap =
                videoio::VideoCapture::from_file(&video_file.to_string_lossy(), videoio::CAP_ANY)?;
            cap.get(videoio::CAP_PROP_FRAME_COUNT)? as usize
        };
        Ok(Self {
            video_files,
            total_frames,
        })
    }

    fn from_vec(video_files: Vec<Option<PathBuf>>) -> Result<Self, opencv::Error> {
        let total_frames = {
            let first_video = video_files
                .get(0)
                .ok_or(opencv::Error::new(-1, "No video files provided"))?
                .as_ref()
                .ok_or(opencv::Error::new(-1, "First video path is None"))?;
            let cap =
                videoio::VideoCapture::from_file(&first_video.to_string_lossy(), videoio::CAP_ANY)?;
            cap.get(videoio::CAP_PROP_FRAME_COUNT)? as usize
        };
        Ok(Self {
            video_files,
            total_frames,
        })
    }
}

#[derive(Default)]
enum PipelineState {
    #[default]
    FolderSetup,
    FetchProject,
    SetupMenu,
    ReadyToProcess,
}

impl eframe::App for ReconstructionApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        self.render_content(ctx);
    }
}

impl ReconstructionApp {
    pub fn new() -> Self {
        Self::default()
    }

    fn render_content(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| match self.pipeline_state {
            PipelineState::FolderSetup => self.render_folder_setup(ui),
            PipelineState::FetchProject => self.fetch_project(ui),
            PipelineState::SetupMenu => self.render_setup_menu(ui),
            PipelineState::ReadyToProcess => todo!(),
        });
    }

    fn render_folder_setup(&mut self, ui: &mut egui::Ui) {
        ui.vertical_centered(|ui| {
            ui.label(
                egui::RichText::new("Выберите папку где находится или будет находится проект")
                    .size(18.0),
            );
            let button = egui::Button::new(egui::RichText::new("Выбрать").size(18.0))
                .min_size(egui::vec2(140.0, 40.0));

            if ui.add(button).clicked() {
                match rfd::FileDialog::new()
                    .set_title("Выбрать папку проекта")
                    .pick_folder()
                {
                    Some(p) => {
                        self.resources = ProjectResources {
                            project_path: Some(p),
                            calibration_data: None,
                            video_data: None,
                        };
                        self.pipeline_state = PipelineState::FetchProject // Переходим к следующему этапу
                    }
                    None => return,
                }
            }
        });
    }

    fn render_setup_menu(&mut self, ui: &mut egui::Ui) {
        ui.vertical_centered(|ui| {
            ui.label(egui::RichText::new(format!(
                "Путь проекта теперь установлен в {}",
                self.resources.project_path.as_ref().unwrap().display()
            )))
        });

        ui.columns(2, |columns| {
            self.render_camera_parameters_setup(&mut columns[0]);
            self.render_video_setup(&mut columns[1]);
        });

        self.button_start_reconstruction(ui);
    }

    fn button_start_reconstruction(&mut self, ui: &mut egui::Ui) {
        let is_enabled = self.resources.calibration_data.is_some()
            && self
                .resources
                .video_data
                .as_ref()
                .map_or(false, |vd| vd.video_files.iter().all(|vf| vf.is_some()));

        let button = egui::Button::new(egui::RichText::new("Начать реконструкцию").size(18.0))
            .min_size(egui::vec2(140.0, 40.0));
        ui.vertical_centered(|ui| {
            if ui.add_enabled(is_enabled, button).clicked() {
                self.run_pipeline();
            };
        });
    }

    fn render_camera_parameters_setup(&mut self, ui: &mut egui::Ui) {
        ui.vertical_centered(|ui| {
            ui.heading("Параметры камеры");

            match &self.resources.calibration_data {
                None => {
                    ui.label(egui::RichText::new("Выберите файл с параметрами камер"));
                    let button = egui::Button::new(egui::RichText::new("Выбрать").size(18.0))
                        .min_size(egui::vec2(140.0, 40.0));

                    if ui.add(button).clicked() {
                        self.pick_camera_parameters_file();
                    }
                }
                Some(calib_data) => {
                    let num_cam = calib_data.num_cameras;
                    ui.label(format!("В параметрах найдено {num_cam} камеры"));
                    let button =
                        egui::Button::new(egui::RichText::new("Изменить параметры").size(18.0))
                            .min_size(egui::vec2(140.0, 40.0));
                    if ui.add(button).clicked() {
                        self.pick_camera_parameters_file();

                        match &self.resources.video_data {
                            Some(vd) => {
                                if vd.video_files.len() != num_cam {
                                    self.resources.video_data = None
                                }
                            }
                            None => (),
                        }
                    }
                }
            }
        });
    }

    fn pick_camera_parameters_file(&mut self) {
        match rfd::FileDialog::new()
            .set_title("Выбрать файл параметров")
            .pick_file()
        {
            Some(file_path) => {
                let project_path = self.resources.project_path.as_ref().unwrap();
                let dest_path = project_path.join("camera_parameters.yml");

                if let Err(_) = std::fs::copy(&file_path, &dest_path) {
                    return;
                }

                let cam_params = match load_camera_parameters(dest_path.to_str().unwrap()) {
                    Ok(c) => c,
                    Err(_) => return,
                };
                self.resources.calibration_data = Some(CalibrationData::new(dest_path, cam_params));
            }
            None => return,
        }
    }

    fn render_video_setup(&mut self, ui: &mut egui::Ui) {
        ui.vertical_centered(|ui| {
            ui.heading("Видео для анализа");

            match &self.resources.calibration_data {
                Some(cb) => {
                    for cam_num in 0..cb.num_cameras {
                        self.button_to_choose_video(ui, cam_num);
                    }
                }
                None => {
                    ui.label(
                        egui::RichText::new("Выберите параметры камер")
                            .size(18.0)
                            .color(egui::Color32::YELLOW),
                    );
                }
            }

            self.button_to_choose_4_combined_video(ui);
        });
    }

    fn button_to_choose_video(&mut self, ui: &mut egui::Ui, cam_num: usize) {
        let action = match self
            .resources
            .video_data
            .as_ref()
            .and_then(|vd| vd.video_files.get(cam_num))
        {
            Some(Some(_)) => "Изменить",
            _ => "Выбрать",
        };

        let button = egui::Button::new(
            egui::RichText::new(format!(
                "{} видео для {} камеры",
                action,
                cam_num + 1 // TODO решить как печатать номера камер. Всегда ли с +1?
            ))
            .size(18.0),
        )
        .min_size(egui::vec2(140.0, 40.0));

        if ui.add(button).clicked() {
            self.pick_camera_video(cam_num);
        }
    }

    fn pick_camera_video(&mut self, cam_num: usize) {
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

    fn button_to_choose_4_combined_video(&mut self, ui: &mut egui::Ui) {
        let button =
            egui::Button::new(egui::RichText::new("Выделить из комбинированного видео").size(18.0))
                .min_size(egui::vec2(140.0, 40.0));

        if ui.add(button).clicked() {
            self.pick_from_4_combined_video();
        }
    }

    fn pick_from_4_combined_video(&mut self) {
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

                if let Ok(paths) = split_video_into_quadrants(&file_path, &dest_path, "camera") {
                    let paths: Vec<Option<PathBuf>> =
                        paths.iter().map(|p| Some(p.clone())).collect();
                    if let Ok(vd) = VideoData::from_vec(paths) {
                        self.resources.video_data = Some(vd);
                    }
                }
            }
            None => return,
        }
    }

    fn fetch_camera_params(&mut self) {
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

    fn fetch_video_data(&mut self) {
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

    fn fetch_project(&mut self, _ui: &mut egui::Ui) {
        self.fetch_camera_params();
        self.fetch_video_data();
        self.pipeline_state = PipelineState::SetupMenu;
    }

    fn run_pipeline(&self) -> Result<(), opencv::Error> {
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

        let mut current_frame: usize = 0;

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

fn open_video_captures(
    caps: &mut Vec<VideoCapture>,
    video_files: &Vec<Option<PathBuf>>,
) -> Result<(), Error> {
    Ok(for video_file in video_files.iter() {
        let cap = VideoCapture::from_file(
            video_file
                .as_ref()
                .ok_or_else(|| Error::new(-1, "Неправильный путь к видео"))?
                .to_str()
                .ok_or_else(|| Error::new(-1, "Путь к видео не является валидной UTF-8 строкой"))?,
            opencv::videoio::CAP_ANY,
        )?;
        caps.push(cap);
    })
}

fn read_frames(caps: &mut Vec<VideoCapture>, frames: &mut Vec<Mat>) -> Result<(), Error> {
    for (i, cap) in caps.iter_mut().enumerate() {
        let mut frame = &mut frames[i];
        cap.read(&mut frame)?;
    }
    Ok(())
}
