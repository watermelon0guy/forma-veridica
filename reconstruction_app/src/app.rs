use eframe::egui;
use lib_cv::calibration::{CameraParameters, load_camera_parameters};
use lib_cv::utils::split_video_into_quadrants;
use opencv::prelude::*;
use opencv::videoio;

use std::path::Path;
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

    fn fetch_project(&mut self, ui: &mut egui::Ui) {
        self.pipeline_state = PipelineState::SetupMenu;
    }
}
