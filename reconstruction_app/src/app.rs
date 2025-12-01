use eframe::egui;
use lib_cv::calibration::{CameraParameters, load_camera_parameters};
use std::path::PathBuf;

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
    video_files: Vec<PathBuf>,
    total_frames: usize,
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
    fn update(&mut self, ctx: &eframe::egui::Context, frame: &mut eframe::Frame) {
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
        ui.label(egui::RichText::new(format!(
            "Путь проекта теперь установлен в {}",
            self.resources.project_path.as_ref().unwrap().display()
        )));
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
                    ui.label(egui::RichText::new("Выберите файл с параметр камер"));
                    let button = egui::Button::new(egui::RichText::new("Выбрать").size(18.0))
                        .min_size(egui::vec2(140.0, 40.0));

                    if ui.add(button).clicked() {
                        self.pick_camera_parameters_file();
                    }
                }
                Some(calib_data) => {
                    let num_cam = calib_data.num_cameras;
                    ui.label(format!("В параметрах найдено {num_cam}"));
                    let button =
                        egui::Button::new(egui::RichText::new("Изменить параметры").size(18.0))
                            .min_size(egui::vec2(140.0, 40.0));
                    if ui.add(button).clicked() {
                        self.pick_camera_parameters_file();
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
        ui.vertical_centered(|ui| ui.heading("Видео для анализа"));
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
        self.fetch_camera_params();
        self.pipeline_state = PipelineState::SetupMenu;
    }
}
