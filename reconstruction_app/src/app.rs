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
    CalibrationSetup,
    VideoSetup,
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
            PipelineState::CalibrationSetup => self.render_calibration_setup(ui),
            PipelineState::VideoSetup => todo!(),
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
                        self.pipeline_state = PipelineState::CalibrationSetup // Переходим к следующему этапу
                    }
                    None => return,
                }
            }
        });
    }

    fn render_calibration_setup(&mut self, ui: &mut egui::Ui) {
        ui.vertical_centered(|ui| {
            ui.label(egui::RichText::new(format!(
                "Путь проекта теперь установлен в {}",
                self.resources.project_path.as_ref().unwrap().display()
            )));
            ui.label(egui::RichText::new("Выберите файл с параметр камер").size(18.0));
            let button = egui::Button::new(egui::RichText::new("Выбрать").size(18.0))
                .min_size(egui::vec2(140.0, 40.0));

            if ui.add(button).clicked() {
                match rfd::FileDialog::new()
                    .set_title("Выбрать файл параметров")
                    .pick_file()
                {
                    Some(p) => {
                        let project_path = &self.resources.project_path.as_ref().unwrap();
                        let dest_path = project_path.join(p.file_name().unwrap());
                        if let Err(_) = std::fs::copy(&p, &dest_path) {
                            return;
                        }

                        let cam_params = match load_camera_parameters(dest_path.to_str().unwrap()) {
                            Ok(c) => c,
                            Err(_) => return,
                        };
                        self.resources.calibration_data =
                            Some(CalibrationData::new(dest_path, cam_params));

                        self.pipeline_state = PipelineState::VideoSetup // Переходим к следующему этапу
                    }
                    None => return,
                }
            }
        });
    }
}
