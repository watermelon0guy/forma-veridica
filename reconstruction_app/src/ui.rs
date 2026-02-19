use crate::{app::ReconstructionApp, model::PipelineState};
use eframe::egui;
use log::error;

pub struct UiRenderer;

impl UiRenderer {
    pub(crate) fn render_content(app: &mut ReconstructionApp, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| match app.pipeline_state {
            PipelineState::FolderSetup => Self::render_folder_setup(app, ui),
            PipelineState::FetchProject => app.fetch_project(),
            PipelineState::SetupMenu => Self::render_setup_menu(app, ui),
            PipelineState::ReadyToProcess => todo!(),
        });
    }

    fn render_folder_setup(app: &mut ReconstructionApp, ui: &mut egui::Ui) {
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
                        app.set_project_folder(p); // Переходим к следующему этапу
                    }
                    None => return,
                }
            }
        });
    }

    fn render_setup_menu(app: &mut ReconstructionApp, ui: &mut egui::Ui) {
        ui.vertical_centered(|ui| {
            ui.label(egui::RichText::new(format!(
                "Путь проекта теперь установлен в {}",
                app.resources.project_path.as_ref().unwrap().display()
            )))
        });

        ui.columns(2, |columns| {
            Self::render_camera_parameters_setup(app, &mut columns[0]);
            Self::render_video_setup(app, &mut columns[1]);
        });

        Self::button_start_reconstruction(app, ui);
    }

    fn pick_camera_parameters_file(app: &mut ReconstructionApp) {
        if let Some(file_path) = rfd::FileDialog::new()
            .set_title("Выбрать файл параметров")
            .pick_file()
        {
            app.load_camera_parameters(file_path);
        }
    }

    fn render_camera_parameters_setup(app: &mut ReconstructionApp, ui: &mut egui::Ui) {
        ui.vertical_centered(|ui| {
            ui.heading("Параметры камеры");

            match &app.resources.calibration_data {
                None => {
                    ui.label(egui::RichText::new("Выберите файл с параметрами камер"));
                    let button = egui::Button::new(egui::RichText::new("Выбрать").size(18.0))
                        .min_size(egui::vec2(140.0, 40.0));

                    if ui.add(button).clicked() {
                        Self::pick_camera_parameters_file(app);
                    }
                }
                Some(calib_data) => {
                    let num_cam = calib_data.num_cameras;
                    ui.label(format!("В параметрах найдено {num_cam} камеры"));
                    let button =
                        egui::Button::new(egui::RichText::new("Изменить параметры").size(18.0))
                            .min_size(egui::vec2(140.0, 40.0));
                    if ui.add(button).clicked() {
                        Self::pick_camera_parameters_file(app);

                        match &app.resources.video_data {
                            Some(vd) => {
                                if vd.video_files.len() != num_cam {
                                    app.resources.video_data = None
                                }
                            }
                            None => (),
                        }
                    }
                }
            }
        });
    }

    fn render_video_setup(app: &mut ReconstructionApp, ui: &mut egui::Ui) {
        ui.vertical_centered(|ui| {
            ui.heading("Видео для анализа");

            match &app.resources.calibration_data {
                Some(cb) => {
                    for cam_num in 0..cb.num_cameras {
                        Self::button_to_choose_video(app, ui, cam_num);
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

            Self::button_to_choose_4_combined_video(app, ui);
        });
    }

    fn button_start_reconstruction(app: &mut ReconstructionApp, ui: &mut egui::Ui) {
        let is_enabled = app.resources.calibration_data.is_some()
            && app
                .resources
                .video_data
                .as_ref()
                .map_or(false, |vd| vd.video_files.iter().all(|vf| vf.is_some()));

        let button = egui::Button::new(egui::RichText::new("Начать реконструкцию").size(18.0))
            .min_size(egui::vec2(140.0, 40.0));
        ui.vertical_centered(|ui| {
            if ui.add_enabled(is_enabled, button).clicked() {
                if let Err(e) = app.run_pipeline() {
                    error!("Ошибка при выполнении пайплайна реконструкции: {}", e);
                }
            };
        });
    }

    fn button_to_choose_video(app: &mut ReconstructionApp, ui: &mut egui::Ui, cam_num: usize) {
        let action = match app
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
            app.pick_camera_video(cam_num);
        }
    }

    fn button_to_choose_4_combined_video(app: &mut ReconstructionApp, ui: &mut egui::Ui) {
        let button =
            egui::Button::new(egui::RichText::new("Выделить из комбинированного видео").size(18.0))
                .min_size(egui::vec2(140.0, 40.0));

        if ui.add(button).clicked() {
            app.pick_from_4_combined_video();
        }
    }
}
