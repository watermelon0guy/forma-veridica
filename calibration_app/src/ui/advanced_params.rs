use std::ops::RangeInclusive;

use eframe::egui::{CollapsingHeader, DragValue, Ui, Window};
use lib_cv::calibration::{DatasetParams, DetectionParams, SolverParams};

use crate::app::CalibrationApp;

/// Окно дополнительных параметров калибровки. Показывается поверх экрана
/// выравнивания видео, когда `app.advanced_params_open == true`.
pub fn advanced_params_window(app: &mut CalibrationApp, ui: &Ui) {
    if !app.advanced_params_open {
        return;
    }

    Window::new("Дополнительные параметры калибровки")
        .collapsible(false)
        .open(&mut app.advanced_params_open)
        .show(ui.ctx(), |ui| {
            detection_params_ui(&mut app.calibration_config.detection, ui);
            dataset_params_ui(&mut app.calibration_config.dataset, ui);
            solver_params_ui(&mut app.calibration_config.solver, ui);
        });
}

pub fn detection_params_ui(params: &mut DetectionParams, ui: &mut Ui) {
    CollapsingHeader::new("Детекция (ArUco/ChArUco)")
        .default_open(true)
        .show(ui, |ui| {
            drag_f32(
                ui,
                "Отступ от границы кадра, px",
                "Маркеры, касающиеся края кадра ближе этого расстояния, отбрасываются",
                &mut params.border_px,
                0.0..=50.0,
            );
            drag_f32(
                ui,
                "Мин. размер маркера",
                "Нижняя граница периметра маркера относительно медианного",
                &mut params.min_size_rel,
                0.1..=1.0,
            );
            drag_f32(
                ui,
                "Макс. размер маркера",
                "Верхняя граница периметра маркера относительно медианного",
                &mut params.max_size_rel,
                1.0..=3.0,
            );
            drag_u32(
                ui,
                "Окно уточнения углов, px",
                "Размер окна cornerSubPix для углов маркеров",
                &mut params.refine_window_px,
                1..=20,
            );
            drag_usize(
                ui,
                "Итераций уточнения",
                "Максимальное число итераций cornerSubPix",
                &mut params.refine_iterations,
                1..=100,
            );
            drag_f32(
                ui,
                "Точность уточнения",
                "Критерий остановки cornerSubPix (меньше — точнее и дольше)",
                &mut params.refine_epsilon,
                0.001..=1.0,
            );
            drag_f32(
                ui,
                "Мин. чистота границы маркера",
                "Минимальная «чистота границы» маркера для принятия декодирования",
                &mut params.min_border_score,
                0.0..=1.0,
            );
            drag_usize(
                ui,
                "Мин. маркеров на угол",
                "Минимальное число маркеров для интерполяции угла; усредняются максимум 2 ближайшие проекции, значения выше 2 работают как более строгий отбор",
                &mut params.min_markers_per_corner,
                1..=4,
            );
            drag_f32(
                ui,
                "Отступ углов от границы, px",
                "Интерполированные углы ближе этого расстояния к краю кадра отбрасываются",
                &mut params.corner_border_margin_px,
                0.0..=20.0,
            );
        });
}

pub fn dataset_params_ui(params: &mut DatasetParams, ui: &mut Ui) {
    CollapsingHeader::new("Отбор кадров (admission)")
        .default_open(true)
        .show(ui, |ui| {
            drag_usize(
                ui,
                "Мин. углов на камеру",
                "Минимальное число углов ChArUco на камеру, чтобы детекция попала в датасет",
                &mut params.min_corners_per_view,
                4..=60,
            );
            drag_usize(
                ui,
                "Мин. камер на кадр",
                "Минимальное число камер с валидной детекцией, чтобы кадр попал в rig-датасет",
                &mut params.min_cameras_per_frame,
                2..=8,
            );
        });
}

pub fn solver_params_ui(params: &mut SolverParams, ui: &mut Ui) {
    CollapsingHeader::new("Solver")
        .default_open(true)
        .show(ui, |ui| {
            drag_f64(
                ui,
                "Порог reproj-ошибки, px",
                "Точки с ошибкой перепроекции выше этого порога отбрасываются",
                &mut params.max_reproj_error,
                0.1..=10.0,
            );
            drag_usize(
                ui,
                "Мин. точек на view",
                "Минимальное число точек на view после фильтрации",
                &mut params.min_points_per_view,
                4..=40,
            );
            ui.checkbox(&mut params.remove_sparse_views, "Удалять разреженные view")
                .on_hover_text(
                    "Удалять view целиком, если в нём осталось меньше минимального числа точек",
                );
        });
}

fn drag_f32(ui: &mut Ui, label: &str, hint: &str, value: &mut f32, range: RangeInclusive<f32>) {
    ui.horizontal(|ui| {
        ui.label(label).on_hover_text(hint);
        ui.add(DragValue::new(value).range(range).speed(0.05));
    });
}

fn drag_f64(ui: &mut Ui, label: &str, hint: &str, value: &mut f64, range: RangeInclusive<f64>) {
    ui.horizontal(|ui| {
        ui.label(label).on_hover_text(hint);
        ui.add(DragValue::new(value).range(range).speed(0.05));
    });
}

fn drag_u32(ui: &mut Ui, label: &str, hint: &str, value: &mut u32, range: RangeInclusive<u32>) {
    ui.horizontal(|ui| {
        ui.label(label).on_hover_text(hint);
        ui.add(DragValue::new(value).range(range));
    });
}

fn drag_usize(
    ui: &mut Ui,
    label: &str,
    hint: &str,
    value: &mut usize,
    range: RangeInclusive<usize>,
) {
    ui.horizontal(|ui| {
        ui.label(label).on_hover_text(hint);
        ui.add(DragValue::new(value).range(range));
    });
}
