use std::ops::RangeInclusive;

use eframe::egui::{CollapsingHeader, DragValue, Ui, Window};
use lib_pipeline::config::ReconstructionParams;

use crate::app::ReconstructionApp;

/// Окно параметров реконструкции. Показывается поверх экрана
/// выравнивания видео, когда `app.advanced_params_open == true`.
pub fn advanced_params_window(app: &mut ReconstructionApp, ui: &Ui) {
    if !app.advanced_params_open {
        return;
    }

    Window::new("Параметры реконструкции")
        .collapsible(false)
        .open(&mut app.advanced_params_open)
        .show(ui.ctx(), |ui| {
            reconstruction_params_ui(&mut app.reconstruction_config.params, ui);
        });
}

fn reconstruction_params_ui(params: &mut ReconstructionParams, ui: &mut Ui) {
    CollapsingHeader::new("Реконструкция")
        .default_open(true)
        .show(ui, |ui| {
            drag_u64(
                ui,
                "Шаг кадров",
                "Обрабатывать каждый N-й кадр видео",
                &mut params.frame_step,
                1..=50,
            );
            drag_f64(
                ui,
                "Эпиполярный порог, px",
                "Максимальное отклонение точки от эпиполярной линии, при котором матч считается валидным",
                &mut params.epipolar_threshold_px,
                0.5..=50.0,
            );
            drag_odd_usize(
                ui,
                "Окно optical flow, px",
                "Размер окна поиска Lucas-Kanade (нечётное)",
                &mut params.lk_window,
                3..=31,
            );
            drag_usize(
                ui,
                "Итераций optical flow",
                "Максимальное число итераций уточнения положения точки",
                &mut params.lk_max_iterations,
                1..=100,
            );
            drag_usize(
                ui,
                "Уровней пирамиды",
                "Число уровней пирамиды изображений для optical flow",
                &mut params.lk_pyramid_levels,
                1..=8,
            );
            drag_f32(
                ui,
                "Мин. уверенность точки",
                "Точки облака с уверенностью ниже порога отбрасываются",
                &mut params.min_confidence,
                0.0..=1.0,
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

fn drag_u64(ui: &mut Ui, label: &str, hint: &str, value: &mut u64, range: RangeInclusive<u64>) {
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

/// DragValue для значения, которое обязано быть нечётным (например, окно LK).
/// Чётное значение, введённое пользователем, сдвигается вниз к ближайшему нечётному.
fn drag_odd_usize(
    ui: &mut Ui,
    label: &str,
    hint: &str,
    value: &mut usize,
    range: RangeInclusive<usize>,
) {
    ui.horizontal(|ui| {
        ui.label(label).on_hover_text(hint);
        let response = ui.add(DragValue::new(value).range(range));
        if response.changed() && *value % 2 == 0 {
            *value = (*value).saturating_sub(1).max(3);
        }
    });
}
