use eframe;
use generate_calibration_pattern::GenCalibPatternApp;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 700.0])
            .with_min_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "ChArUco Generator",
        options,
        Box::new(|_cc| Ok(Box::new(GenCalibPatternApp::new()))),
    )
}
