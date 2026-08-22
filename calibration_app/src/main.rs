use std::{path::PathBuf, process::ExitCode};

mod app;
mod ui;
use clap::{Parser, Subcommand};
use lib_cv::calibration::save_calibration_to_yaml;
use lib_pipeline::config::CalibrationConfig;

/// CLI часть для headless запуска
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Headless-запуск калибровки
    Run {
        /// Путь к YAML к конфигу
        config_path: PathBuf,
        output_path: PathBuf,
    },
}

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .filter_module("calibration_app", log::LevelFilter::Debug)
        .filter_module("lib_cv", log::LevelFilter::Debug)
        .init();

    let cli = Cli::parse();

    let result = match cli.command {
        Some(Commands::Run {
            config_path,
            output_path,
        }) => run_cli(config_path, output_path),
        None => run_gui().map_err(|e| e.into()),
    };

    match result {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn run_cli(config_path: PathBuf, output_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let mut config = CalibrationConfig::load_yaml(&config_path)
        .map_err(|e| format!("Ошибка загрузки конфига: {}", e))?;
    config.output_path = output_path;
    let mut last_printed = 0u8;
    let mut on_progress = |p: f32| {
        let step = (p * 10.0) as u8;
        if step > last_printed {
            last_printed = step;
            eprintln!("Прогресс: {}%", step * 10);
        }
    };

    let calibration_data = lib_pipeline::runner::run_calibration(&config, &mut on_progress)
        .map_err(|e| format!("Ошибка калибровки: {}", e))?;
    save_calibration_to_yaml(&config.output_path, &calibration_data)
        .map_err(|e| format!("Ошибка сохранения калибровки: {}", e))?;
    println!(
        "Калибровка закончена и сохранена по пути:\n{}",
        config.output_path.to_string_lossy()
    );
    Ok(())
}

fn run_gui() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 700.0])
            .with_min_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Calibration",
        options,
        Box::new(|_cc| Ok(Box::new(app::CalibrationApp::default()))),
    )
}
