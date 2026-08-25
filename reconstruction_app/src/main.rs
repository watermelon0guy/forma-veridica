use std::{path::PathBuf, process::ExitCode};

use eframe;
mod app;
mod model;
mod ui;

use clap::{Parser, Subcommand};
use lib_cv::calibration::load_calibration_from_yaml;
use lib_pipeline::config::ReconstructionConfig;

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
        output_dir: PathBuf,
    },
}

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    let result = match cli.command {
        Some(Commands::Run {
            config_path,
            output_dir,
        }) => run_cli(config_path, output_dir),
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

fn run_cli(config_path: PathBuf, output_dir: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let mut config = ReconstructionConfig::load_yaml(&config_path)
        .map_err(|e| format!("Ошибка загрузки конфига: {}", e))?;

    if !output_dir.is_dir() {
        return Err(format!(
            "Указанный путь '{}' не является папкой",
            output_dir.to_string_lossy()
        )
        .into());
    }

    config.output_dir = output_dir;
    let mut last_printed = 0u8;
    let mut on_progress = |p: f32| {
        let step = (p * 10.0) as u8;
        if step > last_printed {
            last_printed = step;
            eprintln!("Прогресс: {}%", step * 10);
        }
    };

    let calibration_data = load_calibration_from_yaml(&config.calibration_path)?;

    let _ = lib_pipeline::runner::run_reconstruction(&config, &calibration_data, &mut on_progress)
        .map_err(|e| format!("Ошибка реконструкции: {}", e))?;
    println!(
        "Реконструкция закончена и данные сохранены по пути:\n{}",
        config.output_dir.to_string_lossy()
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
        "Reconstruction",
        options,
        Box::new(|_cc| Ok(Box::new(app::ReconstructionApp::new()))),
    )
}
