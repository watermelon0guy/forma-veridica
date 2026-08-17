use std::path::{Path, PathBuf};

use calib_targets::printable::CharucoTargetSpec;
use serde::{Deserialize, Serialize};
use vision_calibration::common::config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    pub cameras: Vec<CameraConfig>,
    pub output_path: PathBuf,
    pub frame_step: u64,
    pub charuco_board: CharucoTargetSpec,
}

impl CalibrationConfig {
    pub fn new(
        cameras: Vec<CameraConfig>,
        output_path: PathBuf,
        frame_step: u64,
        charuco_board: CharucoTargetSpec,
    ) -> Self {
        Self {
            cameras,
            output_path,
            frame_step,
            charuco_board,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.cameras.len() < 2 {
            return Err("Нужно минимум две камеры".to_owned());
        }

        if self.frame_step == 0 {
            return Err("frame_step должен быть больше нуля".to_owned());
        }

        for (index, camera) in self.cameras.iter().enumerate() {
            if camera.video_path.as_os_str().is_empty() {
                return Err(format!("Не указан путь для камеры {index}"));
            }

            if !camera.start_time_in_seconds.is_finite() || camera.start_time_in_seconds < 0.0 {
                return Err(format!("Некорректное начальное время камеры {index}"));
            }
        }

        Ok(())
    }

    pub fn save_to_yaml(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let yaml = serde_yml::to_string(self)?;
        std::fs::write(path, yaml)?;
        Ok(())
    }

    pub fn load_yaml(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;

        Ok(serde_yml::from_reader(file)?)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    pub video_path: PathBuf,
    pub start_time_in_seconds: f64,
}
