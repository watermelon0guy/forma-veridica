use std::path::{Path, PathBuf};

use calib_targets::printable::CharucoTargetSpec;
use lib_cv::calibration::{DatasetParams, DetectionParams, SolverParams};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    pub cameras: Vec<CameraConfig>,
    pub output_path: PathBuf,
    pub frame_step: u64,
    pub charuco_board: CharucoTargetSpec,
    pub detection: DetectionParams,
    pub dataset: DatasetParams,
    pub solver: SolverParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    pub video_path: PathBuf,
    pub start_time_in_seconds: f64,
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
            detection: DetectionParams::default(),
            dataset: DatasetParams::default(),
            solver: SolverParams::default(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.cameras.len() < 2 {
            return Err("Нужно минимум две камеры".to_owned());
        }

        if self.frame_step == 0 {
            return Err("frame_step должен быть больше нуля".to_owned());
        }

        if self.dataset.min_corners_per_view < 4 {
            return Err("min_corners_per_view должен быть не меньше 4".to_owned());
        }

        if self.dataset.min_cameras_per_frame < 2 {
            return Err("min_cameras_per_frame должен быть не меньше 2".to_owned());
        }

        if self.solver.max_reproj_error <= 0.0 {
            return Err("max_reproj_error должен быть больше нуля".to_owned());
        }

        if self.solver.min_points_per_view < 4 {
            return Err("min_points_per_view должен быть не меньше 4".to_owned());
        }

        let det = &self.detection;
        if det.border_px < 0.0 {
            return Err("border_px не может быть отрицательным".to_owned());
        }

        if det.min_size_rel > det.max_size_rel {
            return Err("min_size_rel не может быть больше max_size_rel".to_owned());
        }

        if !(0.0..=1.0).contains(&det.min_border_score) {
            return Err("min_border_score должен быть в диапазоне [0, 1]".to_owned());
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
        let config: Self = serde_yml::from_reader(file)?;
        config.validate()?;
        Ok(config)
    }
}
