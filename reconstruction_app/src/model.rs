use std::path::PathBuf;

use lib_cv::{calibration::CameraParameters, utils::get_video_frame_count};

#[derive(Default)]
pub(crate) struct ProjectResources {
    pub project_path: Option<PathBuf>,
    pub calibration_data: Option<CalibrationData>,
    pub video_data: Option<VideoData>,
}

pub(crate) struct CalibrationData {
    pub(crate) calibration_file: PathBuf,
    pub(crate) camera_params: Vec<CameraParameters>,
    pub(crate) num_cameras: usize,
}

impl CalibrationData {
    pub(crate) fn new(calibration_file: PathBuf, camera_params: Vec<CameraParameters>) -> Self {
        let num_cameras = camera_params.len();
        Self {
            calibration_file,
            camera_params,
            num_cameras,
        }
    }
}

pub(crate) struct VideoData {
    pub(crate) video_files: Vec<Option<PathBuf>>,
    pub(crate) total_frames: usize,
}

impl VideoData {
    pub(crate) fn new(
        video_file: &PathBuf,
        cam_i: usize,
        num_cams: usize,
    ) -> Result<Self, opencv::Error> {
        let mut video_files = vec![None; num_cams];
        video_files[cam_i] = Some(video_file.clone());
        let total_frames = get_video_frame_count(video_file)?;
        Ok(Self {
            video_files,
            total_frames,
        })
    }

    pub(crate) fn from_vec(video_files: Vec<Option<PathBuf>>) -> Result<Self, opencv::Error> {
        let total_frames = {
            let first_video = video_files
                .get(0)
                .ok_or(opencv::Error::new(-1, "No video files provided"))?
                .as_ref()
                .ok_or(opencv::Error::new(-1, "First video path is None"))?;
            get_video_frame_count(first_video)?
        };
        Ok(Self {
            video_files,
            total_frames,
        })
    }
}

#[derive(Default)]
pub(crate) enum PipelineState {
    #[default]
    FolderSetup,
    FetchProject,
    SetupMenu,
    ReadyToProcess,
}
