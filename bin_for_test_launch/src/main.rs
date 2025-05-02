use lib_cv::calibration;
use opencv::core::{Scalar, Size};
use opencv::features2d::{DrawMatchesFlags, draw_keypoints, draw_matches_knn_def};
use opencv::gapi::crop;
use opencv::imgcodecs;
use opencv::objdetect::{CharucoBoard, draw_detected_corners_charuco, draw_detected_markers};
use opencv::videoio::VideoCapture;
use opencv::{highgui, prelude::*};

fn main() {
    highgui::named_window("Charuco Доска", highgui::WINDOW_KEEPRATIO).unwrap();
    let mut cap = VideoCapture::from_file(
        "/home/watermelon0guy/Видео/Experiments/raspberry_pi_cardboard/20250427_095907_hires.mp4",
        opencv::videoio::CAP_ANY,
    )
    .unwrap();
    let mut frame = opencv::core::Mat::default();
    cap.read(&mut frame).unwrap();

    let roi = opencv::core::Mat::roi(
        &frame,
        opencv::core::Rect::new(0, 0, frame.cols() / 2, frame.rows() / 2),
    )
    .unwrap();
    let mut cropped = Mat::default();
    roi.copy_to(&mut cropped).unwrap();
    let dictionary = opencv::objdetect::get_predefined_dictionary(
        opencv::objdetect::PredefinedDictionaryType::DICT_4X4_50,
    )
    .unwrap();
    let mut charuco_board = opencv::objdetect::CharucoBoard::new_def(
        opencv::core::Size::new(5, 5),
        10.0,
        7.0,
        &dictionary,
    )
    .unwrap();

    let (marker_corners, marker_ids, charuco_corners, charuco_ids, obj_points, img_points) =
        calibration::get_charuco(&charuco_board, &cropped).unwrap();
    let mut image_with_points = cropped.clone();

    draw_detected_corners_charuco(
        &mut image_with_points,
        &charuco_corners,
        &charuco_ids,
        Scalar::new(0.0, 255.0, 0.0, 255.0),
    )
    .unwrap();

    highgui::imshow("Charuco Доска", &image_with_points).unwrap();
    highgui::wait_key(0).unwrap();

    let mut image_with_markers = cropped.clone();
    draw_detected_markers(
        &mut image_with_markers,
        &marker_corners,
        &marker_ids,
        Scalar::new(255.0, 0.0, 0.0, 255.0),
    )
    .unwrap();

    highgui::imshow("Charuco Доска", &image_with_markers).unwrap();
    highgui::wait_key(0).unwrap();
}
