use opencv::core::{Point2f, Vector};
use opencv::objdetect::{CharucoBoard, CharucoDetector};
use opencv::prelude::*;
use opencv::{self, Error};

fn get_charuco(
    charuco_board: CharucoBoard,
    img: Mat,
) -> Result<(opencv::core::Mat, opencv::core::Mat), opencv::Error> {
    let charuco_detector = CharucoDetector::new_def(&charuco_board)?;
    let mut charuco_corners: Vector<Point2f> = Vector::new();
    let mut charuco_ids: Vector<i32> = Vector::new();
    charuco_detector.detect_board_def(&img, &mut charuco_corners, &mut charuco_ids)?;

    let mut obj_points: Mat = Mat::default();
    let mut img_points: Mat = Mat::default();
    charuco_board.match_image_points(
        &charuco_corners,
        &charuco_ids,
        &mut obj_points,
        &mut img_points,
    )?;

    Ok((obj_points, img_points))
}
