use opencv::calib3d::calibrate_camera;
use opencv::core::{Point2f, TermCriteria, Vector};
use opencv::objdetect::{CharucoBoard, CharucoDetector};
use opencv::prelude::*;
use opencv::{self, Error};

fn get_charuco(
    charuco_board: CharucoBoard,
    img: Mat,
) -> Result<(opencv::core::Mat, opencv::core::Mat), Error> {
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

fn calibrate_with_charuco(
    imgs: Vector<Mat>,
    charuco_board: CharucoBoard,
) -> Result<
    (
        f64,
        Mat,
        Mat,
        Vector<Mat>,
        Vector<Mat>,
        Vector<Mat>,
        Vector<Mat>,
    ),
    Error,
> {
    let charuco_detector = CharucoDetector::new_def(&charuco_board)?;

    let mut all_charuco_corners = Vector::<Vector<Point2f>>::new();
    let mut all_charuco_ids = Vector::<Vector<i32>>::new();
    let mut all_object_points = Vector::<Mat>::new();
    let mut all_image_points = Vector::<Mat>::new();

    let img_size = imgs.get(0)?.size()?;

    for img in imgs {
        let mut charuco_corners: Vector<Point2f> = Vector::new();
        let mut charuco_ids: Vector<i32> = Vector::new();
        charuco_detector.detect_board_def(&img, &mut charuco_corners, &mut charuco_ids)?;
        if charuco_corners.is_empty() {
            continue;
        }

        let mut obj_points = Mat::default();
        let mut img_points = Mat::default();

        charuco_board.match_image_points(
            &charuco_corners,
            &charuco_ids,
            &mut obj_points,
            &mut img_points,
        )?;

        if obj_points.empty() || img_points.empty() {
            continue;
        }
        all_charuco_corners.push(charuco_corners);
        all_charuco_ids.push(charuco_ids);
        all_object_points.push(obj_points);
        all_image_points.push(img_points);
    }

    let mut camera_matrix = Mat::default();
    let mut dist_coeffs = Mat::default();
    let mut r_vecs = Vector::<Mat>::new();
    let mut t_vecs = Vector::<Mat>::new();

    let criteria = TermCriteria::new(
        opencv::core::TermCriteria_COUNT + opencv::core::TermCriteria_EPS,
        30,
        f64::EPSILON,
    )?;

    let ret = calibrate_camera(
        &all_object_points,
        &all_image_points,
        img_size,
        &mut camera_matrix,
        &mut dist_coeffs,
        &mut r_vecs,
        &mut t_vecs,
        0,
        criteria,
    )?;

    Ok((
        ret,
        camera_matrix,
        dist_coeffs,
        r_vecs,
        t_vecs,
        all_object_points,
        all_image_points,
    ))
}
