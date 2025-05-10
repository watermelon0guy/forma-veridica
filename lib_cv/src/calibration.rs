use std::collections::HashSet;

use opencv::calib3d::{calibrate_camera, stereo_calibrate};
use opencv::core::{Point2f, TermCriteria, TermCriteria_Type, Vector};
use opencv::objdetect::{CharucoBoard, CharucoDetector};
use opencv::prelude::*;
use opencv::{self, Error};

pub fn get_charuco(
    charuco_board: &CharucoBoard,
    img: &Mat,
) -> Result<
    (
        Vector<Vector<Point2f>>,
        Vector<i32>,
        Vector<Point2f>,
        Vector<i32>,
        Mat,
        Mat,
    ),
    Error,
> {
    let charuco_detector = CharucoDetector::new_def(charuco_board)?;
    let mut charuco_corners: Vector<Point2f> = Vector::new();
    let mut charuco_ids: Vector<i32> = Vector::new();
    let mut marker_corners: Vector<Vector<Point2f>> = Vector::new();
    let mut marker_ids: Vector<i32> = Vector::new();
    charuco_detector.detect_board(
        &img,
        &mut charuco_corners,
        &mut charuco_ids,
        &mut marker_corners,
        &mut marker_ids,
    )?;

    let mut obj_points: Mat = Mat::default();
    let mut img_points: Mat = Mat::default();
    let _ = charuco_board.match_image_points(
        &charuco_corners,
        &charuco_ids,
        &mut obj_points,
        &mut img_points,
    );

    Ok((
        marker_corners,
        marker_ids,
        charuco_corners,
        charuco_ids,
        obj_points,
        img_points,
    ))
}

pub fn calibrate_with_charuco(
    imgs: &Vector<Mat>,
    charuco_board: &CharucoBoard,
) -> Result<
    (
        f64,
        Mat,
        Mat,
        Vector<Mat>,
        Vector<Mat>,
        Vector<Mat>,
        Vector<Mat>,
        Vector<Vector<i32>>,
        Vector<Vector<Point2f>>,
    ),
    Error,
> {
    let charuco_detector = CharucoDetector::new_def(charuco_board)?;

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
        all_charuco_ids,
        all_charuco_corners,
    ))
}

pub fn calibrate_multiple_with_charuco(
    imgs: &Vec<Vector<Mat>>,
    charuco_board: &CharucoBoard,
) -> Result<Vec<CameraParameters>, opencv::Error> {
    let mut ret: Vec<f64> = Vec::default();
    let mut camera_matrix: Vec<Mat> = Vec::default();
    let mut dist_coeffs: Vec<Mat> = Vec::default();
    let mut r_vecs: Vec<Vector<Mat>> = Vec::default();
    let mut t_vecs: Vec<Vector<Mat>> = Vec::default();
    let mut object_points: Vec<Vector<Mat>> = Vec::default();
    let mut image_points: Vec<Vector<Mat>> = Vec::default();
    let mut charuco_ids: Vec<Vector<Vector<i32>>> = Vec::default();
    let mut charuco_corners: Vec<Vector<Vector<Point2f>>> = Vec::default();

    if imgs.len() < 2 {
        eprintln!("Ошибка: для калибровки требуется как минимум 2 набора изображений");
        return Ok(vec![]);
    }

    for img_set in imgs {
        match calibrate_with_charuco(img_set, charuco_board) {
            Ok((
                curr_cam_ret_val,
                curr_cam_camera_matrix_val,
                curr_cam_dist_coeffs_val,
                curr_cam_r_vecs_val,
                curr_cam_t_vecs_val,
                curr_cam_all_object_points_val,
                curr_cam_all_image_points_val,
                curr_cam_all_charuco_ids,
                curr_cam_charuco_corners,
            )) => {
                ret.push(curr_cam_ret_val);
                camera_matrix.push(curr_cam_camera_matrix_val);
                dist_coeffs.push(curr_cam_dist_coeffs_val);
                r_vecs.push(curr_cam_r_vecs_val);
                t_vecs.push(curr_cam_t_vecs_val);
                object_points.push(curr_cam_all_object_points_val);
                image_points.push(curr_cam_all_image_points_val);
                charuco_ids.push(curr_cam_all_charuco_ids);
                charuco_corners.push(curr_cam_charuco_corners);
            }
            Err(e) => eprintln!("Ошибка калибровки calibrate_with_charuco: {:?}", e),
        }
    }

    let camera_count = camera_matrix.len();

    let criteria = TermCriteria::new(
        TermCriteria_Type::COUNT as i32 | TermCriteria_Type::EPS as i32,
        50,
        1e-6,
    )
    .unwrap();

    let mut cameras = Vec::with_capacity(camera_count);

    // Параметры для первой камеры (основной). Вообще можно сделать выбор основной камеры кастомизируемый.
    cameras.push(CameraParameters {
        intrinsic: camera_matrix[0].clone(),
        distortion: dist_coeffs[0].clone(),
        ..CameraParameters::new().unwrap()
    });

    for i in 1..camera_count {
        let mut common_object_points = Vector::<Mat>::new();
        let mut common_image_points1 = Vector::<Mat>::new();
        let mut common_image_points2 = Vector::<Mat>::new();

        for frame_idx in 0..charuco_ids[0].len() {
            let ids_cam1 = &charuco_ids[0].get(frame_idx)?;
            let ids_cam2 = &charuco_ids[i].get(frame_idx)?;

            let common: HashSet<i32> = find_common_points(&[ids_cam1.clone(), ids_cam2.clone()]);
            if common.len() < 10 {
                continue;
            }

            let mut idx_cam1 = Vector::<i32>::new();
            let mut idx_cam2 = Vector::<i32>::new();

            for (pos, id) in ids_cam1.iter().enumerate() {
                if common.contains(&id) {
                    idx_cam1.push(pos as i32);
                }
            }
            for (pos, id) in ids_cam2.iter().enumerate() {
                if common.contains(&id) {
                    idx_cam2.push(pos as i32);
                }
            }

            common_object_points.push(select_rows(&object_points[0].get(frame_idx)?, &idx_cam1)?);
            common_image_points1.push(select_rows(&image_points[0].get(frame_idx)?, &idx_cam1)?);
            common_image_points2.push(select_rows(&image_points[i].get(frame_idx)?, &idx_cam2)?);
        }

        let img_size = imgs[0].get(0)?.size()?;

        // Надо временно поделить на несколько частей, так как иначе получим множественное заимствование.
        let mut cam_1_matrix = camera_matrix[0].clone();
        let mut cam_1_dist = dist_coeffs[0].clone();
        let mut cam_2_matrix = camera_matrix[i].clone();
        let mut cam_2_dist = dist_coeffs[i].clone();

        let mut r = Mat::default();
        let mut t = Mat::default();
        let mut e = Mat::default();
        let mut f = Mat::default();

        let _ = stereo_calibrate(
            &common_object_points,
            &common_image_points1,
            &common_image_points2,
            &mut cam_1_matrix,
            &mut cam_1_dist,
            &mut cam_2_matrix,
            &mut cam_2_dist,
            img_size,
            &mut r,
            &mut t,
            &mut e,
            &mut f,
            opencv::calib3d::CALIB_FIX_INTRINSIC,
            criteria,
        )?;

        // Обновляем ориг. матрицы
        camera_matrix[0] = cam_1_matrix;
        dist_coeffs[0] = cam_1_dist;
        camera_matrix[i] = cam_2_matrix;
        dist_coeffs[i] = cam_2_dist;

        cameras.push(CameraParameters {
            intrinsic: camera_matrix[i].clone(),
            distortion: dist_coeffs[i].clone(),
            rotation: r,
            translation: t,
            essential_matrix: e,
            fundamental_matrix: f,
        });
    }
    Ok(cameras)
}

fn select_rows(src: &Mat, indices: &Vector<i32>) -> opencv::Result<Mat> {
    // имя/тип исходной матрицы
    let cols = src.cols();
    let typ = src.typ();

    // создаём пустой мат той же глубины/каналов
    let mut dst = Mat::zeros(indices.len() as i32, cols, typ)?.to_mat()?; // zeros вернёт MatExpr

    for (dst_r, src_r) in indices.iter().enumerate() {
        let src_row = src.row(src_r)?; // 1×C view
        let mut dst_row = dst.row_mut(dst_r as i32)?; // 1×C view (mutable)
        src_row.copy_to(&mut dst_row)?; // memcpy-эквивалент
    }
    Ok(dst)
}

#[derive(Debug)]
pub struct CameraParameters {
    pub intrinsic: Mat,
    pub distortion: Mat,
    pub rotation: Mat,
    pub translation: Mat,
    pub essential_matrix: Mat,
    pub fundamental_matrix: Mat,
}

impl CameraParameters {
    pub fn new() -> opencv::Result<Self> {
        Ok(Self {
            intrinsic: Mat::default(),
            distortion: Mat::default(),
            rotation: Mat::eye(3, 3, opencv::core::CV_64F)?.to_mat()?,
            translation: Mat::zeros(3, 1, opencv::core::CV_64F)?.to_mat()?,
            essential_matrix: Mat::default(),
            fundamental_matrix: Mat::default(),
        })
    }
}

#[derive(Debug)]
pub struct CalibrationFrame {
    pub object_points: Mat,       // CV_32FC3 (3D точки)
    pub image_points: Mat,        // CV_32FC2 (2D точки изображения)
    pub charuco_ids: Vector<i32>, // ID точек
}

// Функция для нахождения общих точек
pub fn find_common_points(frames: &[Vector<i32>]) -> HashSet<i32> {
    if frames.is_empty() {
        return HashSet::new();
    }

    // Первый набор - копируем значения
    let mut common_ids: HashSet<i32> = frames.get(0).unwrap().iter().collect();

    for frame in frames.iter().skip(1) {
        // Временный HashSet для сравнения
        let current_ids: HashSet<_> = frame.iter().collect();
        common_ids = common_ids.intersection(&current_ids).cloned().collect();
    }

    common_ids
}
