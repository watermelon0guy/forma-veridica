use std::collections::HashSet;
use std::fs;
use std::path::Path;

use log::{debug, error, info};
use opencv::calib3d::{calibrate_camera, stereo_calibrate};
use opencv::core::{
    FileStorage, FileStorage_Mode, NORM_L2, Point2f, TermCriteria, TermCriteria_Type, Vector, norm,
};
use opencv::imgcodecs::{IMREAD_COLOR, imread};
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
    debug!("Начало калибровки камер");
    debug!("Параметры доски ChArUco: {:?}", charuco_board);
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
        error!("Ошибка: для калибровки требуется как минимум 2 набора изображений");
        return Ok(vec![]);
    }

    debug!(
        "Количество наборов изображений для калибровки: {}",
        imgs.len()
    );

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
                debug!("Ошибка обычной калибровки {}", curr_cam_ret_val);
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
            Err(e) => error!("Ошибка калибровки calibrate_with_charuco: {:?}", e),
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
            debug!(
                "Камера 0 и камера {}: найдено {} общих точек",
                i,
                common.len()
            );
            if common.len() < 10 {
                debug!(
                    "ВНИМАНИЕ: недостаточно общих точек между камерой 0 и камерой {}",
                    i
                );
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

            let obj_points = select_rows(&object_points[0].get(frame_idx)?, &idx_cam1)?;
            let img_points1 = select_rows(&image_points[0].get(frame_idx)?, &idx_cam1)?;
            let img_points2 = select_rows(&image_points[i].get(frame_idx)?, &idx_cam2)?;

            debug!(
                "Кадр {}, Камера 0 и {}: выбрано {} 3D точек, {} точек на изображении 1, {} точек на изображении 2",
                frame_idx,
                i,
                obj_points.rows(),
                img_points1.rows(),
                img_points2.rows()
            );

            common_object_points.push(obj_points);
            common_image_points1.push(img_points1);
            common_image_points2.push(img_points2);
        }

        let img_size = imgs[0].get(0)?.size()?;

        debug!("Подготовка 1 камеры к стереокалибровке");
        debug!(
            "Количество кадров с общими точками: {}",
            common_object_points.len()
        );

        // Надо временно поделить на несколько частей, так как иначе получим множественное заимствование.
        let mut cam_1_matrix = camera_matrix[0].clone();
        let mut cam_1_dist = dist_coeffs[0].clone();
        let mut cam_2_matrix = camera_matrix[i].clone();
        let mut cam_2_dist = dist_coeffs[i].clone();

        debug!("Матрица камеры 0 до стерео калибровки:\n{:?}", cam_1_matrix);
        debug!("Дисторсия камеры 0 до стерео калибровки:\n{:?}", cam_1_dist);
        debug!(
            "Матрица камеры {} до стерео калибровки:\n{:?}",
            i, cam_2_matrix
        );
        debug!(
            "Дисторсия камеры {} до стерео калибровки:\n{:?}",
            i, cam_2_dist
        );

        let mut r = Mat::default();
        let mut t = Mat::default();
        let mut e = Mat::default();
        let mut f = Mat::default();

        debug!("Выполнение stereo_calibrate...");
        let stereo_error = stereo_calibrate(
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

        debug!(
            "Ошибка стерео калибровки для камеры {}: {}",
            i, stereo_error
        );
        debug!(
            "Матрица камеры 0 после стерео калибровки:\n{:?}",
            cam_1_matrix
        );
        debug!(
            "Дисторсия камеры 0 после стерео калибровки:\n{:?}",
            cam_1_dist
        );
        debug!(
            "Матрица камеры {} после стерео калибровки:\n{:?}",
            i, cam_2_matrix
        );
        debug!(
            "Дисторсия камеры {} после стерео калибровки:\n{:?}",
            i, cam_2_dist
        );
        debug!("Матрица вращения:\n{:#?}", r);
        debug!("Вектор трансляции:\n{:#?}", t);

        // Вычисляем норму вектора трансляции для получения расстояния
        let t_norm = norm(&t, opencv::core::NORM_L2, &Mat::default())?;
        debug!("Расстояние между камерой 0 и камерой {}: {} мм", i, t_norm);

        // Удаляем обновление матриц камеры
        // camera_matrix[0] = cam_1_matrix;
        // dist_coeffs[0] = cam_1_dist;
        // camera_matrix[i] = cam_2_matrix;
        // dist_coeffs[i] = cam_2_dist;

        cameras.push(CameraParameters {
            intrinsic: camera_matrix[i].clone(),
            distortion: dist_coeffs[i].clone(),
            rotation: r,
            translation: t,
            essential_matrix: e,
            fundamental_matrix: f,
        });

        debug!("=== Калибровка камеры {} завершена ===", i);
    }
    debug!("=== Калибровка множества камер завершена ===");

    // Анализируем расстояния между камерами
    let _ = calculate_adjacent_camera_distances(&cameras);
    debug!("Проверка {:#?}", cameras[1]);
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

/// Вычисляет расстояния между соседними камерами и возвращает их в виде вектора
pub fn calculate_adjacent_camera_distances(
    cameras: &[CameraParameters],
) -> Result<Vec<f64>, opencv::Error> {
    debug!("\n=== Анализ расстояний между соседними камерами ===");

    if cameras.len() < 2 {
        debug!("Недостаточно камер для анализа расстояний");
        return Ok(Vec::new());
    }

    let mut distances = Vec::with_capacity(cameras.len() - 1);

    for i in 1..cameras.len() {
        let t = &cameras[i].translation;
        let t_norm = norm(t, opencv::core::NORM_L2, &Mat::default())?;

        // Получаем компоненты вектора трансляции
        let tx = t.at_2d::<f64>(0, 0)?;
        let ty = t.at_2d::<f64>(1, 0)?;
        let tz = t.at_2d::<f64>(2, 0)?;

        debug!("Камера {} → Камера 0:", i);
        debug!("  Полное расстояние: {:.2} мм", t_norm);
        debug!(
            "  Компоненты вектора: X={:.2} мм, Y={:.2} мм, Z={:.2} мм",
            tx, ty, tz
        );

        // Если это не первая камера (т.е. i > 1), также вычисляем относительное расстояние
        // от предыдущей камеры
        if i > 1 {
            let prev_t = &cameras[i - 1].translation;
            let prev_tx = prev_t.at_2d::<f64>(0, 0)?;
            let prev_ty = prev_t.at_2d::<f64>(1, 0)?;
            let prev_tz = prev_t.at_2d::<f64>(2, 0)?;

            let rel_tx = tx - prev_tx;
            let rel_ty = ty - prev_ty;
            let rel_tz = tz - prev_tz;
            let rel_t_norm = (rel_tx * rel_tx + rel_ty * rel_ty + rel_tz * rel_tz).sqrt();

            debug!("  Относительно камеры {}:", i - 1);
            debug!("    Относительное расстояние: {:.2} мм", rel_t_norm);
            debug!(
                "    Относительные компоненты: X={:.2} мм, Y={:.2} мм, Z={:.2} мм",
                rel_tx, rel_ty, rel_tz
            );
        }

        distances.push(t_norm);
    }

    debug!("=== Конец анализа расстояний ===\n");
    Ok(distances)
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

pub fn perform_calibration(
    image_path: &str,
    cameras_params_path: &Path,
    charuco_board: &CharucoBoard,
    num_cameras: usize,
) {
    debug!("Поиск калибровочных изображений в: {}", image_path);

    // Собираем все файлы в директории
    let dir_entries = match fs::read_dir(image_path) {
        Ok(entries) => entries,
        Err(e) => {
            error!("Ошибка чтения директории: {}", e);
            return;
        }
    };

    // Группируем изображения по камерам и кадрам
    let mut frame_numbers = Vec::new();
    let mut camera_images: Vec<Vector<Mat>> = vec![Vector::<Mat>::new(); num_cameras];

    for entry in dir_entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        debug!("Загружаю {}", file_name);

        if file_name.starts_with("img_") && file_name.ends_with(".png") {
            let parts: Vec<&str> = file_name.split('_').collect();
            if parts.len() == 3 {
                if let Ok(cam_num) = parts[1].parse::<usize>() {
                    if let Ok(frame_num) = parts[2].trim_end_matches(".png").parse::<usize>() {
                        if let Ok(img) = imread(&entry.path().to_string_lossy(), IMREAD_COLOR) {
                            camera_images[cam_num - 1].push(img);
                            frame_numbers.push(frame_num);
                        }
                    }
                }
            }
        }
    }

    // Удаляем дубликаты frame_numbers и сортируем
    frame_numbers.sort();
    frame_numbers.dedup();

    info!("Найдено {} наборов(сцен) изображений", frame_numbers.len());

    // Выполняем калибровку
    match calibrate_multiple_with_charuco(&camera_images, charuco_board) {
        Ok(cameras) => {
            info!(
                "Калибровка успешно завершена. Получено {} камер:",
                cameras.len()
            );
            for (i, cam) in cameras.iter().enumerate() {
                if i > 0 {
                    debug!(
                        "Дистанция от основной камеры: {:.2} мм",
                        norm(&cam.translation, NORM_L2, &Mat::default()).unwrap()
                    );
                }
            }

            // Сохранение параметров в файл (опционально)
            if let Err(e) = save_camera_parameters(
                &cameras,
                &format!(
                    "{}/calibration_params.yml",
                    cameras_params_path.to_str().unwrap()
                ),
            ) {
                error!("Ошибка при сохранении параметров: {}", e);
            }
        }
        Err(e) => error!("Ошибка при калибровке: {:?}", e),
    }
}

fn save_camera_parameters(cameras: &[CameraParameters], path: &str) -> opencv::Result<()> {
    let mut fs = FileStorage::new(path, FileStorage_Mode::WRITE as i32, "")?;

    for (i, cam) in cameras.iter().enumerate() {
        // Для матриц используем специальные методы записи
        fs.write_mat(&format!("camera_{}_intrinsic", i), &cam.intrinsic)?;
        fs.write_mat(&format!("camera_{}_distortion", i), &cam.distortion)?;

        if i > 0 {
            fs.write_mat(&format!("camera_{}_rotation", i), &cam.rotation)?;
            fs.write_mat(&format!("camera_{}_translation", i), &cam.translation)?;
        }
    }

    fs.release()?;
    Ok(())
}

pub fn load_camera_parameters(path: &str) -> opencv::Result<Vec<CameraParameters>> {
    let mut fs = FileStorage::new(path, FileStorage_Mode::READ as i32, "")?;

    let mut cameras = Vec::new();
    let mut i = 0;

    loop {
        let intrinsic_name = format!("camera_{}_intrinsic", i);
        debug!("Попытка считать данные для камеры {}", i);
        if fs.get_node(&intrinsic_name)?.empty()? {
            break;
        }

        let mut cam_params = CameraParameters::new()?;

        cam_params.intrinsic = fs.get_node(&intrinsic_name)?.mat()?;
        cam_params.distortion = fs.get_node(&format!("camera_{}_distortion", i))?.mat()?;

        if i > 0 {
            cam_params.rotation = fs.get_node(&format!("camera_{}_rotation", i))?.mat()?;
            cam_params.translation = fs.get_node(&format!("camera_{}_translation", i))?.mat()?;
        }

        cameras.push(cam_params);
        i += 1;
    }

    fs.release()?;

    if cameras.is_empty() {
        return Err(opencv::Error::new(
            opencv::core::StsError as i32,
            "Не удалось загрузить параметры ни одной камеры".to_string(),
        ));
    }

    Ok(cameras)
}
