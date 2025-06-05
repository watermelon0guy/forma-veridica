use log::{debug, error, info, warn};
use opencv::{
    Error,
    calib3d::undistort_points,
    core::{DMatch, KeyPoint, Mat, Point3d, StsError, Vec2d, Vector, gemm},
    prelude::*,
    sfm::triangulate_points,
};
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

use crate::{
    calibration::CameraParameters,
    correspondence::{bf_match_knn, sift},
};

#[derive(Debug, Clone)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub color: Option<(u8, u8, u8)>, // RGB цвет точки
    pub track_id: Option<usize>,     // ID для отслеживания точки во времени
    pub confidence: f32,             // Уверенность в позиции точки
}

impl Point3D {
    pub fn new(x: f64, y: f64, z: f64, confidence: f32) -> Self {
        Self {
            x,
            y,
            z,
            color: None,
            track_id: None,
            confidence,
        }
    }

    pub fn from_opencv_point(point: Point3d, confidence: f32) -> Self {
        Self {
            x: point.x,
            y: point.y,
            z: point.z,
            color: None,
            track_id: None,
            confidence,
        }
    }

    pub fn to_opencv_point(&self) -> Point3d {
        Point3d::new(self.x, self.y, self.z)
    }
}

/// Структура для хранения облака точек
#[derive(Debug, Clone)]
pub struct PointCloud {
    pub points: Vec<Point3D>,
    pub timestamp: usize, // Временная метка кадра
}

pub fn triangulate_points_multiple(
    points_2d: &Vector<Mat>,
    camera_params: &[CameraParameters],
) -> Result<Vec<Point3D>, Error> {
    if points_2d.len() < 2 || camera_params.len() < 2 {
        error!("Недостаточно камер или наборов точек");
        return Err(Error::new(
            StsError as i32,
            "Требуется минимум 2 камеры для триангуляции".to_string(),
        ));
    }

    if points_2d.len() != camera_params.len() {
        error!("Количество наборов точек не соответствует количеству камер");
        return Err(Error::new(
            StsError as i32,
            "Количество списков точек должно совпадать с количеством камер".to_string(),
        ));
    }

    // Количество точек (предполагаем, что все матрицы имеют одинаковое количество строк)
    let num_points = points_2d.get(0)?.rows();
    debug!("Количество точек для триангуляции: {}", num_points);

    // Проверка, что все матрицы имеют правильный размер
    for (i, points) in points_2d.iter().enumerate() {
        if points.rows() != num_points || points.cols() != 2 {
            error!("Неверный размер матрицы точек для камеры {}", i);
            return Err(Error::new(
                StsError as i32,
                format!(
                    "Матрица точек камеры {} имеет неверный размер. Ожидается {}x2, получено {}x{}",
                    i,
                    num_points,
                    points.rows(),
                    points.cols()
                ),
            ));
        }
    }

    // Подготовка матриц проекций для всех камер
    let mut projection_matrices = Vector::<Mat>::default();

    for (i, cam) in camera_params.iter().enumerate() {
        // Проверка первой камеры
        if i == 0 {
            // Проверяем, является ли матрица вращения единичной
            let mut is_identity = true;
            for r in 0..3 {
                for c in 0..3 {
                    let expected = if r == c { 1.0 } else { 0.0 };
                    let actual = cam.rotation.at_2d::<f64>(r, c)?;
                    if (actual - expected).abs() > 1e-5 {
                        is_identity = false;
                        break;
                    }
                }
                if !is_identity {
                    break;
                }
            }

            // Проверяем, является ли вектор трансляции нулевым
            let mut is_zero_translation = true;
            for r in 0..3 {
                let val = cam.translation.at_2d::<f64>(r, 0)?;
                if val.abs() > 1e-5 {
                    is_zero_translation = false;
                    break;
                }
            }

            if !is_identity || !is_zero_translation {
                warn!(
                    "Вектор трансляции не нулевой или матрица вращения не единичная для главной камеры"
                );
            }
        }

        let mut r_t = Mat::default();
        opencv::core::hconcat2(&cam.rotation, &cam.translation, &mut r_t)?;

        let mut projection_matrix = Mat::default();
        opencv::sfm::projection_from_k_rt(
            &cam.intrinsic,
            &cam.rotation,
            &cam.translation,
            &mut projection_matrix,
        )
        .unwrap();
        projection_matrices.push(projection_matrix);
    }

    // Преобразование точек в формат для trianguluate_points (2xN матрицы)
    let converted_points: Vector<Mat> = points_2d
        .iter()
        .map(|points| {
            let mut transposed = Mat::default();
            opencv::core::transpose(&points, &mut transposed)?;
            Ok(transposed)
        })
        .collect::<Result<Vector<Mat>, Error>>()?;

    let mut points_3d = Mat::default();

    match triangulate_points(&converted_points, &projection_matrices, &mut points_3d) {
        Ok(_) => {
            debug!(
                "Триангуляция успешно выполнена. Количество точек: {}",
                points_3d.cols()
            );
        }
        Err(e) => {
            error!("Ошибка при триангуляции: {:?}", e);
            return Err(e);
        }
    }

    let mut result = Vec::with_capacity(num_points as usize);

    let mut total_errors = Vec::new();
    let mut num_bad_points = 0;

    for i in 0..num_points {
        let x = *points_3d.at_2d::<f64>(0, i)?;
        let y = *points_3d.at_2d::<f64>(1, i)?;
        let z = *points_3d.at_2d::<f64>(2, i)?;

        // Вычисление перепроекционной ошибки для оценки качества триангуляции
        let mut total_reproj_error = 0.0;
        let mut errors_by_camera = Vec::new();

        for (j, projection) in projection_matrices.iter().enumerate() {
            // Создаем 4D точку (X, Y, Z, 1)
            let mut point_4d = Mat::zeros(4, 1, opencv::core::CV_64F)?.to_mat()?;
            *point_4d.at_2d_mut::<f64>(0, 0)? = x;
            *point_4d.at_2d_mut::<f64>(1, 0)? = y;
            *point_4d.at_2d_mut::<f64>(2, 0)? = z;
            *point_4d.at_2d_mut::<f64>(3, 0)? = 1.0;

            // Проекция на изображение: x' = P * X
            let mut projected = Mat::default();
            gemm(
                &projection,
                &point_4d,
                1.0,
                &Mat::default(),
                0.0,
                &mut projected,
                0,
            )?;

            // Нормализуем проекцию
            let p_x = *projected.at_2d::<f64>(0, 0)? / *projected.at_2d::<f64>(2, 0)?;
            let p_y = *projected.at_2d::<f64>(1, 0)? / *projected.at_2d::<f64>(2, 0)?;

            // Исходная точка на изображении
            let orig_x = *points_2d.get(j)?.at_2d::<f64>(i, 0)?;
            let orig_y = *points_2d.get(j)?.at_2d::<f64>(i, 1)?;

            // Вычисляем ошибку (евклидово расстояние)
            let error = ((p_x - orig_x).powi(2) + (p_y - orig_y).powi(2)).sqrt();
            errors_by_camera.push(error);
            total_reproj_error += error;
        }

        // Средняя ошибка репроекции для этой точки
        let avg_error = total_reproj_error / camera_params.len() as f64;
        total_errors.push(avg_error);

        // Преобразуем в нормализованную уверенность (1.0 - хорошо, 0.0 - плохо)
        // Порог ошибки - настраиваемый параметр (например, 5 пикселей)
        let confidence = (1.0 - (avg_error / 5.0).min(1.0)) as f32;

        // Считаем плохие точки (с большой ошибкой)
        if avg_error > 5.0 {
            num_bad_points += 1;
        }

        result.push(Point3D::new(x, y, z, confidence));
    }

    // Вывод статистики по ошибкам
    if !total_errors.is_empty() {
        total_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let min_error = total_errors[0];
        let max_error = total_errors[total_errors.len() - 1];
        let median_error = total_errors[total_errors.len() / 2];
        let mean_error = total_errors.iter().sum::<f64>() / total_errors.len() as f64;

        info!("Минимальная ошибка: {:.2} пикс.", min_error);
        info!("Медианная ошибка:  {:.2} пикс.", median_error);
        info!("Средняя ошибка:    {:.2} пикс.", mean_error);
        info!("Максимальная ошибка: {:.2} пикс.", max_error);
        info!(
            "Количество точек с ошибкой > 5 пикс.: {} из {} ({:.1}%)",
            num_bad_points,
            num_points,
            100.0 * num_bad_points as f64 / num_points as f64
        );
    }
    Ok(result)
}

pub fn save_point_cloud<P: AsRef<Path>>(cloud: &PointCloud, path: P) -> io::Result<()> {
    let mut file = File::create(path)?;

    // Определяем, сколько точек имеют цвет (для заголовка PLY)
    let points_with_color = cloud.points.iter().filter(|p| p.color.is_some()).count();
    let has_color = points_with_color > 0;

    // Записываем заголовок PLY
    writeln!(file, "ply")?;
    writeln!(file, "format ascii 1.0")?;
    writeln!(file, "element vertex {}", cloud.points.len())?;
    writeln!(file, "property float x")?;
    writeln!(file, "property float y")?;
    writeln!(file, "property float z")?;

    // Добавляем свойства цвета, если они есть
    if has_color {
        writeln!(file, "property uchar red")?;
        writeln!(file, "property uchar green")?;
        writeln!(file, "property uchar blue")?;
    }

    // Добавляем свойство уверенности
    writeln!(file, "property float confidence")?;

    // Конец заголовка
    writeln!(file, "end_header")?;

    // Записываем данные
    for point in &cloud.points {
        if has_color {
            // С цветом
            let (r, g, b) = point.color.unwrap_or((128, 128, 128));
            writeln!(
                file,
                "{} {} {} {} {} {} {}",
                point.x, point.y, point.z, r, g, b, point.confidence
            )?;
        } else {
            // Без цвета
            writeln!(
                file,
                "{} {} {} {}",
                point.x, point.y, point.z, point.confidence
            )?;
        }
    }

    Ok(())
}

pub fn match_first_camera_features_to_all(
    images: &Vec<Mat>,
) -> (Vec<Vector<Vector<DMatch>>>, Vec<Vector<KeyPoint>>, Vec<Mat>) {
    let mut keypoints_list = Vec::new();
    let mut descriptors_list = Vec::new();

    for (i, image) in images.iter().enumerate() {
        info!("Обработка изображения {} из {}", i + 1, images.len());
        let (keypoints, descriptors) = match sift(&image, 0, 4, 0.04, 10f64, 1.6, false) {
            Ok(it) => {
                info!("  -> Найдено {} ключевых точек", it.0.len());
                it
            }
            Err(e) => {
                error!("  -> Ошибка при выполнении SIFT: {:?}", e);
                continue;
            }
        };
        keypoints_list.push(keypoints);
        descriptors_list.push(descriptors);
    }

    let mut all_matches = Vec::new();
    // Первая камера - референсная
    let ref_descriptor = &descriptors_list[0];

    for i in 1..descriptors_list.len() {
        info!("Сопоставление камеры 1 с камерой {}", i + 1);
        let matches = match bf_match_knn(
            &ref_descriptor,
            &descriptors_list[i],
            2,   // k = 2 соседа
            0.7, // ratio = 0.7
        ) {
            Ok(it) => {
                info!("Найдено {} сопоставлений", it.len());
                it
            }
            Err(e) => {
                error!("Ошибка при выполнении сопоставления BF KNN: {:?}", e);
                continue;
            }
        };
        all_matches.push(matches);
    }
    (all_matches, keypoints_list, descriptors_list)
    // TODO добавить вывод ошибки при отсутсвии сопоставлений
}

pub fn min_visible_match_set(
    all_matches: &Vec<Vector<Vector<DMatch>>>,
    keypoints_list: &Vec<Vector<KeyPoint>>,
) -> Vec<Vector<Vector<DMatch>>> {
    // Создаем множество индексов ключевых точек из референсной камеры,
    // которые имеют соответствие во всех других камерах
    let mut common_points_indices = Vec::new();

    // Для каждой ключевой точки из референсной камеры
    for i in 0..keypoints_list[0].len() {
        // Проверяем, есть ли соответствие этой точки во всех других камерах
        let mut visible_in_all_cameras = true;

        for camera_matches in all_matches {
            // Проверяем, существует ли соответствие для текущей точки в данной камере
            let point_has_match = camera_matches
                .iter()
                .any(|m| m.get(0).unwrap().query_idx as usize == i);

            if !point_has_match {
                visible_in_all_cameras = false;
                break;
            }
        }

        if visible_in_all_cameras {
            common_points_indices.push(i);
        }
    }

    info!(
        "Найдено {} точек, видимых во всех камерах",
        common_points_indices.len()
    );

    // Фильтруем matches, оставляя только точки, видимые во всех камерах
    let mut filtered_matches = Vec::new();
    for camera_matches in all_matches {
        let mut filtered_camera_matches = Vector::<Vector<DMatch>>::new();

        for idx in &common_points_indices {
            // Находим соответствие для этой точки в текущей камере
            for m in camera_matches {
                if m.get(0).unwrap().query_idx as usize == *idx {
                    filtered_camera_matches.push(m.clone());
                    break;
                }
            }
        }

        filtered_matches.push(filtered_camera_matches);
    }

    filtered_matches
}

pub fn filter_point_cloud_by_confindence(cloud: &mut PointCloud, confidence_threshold: f32) {
    cloud
        .points
        .retain(|point| point.confidence >= confidence_threshold);
}

pub fn add_color_to_point_cloud(
    cloud: &mut PointCloud,
    distorted_points: &Vector<Mat>,
    ref_image: &Mat,
) {
    // Добавляем цвет из исходного изображения
    for (i, point) in cloud.points.iter_mut().enumerate() {
        let x = *distorted_points
            .get(0)
            .unwrap()
            .at_2d::<f64>(i as i32, 0)
            .unwrap() as i32;
        let y = *distorted_points
            .get(0)
            .unwrap()
            .at_2d::<f64>(i as i32, 1)
            .unwrap() as i32;

        // Проверяем, что координаты в пределах изображения
        if x >= 0 && y >= 0 && x < ref_image.cols() && y < ref_image.rows() {
            let color = ref_image.at_2d::<opencv::core::Vec3b>(y, x).unwrap();
            point.color = Some((color[2], color[1], color[0])); // BGR -> RGB
        }
    }
}

pub fn undistort_points_single_camera(
    points: &Mat, // Nx2, CV_64F
    camera: &CameraParameters,
) -> Result<Mat, Error> {
    let num_points = points.rows();
    let mut undistorted_points = Mat::zeros(num_points, 1, opencv::core::CV_64FC2)?.to_mat()?;

    undistort_points(
        points,
        &mut undistorted_points,
        &camera.intrinsic,
        &camera.distortion,
        &Mat::default(),
        &camera.intrinsic,
    )?;

    let mut undistorted_nx2 = Mat::zeros(num_points, 2, opencv::core::CV_64F)?.to_mat()?;
    for j in 0..num_points {
        let pt = undistorted_points.at_2d::<Vec2d>(j, 0)?;
        *undistorted_nx2.at_2d_mut::<f64>(j, 0)? = pt[0];
        *undistorted_nx2.at_2d_mut::<f64>(j, 1)? = pt[1];
    }
    Ok(undistorted_nx2)
}
