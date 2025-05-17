use opencv::{
    Error,
    core::{Mat, Point3d, StsError, Vector, multiply},
    prelude::*,
    sfm::triangulate_points,
};

use crate::calibration::CameraParameters;

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
        return Err(Error::new(
            StsError as i32,
            "Требуется минимум 2 камеры для триангуляции".to_string(),
        ));
    }

    if points_2d.len() != camera_params.len() {
        return Err(Error::new(
            StsError as i32,
            "Количество списков точек должно совпадать с количеством камер".to_string(),
        ));
    }

    // Количество точек (предполагаем, что все матрицы имеют одинаковое количество строк)
    let num_points = points_2d.get(0)?.rows();

    // Проверка, что все матрицы имеют правильный размер
    for (i, points) in points_2d.iter().enumerate() {
        if points.rows() != num_points || points.cols() != 2 {
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
    for cam in camera_params {
        let mut r_t = Mat::default();
        opencv::core::hconcat2(&cam.rotation, &cam.translation, &mut r_t)?;

        let mut projection_matrix = Mat::default();
        multiply(&cam.intrinsic, &r_t, &mut projection_matrix, 1.0, -1)?;
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

    // Триангуляция точек
    let mut points_3d = Mat::default();
    triangulate_points(&converted_points, &projection_matrices, &mut points_3d)?;

    // Преобразование результата в вектор Point3D
    let mut result = Vec::with_capacity(num_points as usize);
    for i in 0..num_points {
        let x = *points_3d.at_2d::<f64>(0, i)?;
        let y = *points_3d.at_2d::<f64>(1, i)?;
        let z = *points_3d.at_2d::<f64>(2, i)?;

        // Вычисление перепроекционной ошибки для оценки качества триангуляции
        let mut total_reproj_error = 0.0;

        for (j, projection) in projection_matrices.iter().enumerate() {
            // Создаем 4D точку (X, Y, Z, 1)
            let mut point_4d = Mat::zeros(4, 1, opencv::core::CV_64F)?.to_mat()?;
            *point_4d.at_2d_mut::<f64>(0, 0)? = x;
            *point_4d.at_2d_mut::<f64>(1, 0)? = y;
            *point_4d.at_2d_mut::<f64>(2, 0)? = z;
            *point_4d.at_2d_mut::<f64>(3, 0)? = 1.0;

            // Проекция на изображение: x' = P * X
            let mut projected = Mat::default();
            multiply(&projection, &point_4d, &mut projected, 1.0, -1)?;

            // Нормализуем проекцию
            let p_x = *projected.at_2d::<f64>(0, 0)? / *projected.at_2d::<f64>(2, 0)?;
            let p_y = *projected.at_2d::<f64>(1, 0)? / *projected.at_2d::<f64>(2, 0)?;

            // Исходная точка на изображении
            let orig_x = *points_2d.get(j)?.at_2d::<f64>(i, 0)?;
            let orig_y = *points_2d.get(j)?.at_2d::<f64>(i, 1)?;

            // Вычисляем ошибку (евклидово расстояние)
            let error = ((p_x - orig_x).powi(2) + (p_y - orig_y).powi(2)).sqrt();
            total_reproj_error += error;
        }

        // Средняя ошибка репроекции для этой точки
        let avg_error = total_reproj_error / camera_params.len() as f64;

        // Преобразуем в нормализованную уверенность (1.0 - хорошо, 0.0 - плохо)
        // Порог ошибки - настраиваемый параметр (например, 5 пикселей)
        let confidence = (1.0 - (avg_error / 5.0).min(1.0)) as f32;

        result.push(Point3D::new(x, y, z, confidence));
    }

    Ok(result)
}
