use opencv::{
    Error,
    core::{Mat, Point3d, StsError, Vector, gemm},
    prelude::*,
    sfm::triangulate_points,
};
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

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
    println!("\n=== НАЧАЛО ФУНКЦИИ triangulate_points_multiple ===");
    println!("Количество наборов точек: {}", points_2d.len());
    println!("Количество камер: {}", camera_params.len());

    if points_2d.len() < 2 || camera_params.len() < 2 {
        println!("ОШИБКА: Недостаточно камер или наборов точек");
        return Err(Error::new(
            StsError as i32,
            "Требуется минимум 2 камеры для триангуляции".to_string(),
        ));
    }

    if points_2d.len() != camera_params.len() {
        println!("ОШИБКА: Количество наборов точек не соответствует количеству камер");
        return Err(Error::new(
            StsError as i32,
            "Количество списков точек должно совпадать с количеством камер".to_string(),
        ));
    }

    // Количество точек (предполагаем, что все матрицы имеют одинаковое количество строк)
    let num_points = points_2d.get(0)?.rows();
    println!("Количество точек для триангуляции: {}", num_points);

    // Проверка, что все матрицы имеют правильный размер
    for (i, points) in points_2d.iter().enumerate() {
        if points.rows() != num_points || points.cols() != 2 {
            println!("ОШИБКА: Неверный размер матрицы точек для камеры {}", i);
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
        println!(
            "Камера {}: {} точек, размер {}x{}",
            i,
            points.rows(),
            points.rows(),
            points.cols()
        );
    }

    // Подготовка матриц проекций для всех камер
    let mut projection_matrices = Vector::<Mat>::default();
    println!("\n=== АНАЛИЗ ПАРАМЕТРОВ КАМЕР И ПОСТРОЕНИЕ МАТРИЦ ПРОЕКЦИИ ===");

    for (i, cam) in camera_params.iter().enumerate() {
        println!("\nПараметры камеры #{}:", i);
        println!("Внутренняя матрица (intrinsic):\n{:?}", cam.intrinsic);
        println!("Дисторсия:\n{:?}", cam.distortion);
        println!("Матрица вращения (rotation):\n{:?}", cam.rotation);
        println!("Вектор трансляции (translation):\n{:?}", cam.translation);

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

            println!("Проверка первой камеры:");
            println!("  Матрица вращения является единичной: {}", is_identity);
            println!(
                "  Вектор трансляции является нулевым: {}",
                is_zero_translation
            );

            if !is_identity || !is_zero_translation {
                println!(
                    "  ВНИМАНИЕ: Для первой камеры ожидается единичная матрица вращения и нулевой вектор трансляции!"
                );
            }
        }

        let mut r_t = Mat::default();
        opencv::core::hconcat2(&cam.rotation, &cam.translation, &mut r_t)?;
        println!("Объединенная матрица [R|t] для камеры #{}:\n{:?}", i, r_t);

        let mut projection_matrix = Mat::default();
        opencv::sfm::projection_from_k_rt(
            &cam.intrinsic,
            &cam.rotation,
            &cam.translation,
            &mut projection_matrix,
        )
        .unwrap();
        // gemm(
        //     &cam.intrinsic,
        //     &r_t,
        //     1.0,
        //     &Mat::default(),
        //     0.0,
        //     &mut projection_matrix,
        //     0,
        // )?;
        println!(
            "Матрица проекции для камеры #{}:\n{:?}",
            i, projection_matrix
        );

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
    println!("\n=== ТРИАНГУЛЯЦИЯ ТОЧЕК ===");
    let mut points_3d = Mat::default();

    println!(
        "Размер вектора с converted_points: {}",
        converted_points.len()
    );
    for (i, pts) in converted_points.iter().enumerate() {
        println!(
            "converted_points[{}] размер: {}x{}, тип: {:?}",
            i,
            pts.rows(),
            pts.cols(),
            pts.typ()
        );
    }

    println!(
        "Размер вектора с projection_matrices: {}",
        projection_matrices.len()
    );
    for (i, proj) in projection_matrices.iter().enumerate() {
        println!(
            "projection_matrices[{}] размер: {}x{}, тип: {:?}",
            i,
            proj.rows(),
            proj.cols(),
            proj.typ()
        );
    }

    println!("Вызов функции triangulate_points...");
    match triangulate_points(&converted_points, &projection_matrices, &mut points_3d) {
        Ok(_) => {
            println!(
                "Триангуляция успешно выполнена. Получена матрица размером {}x{}, тип: {:?}",
                points_3d.rows(),
                points_3d.cols(),
                points_3d.typ()
            );
        }
        Err(e) => {
            println!("ОШИБКА при триангуляции: {:?}", e);
            return Err(e);
        }
    }

    // Преобразование результата в вектор Point3D
    println!("\n=== АНАЛИЗ РЕЗУЛЬТАТОВ ТРИАНГУЛЯЦИИ ===");
    let mut result = Vec::with_capacity(num_points as usize);

    // Выведем несколько примеров точек
    let sample_size = num_points.min(5) as usize;
    println!("Пример первых {} триангулированных точек:", sample_size);

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

        // Выводим примеры первых нескольких точек
        if i < sample_size as i32 {
            println!(
                "Точка #{}: ({:.2}, {:.2}, {:.2}), средняя ошибка: {:.2} пикс., уверенность: {:.2}",
                i, x, y, z, avg_error, confidence
            );
            println!("  Ошибки по камерам: {:?}", errors_by_camera);
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

        println!("\nСтатистика ошибок репроекции:");
        println!("  Минимальная ошибка: {:.2} пикс.", min_error);
        println!("  Медианная ошибка:  {:.2} пикс.", median_error);
        println!("  Средняя ошибка:    {:.2} пикс.", mean_error);
        println!("  Максимальная ошибка: {:.2} пикс.", max_error);
        println!(
            "  Количество точек с ошибкой > 5 пикс.: {} из {} ({:.1}%)",
            num_bad_points,
            num_points,
            100.0 * num_bad_points as f64 / num_points as f64
        );
    }

    println!("\n=== КОНЕЦ ФУНКЦИИ triangulate_points_multiple ===");
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
