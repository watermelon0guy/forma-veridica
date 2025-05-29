use lib_cv::calibration::load_camera_parameters;
use lib_cv::reconstruction::{PointCloud, save_point_cloud};
use lib_cv::utils::split_image_into_quadrants;
use opencv::core::Vector;
use opencv::videoio::VideoCapture;
use opencv::{highgui, prelude::*};
use std::time::Instant;

fn main() {
    let total_start_time = Instant::now();
    println!("==== ЭТАП 0: Инициализация программы ====");
    const IMAGE_PATH: &str =
        "/home/watermelon0guy/Изображения/Experiments/raspberry_pi_cardboard/calibration";
    println!("Путь к изображениям: {}", IMAGE_PATH);

    println!("==== ЭТАП 1: Загрузка параметров камеры ====");
    let start_time = Instant::now();
    let camera_params = load_camera_parameters("/home/watermelon0guy/Изображения/Experiments/raspberry_pi_cardboard/calibration/picked/calibration_params.yml")
        .expect("Не получилось загрузить параметры камеры");
    println!(
        "Загружено {} наборов параметров камер за {:?}",
        camera_params.len(),
        start_time.elapsed()
    );

    println!("==== ЭТАП 2: Открытие видеофайла ====");
    let start_time = Instant::now();
    let mut cap = VideoCapture::from_file(
        "/home/watermelon0guy/Видео/Experiments/raspberry_pi_cardboard/20250529_102950_hires.mp4",
        opencv::videoio::CAP_ANY,
    )
    .unwrap();
    println!("Видеофайл успешно открыт за {:?}", start_time.elapsed());

    println!("==== ЭТАП 3: Чтение кадра ====");
    let start_time = Instant::now();
    let mut frame = opencv::core::Mat::default();
    cap.read(&mut frame).unwrap();
    println!(
        "Кадр считан. Размер: {}x{} за {:?}",
        frame.cols(),
        frame.rows(),
        start_time.elapsed()
    );

    let mut current_i = 0;

    println!("==== ЭТАП 4: Разделение изображения на камеры ====");
    let start_time = Instant::now();
    let images = match split_image_into_quadrants(&frame) {
        Ok(it) => {
            println!(
                "Изображение успешно разделено на {} части за {:?}",
                it.len(),
                start_time.elapsed()
            );
            it
        }
        Err(e) => {
            eprintln!("Ошибка при разделении изображения: {:?}", e);
            return;
        }
    };

    println!("==== ЭТАП 5: Извлечение ключевых точек (SIFT) ====");
    let start_time = Instant::now();
    let mut keypoints_list = Vec::new();
    let mut descriptors_list = Vec::new();

    for (i, image) in images.iter().enumerate() {
        println!("Обработка изображения {} из {}", i + 1, images.len());
        let (keypoints, descriptors) = match lib_cv::correspondence::sift(&image) {
            Ok(it) => {
                println!("  -> Найдено {} ключевых точек", it.0.len());
                it
            }
            Err(e) => {
                eprintln!("  -> Ошибка при выполнении SIFT: {:?}", e);
                continue;
            }
        };
        keypoints_list.push(keypoints);
        descriptors_list.push(descriptors);
    }
    println!("Обработка SIFT завершена за {:?}", start_time.elapsed());

    println!("==== ЭТАП 6: Сопоставление ключевых точек ====");
    let start_time = Instant::now();

    let mut all_matches = Vec::new();
    // Первая камера - референсная
    let ref_descriptor = &descriptors_list[0];

    println!("Сопоставление ключевых точек между камерами:");
    for i in 1..descriptors_list.len() {
        println!("  -> Сопоставление камеры 1 с камерой {}", i + 1);
        let matches = match lib_cv::correspondence::bf_match_knn(
            &ref_descriptor,
            &descriptors_list[i],
            2,   // k = 2 соседа
            0.7, // ratio = 0.7
        ) {
            Ok(it) => {
                println!("    -> Найдено {} сопоставлений", it.len());
                it
            }
            Err(e) => {
                eprintln!("    -> Ошибка при выполнении сопоставления BF KNN: {:?}", e);
                continue;
            }
        };
        all_matches.push(matches);
    }
    println!("Сопоставление завершено за {:?}", start_time.elapsed());

    println!("==== ЭТАП 6.5: Минимально видимый набор ====");
    let start_time = Instant::now();

    // Создаем множество индексов ключевых точек из референсной камеры,
    // которые имеют соответствие во всех других камерах
    let mut common_points_indices = Vec::new();

    // Для каждой ключевой точки из референсной камеры
    for i in 0..keypoints_list[0].len() {
        // Проверяем, есть ли соответствие этой точки во всех других камерах
        let mut visible_in_all_cameras = true;

        for camera_matches in &all_matches {
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

    println!(
        "Найдено {} точек, видимых во всех камерах",
        common_points_indices.len()
    );

    // Фильтруем matches, оставляя только точки, видимые во всех камерах
    let mut filtered_matches = Vec::new();
    for camera_matches in &all_matches {
        let mut filtered_camera_matches = Vector::new();

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

    // Заменяем старые matches на отфильтрованные
    all_matches = filtered_matches;

    println!("Фильтрация завершена за {:?}", start_time.elapsed());

    let mut img_with_points = images[0].clone();
    for &idx in &common_points_indices {
        let kp = &keypoints_list[0].get(idx).unwrap();
        let center = opencv::core::Point::new(kp.pt().x as i32, kp.pt().y as i32);
        opencv::imgproc::circle(
            &mut img_with_points,
            center,
            5,
            opencv::core::Scalar::new(0.0, 0.0, 255.0, 0.0),
            2,
            opencv::imgproc::LINE_8,
            0,
        )
        .unwrap();
    }
    highgui::named_window("Common Points - Camera 1", highgui::WINDOW_AUTOSIZE).unwrap();
    highgui::imshow("Common Points - Camera 1", &img_with_points).unwrap();
    highgui::wait_key(0).unwrap();

    println!("==== ЭТАП 7: Подготовка матриц точек для триангуляции ====");
    let start_time = Instant::now();

    // Создаем матрицы с 2D точками для всех камер
    let mut points_2d = Vector::<Mat>::default();

    // Для первой (референсной) камеры
    let num_matches = all_matches[0].len();
    println!("Общее количество сопоставленных точек: {}", num_matches);
    let mut points_cam1 = Mat::zeros(num_matches as i32, 2, opencv::core::CV_64F)
        .unwrap()
        .to_mat()
        .unwrap();

    println!("Заполнение точек для первой (референсной) камеры...");
    for (j, matches) in all_matches[0].iter().enumerate() {
        let match_ref = matches.get(0).unwrap();
        let kp = keypoints_list[0].get(match_ref.query_idx as usize).unwrap();
        *points_cam1.at_2d_mut::<f64>(j as i32, 0).unwrap() = kp.pt().x as f64;
        *points_cam1.at_2d_mut::<f64>(j as i32, 1).unwrap() = kp.pt().y as f64;
    }
    points_2d.push(points_cam1);

    println!("Заполнение точек для остальных камер...");
    for i in 1..camera_params.len() {
        println!("  -> Заполнение точек для камеры {}", i + 1);
        let mut points_cam = Mat::zeros(num_matches as i32, 2, opencv::core::CV_64F)
            .unwrap()
            .to_mat()
            .unwrap();

        for (j, matches) in all_matches[i - 1].iter().enumerate() {
            let match_ref = matches.get(0).unwrap();
            let kp = keypoints_list[i].get(match_ref.train_idx as usize).unwrap();
            *points_cam.at_2d_mut::<f64>(j as i32, 0).unwrap() = kp.pt().x as f64;
            *points_cam.at_2d_mut::<f64>(j as i32, 1).unwrap() = kp.pt().y as f64;
        }
        points_2d.push(points_cam);
    }
    println!(
        "Подготовка матриц точек завершена за {:?}",
        start_time.elapsed()
    );

    // НОВЫЙ КОД: Undistort points before triangulation
    println!("==== ЭТАП 7.5: Исправление дисторсии точек ====");
    let start_time = Instant::now();

    println!("Исправление дисторсии для всех точек перед триангуляцией...");
    let mut undistorted_points_2d = Vector::<Mat>::default();

    for (i, points) in points_2d.iter().enumerate() {
        println!("  -> Исправление дисторсии для камеры {}", i + 1);

        // Преобразуем формат точек для функции undistortPoints
        // undistortPoints ожидает Nx1 матрицу с 2 каналами (x,y)
        let num_points = points.rows();
        let mut points_for_undistort = Mat::zeros(num_points, 1, opencv::core::CV_64FC2)
            .unwrap()
            .to_mat()
            .unwrap();

        for j in 0..num_points {
            let x = *points.at_2d::<f64>(j, 0).unwrap();
            let y = *points.at_2d::<f64>(j, 1).unwrap();

            // Установка значений в 2-канальную матрицу
            let pt = points_for_undistort
                .at_2d_mut::<opencv::core::Vec2d>(j, 0)
                .unwrap();
            pt[0] = x;
            pt[1] = y;
        }

        // Создаем матрицу для результата
        let mut undistorted_points = Mat::zeros(num_points, 1, opencv::core::CV_64FC2)
            .unwrap()
            .to_mat()
            .unwrap();

        // Исправление дисторсии
        opencv::calib3d::undistort_points(
            &points_for_undistort,
            &mut undistorted_points,
            &camera_params[i].intrinsic,
            &camera_params[i].distortion,
            &Mat::default(),             // Не используем R (матрицу ректификации)
            &camera_params[i].intrinsic, // Используем исходную матрицу камеры как P
        )
        .unwrap();

        // Преобразуем обратно в формат Nx2
        let mut undistorted_nx2 = Mat::zeros(num_points, 2, opencv::core::CV_64F)
            .unwrap()
            .to_mat()
            .unwrap();

        for j in 0..num_points {
            let pt = undistorted_points
                .at_2d::<opencv::core::Vec2d>(j, 0)
                .unwrap();

            *undistorted_nx2.at_2d_mut::<f64>(j, 0).unwrap() = pt[0];
            *undistorted_nx2.at_2d_mut::<f64>(j, 1).unwrap() = pt[1];
        }

        undistorted_points_2d.push(undistorted_nx2);
    }

    println!(
        "Исправление дисторсии завершено за {:?}",
        start_time.elapsed()
    );

    println!("==== ЭТАП 8: Триангуляция точек ====");
    let start_time = Instant::now();

    let points_3d = match lib_cv::reconstruction::triangulate_points_multiple(
        &undistorted_points_2d,
        &camera_params,
    ) {
        Ok(points) => {
            println!(
                "Триангуляция успешно выполнена. Получено {} 3D точек за {:?}",
                points.len(),
                start_time.elapsed()
            );
            points
        }
        Err(e) => {
            eprintln!("Ошибка при триангуляции точек: {:?}", e);
            return;
        }
    };

    println!("==== ЭТАП 9: Создание облака точек ====");
    let start_time = Instant::now();

    // Получаем цвета точек из первого изображения (референсной камеры)
    println!("Создание структуры облака точек...");
    let mut cloud = PointCloud {
        points: points_3d,
        timestamp: current_i as usize,
    };

    // Добавляем цвет из исходного изображения
    println!("Добавление цветовой информации...");
    let mut colored_count = 0;
    for (i, point) in cloud.points.iter_mut().enumerate() {
        let x = *undistorted_points_2d
            .get(0)
            .unwrap()
            .at_2d::<f64>(i as i32, 0)
            .unwrap() as i32;
        let y = *undistorted_points_2d
            .get(0)
            .unwrap()
            .at_2d::<f64>(i as i32, 1)
            .unwrap() as i32;

        // Проверяем, что координаты в пределах изображения
        if x >= 0 && y >= 0 && x < images[0].cols() && y < images[0].rows() {
            let color = images[0].at_2d::<opencv::core::Vec3b>(y, x).unwrap();
            point.color = Some((color[2], color[1], color[0])); // BGR -> RGB
            colored_count += 1;
        }
    }
    println!(
        "Добавлена цветовая информация для {} из {} точек",
        colored_count,
        cloud.points.len()
    );

    // Фильтрация по уверенности
    println!("Фильтрация точек по уверенности...");
    let initial_count = cloud.points.len();
    let confidence_threshold = 0.0;
    cloud
        .points
        .retain(|point| point.confidence >= confidence_threshold);
    println!(
        "Отфильтровано {} точек (оставлено {})",
        initial_count - cloud.points.len(),
        cloud.points.len()
    );
    println!(
        "Обработка облака точек завершена за {:?}",
        start_time.elapsed()
    );

    println!("==== ЭТАП 10: Отображение результатов и сохранение ====");
    let start_time = Instant::now();

    println!("Сохранение облака точек в PLY файл...");
    let filename = format!("point_cloud_{}.ply", current_i);
    match save_point_cloud(&cloud, &filename) {
        Ok(_) => println!("Облако точек успешно сохранено в файл: {}", filename),
        Err(e) => eprintln!("Ошибка при сохранении облака точек: {:?}", e),
    };

    println!("Ожидание нажатия клавиши...");

    println!("Общее время выполнения: {:?}", total_start_time.elapsed());
    println!("==== Программа завершена ====");
}
