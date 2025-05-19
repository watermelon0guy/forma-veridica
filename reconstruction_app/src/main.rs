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
    const _IMAGE_PATH: &str =
        "/home/watermelon0guy/Изображения/Experiments/raspberry_pi_cardboard/calibration";
    println!("Путь к изображениям: {}", _IMAGE_PATH);

    println!("==== ЭТАП 1: Загрузка параметров камеры ====");
    let start_time = Instant::now();
    let camera_params = load_camera_parameters("/home/watermelon0guy/Изображения/Experiments/raspberry_pi_cardboard/calibration_params.yml")
        .expect("Не получилось загрузить параметры камеры");
    println!(
        "Загружено {} наборов параметров камер за {:?}",
        camera_params.len(),
        start_time.elapsed()
    );

    println!("==== ЭТАП 2: Открытие видеофайла ====");
    let start_time = Instant::now();
    let mut cap = VideoCapture::from_file(
        "/home/watermelon0guy/Видео/Experiments/raspberry_pi_cardboard/20250427_093804_hires.mp4",
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

    println!("==== ЭТАП 8: Триангуляция точек ====");
    let start_time = Instant::now();

    let points_3d =
        match lib_cv::reconstruction::triangulate_points_multiple(&points_2d, &camera_params) {
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
        let x = *points_2d.get(0).unwrap().at_2d::<f64>(i as i32, 0).unwrap() as i32;
        let y = *points_2d.get(0).unwrap().at_2d::<f64>(i as i32, 1).unwrap() as i32;

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
