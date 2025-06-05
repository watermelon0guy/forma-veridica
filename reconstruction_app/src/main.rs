use lib_cv::calibration::load_camera_parameters;
use lib_cv::reconstruction::{
    PointCloud, add_color_to_point_cloud, filter_point_cloud_by_confindence,
    match_first_camera_features_to_all, min_visible_match_set, save_point_cloud,
};
use lib_cv::utils::split_image_into_quadrants;
use log::{debug, error, info};
use opencv::core::{Point2f, Vector};
use opencv::video::calc_optical_flow_pyr_lk;
use opencv::videoio::VideoCapture;
use opencv::{highgui, prelude::*};
use std::time::Instant;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    const POINT_CLOUD_PATH: &str =
        "/home/watermelon0guy/Изображения/Experiments/raspberry_pi_cardboard/point_clouds";
    const IMAGES_PATH: &str =
        "/home/watermelon0guy/Изображения/Experiments/raspberry_pi_cardboard/calibration/picked";
    debug!("Путь к изображениям: {}", IMAGES_PATH);
    let start_time = Instant::now();
    let camera_params = load_camera_parameters("/home/watermelon0guy/Изображения/Experiments/raspberry_pi_cardboard/calibration/calibration_params.yml")
        .expect("Не получилось загрузить параметры камеры");
    info!(
        "Загружено {} наборов параметров камер за {:?}",
        camera_params.len(),
        start_time.elapsed()
    );
    let mut cap = VideoCapture::from_file(
        "/home/watermelon0guy/Видео/Experiments/raspberry_pi_cardboard/20250603_114637_hires.mp4",
        opencv::videoio::CAP_ANY,
    )
    .unwrap();
    let mut fr = 0;

    let start_time = Instant::now();
    let mut frame = opencv::core::Mat::default();
    cap.read(&mut frame).unwrap();
    debug!(
        "Кадр считан. Размер: {}x{} за {:?}",
        frame.cols(),
        frame.rows(),
        start_time.elapsed()
    );

    let mut current_i = 0;

    let start_time = Instant::now();
    let images = match split_image_into_quadrants(&frame) {
        Ok(it) => {
            debug!(
                "Изображение успешно разделено на {} части за {:?}",
                it.len(),
                start_time.elapsed()
            );
            it
        }
        Err(e) => {
            error!("Ошибка при разделении изображения: {:?}", e);
            return;
        }
    };

    let (mut all_matches, keypoints_list, descriptors_list) =
        match_first_camera_features_to_all(&images);

    let start_time = Instant::now();

    all_matches = min_visible_match_set(&mut all_matches, &keypoints_list);

    info!("Фильтрация завершена за {:?}", start_time.elapsed());

    let start_time = Instant::now();

    // Создаем матрицы с 2D точками для всех камер
    let mut points_2d = Vector::<Mat>::default();

    // Для первой (референсной) камеры
    let num_matches = all_matches[0].len();
    info!("Общее количество сопоставленных точек: {}", num_matches);
    let mut points_cam_1 = Mat::zeros(num_matches as i32, 2, opencv::core::CV_64F)
        .unwrap()
        .to_mat()
        .unwrap();

    for (j, matches) in all_matches[0].iter().enumerate() {
        let match_ref = matches.get(0).unwrap();
        let kp = keypoints_list[0].get(match_ref.query_idx as usize).unwrap();
        *points_cam_1.at_2d_mut::<f64>(j as i32, 0).unwrap() = kp.pt().x as f64;
        *points_cam_1.at_2d_mut::<f64>(j as i32, 1).unwrap() = kp.pt().y as f64;
    }
    points_2d.push(points_cam_1);

    for i in 1..camera_params.len() {
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
    info!(
        "Подготовка матриц точек завершена за {:?}",
        start_time.elapsed()
    );

    let start_time = Instant::now();

    let mut undistorted_points_2d = Vector::<Mat>::default();

    for (i, points) in points_2d.iter().enumerate() {
        let num_points = points.rows();
        let mut undistorted_points = Mat::zeros(num_points, 1, opencv::core::CV_64FC2)
            .unwrap()
            .to_mat()
            .unwrap();

        // Исправление дисторсии
        opencv::calib3d::undistort_points(
            &points,
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

    info!(
        "Исправление дисторсии завершено за {:?}",
        start_time.elapsed()
    );

    let start_time = Instant::now();

    let points_3d = match lib_cv::reconstruction::triangulate_points_multiple(
        &undistorted_points_2d,
        &camera_params,
    ) {
        Ok(points) => {
            info!(
                "Триангуляция успешно выполнена. Получено {} 3D точек за {:?}",
                points.len(),
                start_time.elapsed()
            );
            points
        }
        Err(e) => {
            error!("Ошибка при триангуляции точек: {:?}", e);
            return;
        }
    };

    let start_time = Instant::now();

    let mut cloud = PointCloud {
        points: points_3d,
        timestamp: current_i as usize,
    };

    add_color_to_point_cloud(&mut cloud, &points_2d, &images[0]);

    // Фильтрация по уверенности
    let initial_count = cloud.points.len();
    filter_point_cloud_by_confindence(&mut cloud, 0.5);
    info!(
        "Отфильтровано {} точек (оставлено {})",
        initial_count - cloud.points.len(),
        cloud.points.len()
    );
    info!(
        "Обработка облака точек завершена за {:?}",
        start_time.elapsed()
    );

    let filename = format!("{}/point_cloud_{}.ply", POINT_CLOUD_PATH, current_i);
    match save_point_cloud(&cloud, &filename) {
        Ok(_) => info!("Облако точек успешно сохранено в файл: {}", filename),
        Err(e) => error!("Ошибка при сохранении облака точек: {:?}", e),
    };

    let mut prev_image = images;
    let mut next_image = Vec::default();

    let mut prev_points: Vec<Vector<Point2f>> =
        vec![Vector::<Point2f>::default(); camera_params.len()];
    for camera_i in 0..camera_params.len() {
        for j in 0..points_2d.get(camera_i).unwrap().rows() {
            let x = *points_2d
                .get(camera_i as usize)
                .unwrap()
                .at_2d::<f64>(j, 0)
                .unwrap() as f32;
            let y = *points_2d
                .get(camera_i as usize)
                .unwrap()
                .at_2d::<f64>(j, 1)
                .unwrap() as f32;
            prev_points[camera_i].push(opencv::core::Point2f::new(x, y));
        }
    }

    for frame_number in 1..50 {
        cap.read(&mut frame).unwrap();
        next_image = match split_image_into_quadrants(&frame) {
            Ok(it) => {
                debug!("Изображение успешно разделено на {} части", it.len());
                it
            }
            Err(e) => {
                error!("Ошибка при разделении изображения: {:?}", e);
                return;
            }
        };

        let win_size = opencv::core::Size::new(13, 13);
        let max_level = 3;
        let criteria = opencv::core::TermCriteria::new(
            opencv::core::TermCriteria_EPS + opencv::core::TermCriteria_COUNT,
            1000_000,
            0.000_001,
        )
        .unwrap();
        let flags = 0;
        let min_eig_threshold = 1e-4;

        for (camera_i, (prev, next)) in prev_image.iter().zip(next_image.iter()).enumerate() {
            // Подготавливаем данные для оптического потока
            let mut next_points = Vector::<Point2f>::default();
            let mut status = Vector::<u8>::default();
            let mut err = Vector::<f32>::default();

            // Преобразуем points_2d в формат для оптического потока (используем точки первой камеры)

            calc_optical_flow_pyr_lk(
                &prev,
                &next,
                &prev_points[camera_i],
                &mut next_points,
                &mut status,
                &mut err,
                win_size,
                max_level,
                criteria,
                flags,
                min_eig_threshold,
            )
            .unwrap();

            debug!(
                "Потеряно треков: {}",
                status.iter().filter(|&s| s == 0).count()
            );

            // Если это первая камера, показываем точки на изображении
            if camera_i == 0 {
                let mut display_image = next.clone();

                // Рисуем точки на изображении
                for (i, point) in next_points.iter().enumerate() {
                    if i < status.len() && status.get(i).unwrap() == 1 {
                        opencv::imgproc::circle(
                            &mut display_image,
                            opencv::core::Point::new(point.x as i32, point.y as i32),
                            3,
                            opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0), // Зеленый цвет
                            -1,
                            opencv::imgproc::LINE_8,
                            0,
                        )
                        .unwrap();
                    }
                }

                // Показываем изображение
                highgui::named_window("Tracked Points - Camera 0", highgui::WINDOW_NORMAL).unwrap();
                highgui::imshow("Tracked Points - Camera 0", &display_image).unwrap();
                highgui::wait_key(0).unwrap();
            }

            let num_points = next_points.len();
            let mut undistorted_points = Mat::zeros(num_points as i32, 1, opencv::core::CV_64FC2)
                .unwrap()
                .to_mat()
                .unwrap();

            // Исправление дисторсии
            opencv::calib3d::undistort_points(
                &next_points,
                &mut undistorted_points,
                &camera_params[camera_i].intrinsic,
                &camera_params[camera_i].distortion,
                &Mat::default(), // Не используем R (матрицу ректификации)
                &camera_params[camera_i].intrinsic, // Используем исходную матрицу камеры как P
            )
            .unwrap();

            // TODO
            prev_points[camera_i] = next_points;
        }
        prev_image = next_image.clone();
    }
}
