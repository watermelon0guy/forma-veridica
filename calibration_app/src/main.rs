use std::fs;

use lib_cv::calibration::{self, CameraParameters, get_charuco};
use lib_cv::utils::{combine_quadrants, split_image_into_quadrants};
use opencv::core::{Scalar, Vector};
use opencv::imgcodecs;
use opencv::objdetect::{CharucoBoard, draw_detected_corners_charuco};
use opencv::videoio::{CAP_PROP_POS_FRAMES, VideoCapture};
use opencv::{highgui, prelude::*};

fn main() {
    const IMAGE_PATH: &str =
        "/home/watermelon0guy/Изображения/Experiments/raspberry_pi_cardboard/calibration";

    highgui::named_window("Charuco Доска", highgui::WINDOW_KEEPRATIO).unwrap();

    let mut frames = Vec::new();

    {
        let mut cap = VideoCapture::from_file(
        "/home/watermelon0guy/Видео/Experiments/raspberry_pi_cardboard/20250427_095907_hires.mp4",
        opencv::videoio::CAP_ANY,
    )
    .unwrap();
        let mut frame = opencv::core::Mat::default();

        while cap.read(&mut frame).unwrap() {
            frames.push(frame.clone());
        }
        cap.set(CAP_PROP_POS_FRAMES, 0.0).unwrap();
    }

    let dictionary = opencv::objdetect::get_predefined_dictionary(
        opencv::objdetect::PredefinedDictionaryType::DICT_4X4_50,
    )
    .unwrap();
    let charuco_board = opencv::objdetect::CharucoBoard::new_def(
        opencv::core::Size::new(5, 5),
        10.0,
        7.0,
        &dictionary,
    )
    .unwrap();

    let mut current_i = 0;
    loop {
        let Some(current_frame) = frames.get(current_i) else {
            eprintln!("Не получилось считать кадр");
            continue;
        };

        let Ok((img_1, img_2, img_3, img_4)) = split_image_into_quadrants(&current_frame) else {
            eprintln!("Не получилось разбить изображение");
            continue;
        };

        let Ok((
            marker_corners_1,
            marker_ids_1,
            charuco_corners_1,
            charuco_ids_1,
            obj_points_1,
            img_points_1,
        )) = get_charuco(&charuco_board, &img_1)
        else {
            eprintln!("Ошибка при извлечении Charuco углов");
            continue;
        };
        let mut edited_1 = img_1.clone();
        draw_detected_corners_charuco(
            &mut edited_1,
            &charuco_corners_1,
            &charuco_ids_1,
            Scalar::new(0.0, 255.0, 0.0, 255.0),
        )
        .expect("Не получилось нарисовать на изображении углы Charuco");

        let Ok((
            marker_corners_2,
            marker_ids_2,
            charuco_corners_2,
            charuco_ids_2,
            obj_points_2,
            img_points_2,
        )) = get_charuco(&charuco_board, &img_2)
        else {
            eprintln!("Ошибка при извлечении Charuco углов");
            continue;
        };
        let mut edited_2 = img_2.clone();
        draw_detected_corners_charuco(
            &mut edited_2,
            &charuco_corners_2,
            &charuco_ids_2,
            Scalar::new(0.0, 255.0, 0.0, 255.0),
        )
        .expect("Не получилось нарисовать на изображении углы Charuco");

        let Ok((
            marker_corners_3,
            marker_ids_3,
            charuco_corners_3,
            charuco_ids_3,
            obj_points_3,
            img_points_3,
        )) = get_charuco(&charuco_board, &img_3)
        else {
            eprintln!("Ошибка при извлечении Charuco углов");
            continue;
        };
        let mut edited_3 = img_3.clone();
        draw_detected_corners_charuco(
            &mut edited_3,
            &charuco_corners_3,
            &charuco_ids_3,
            Scalar::new(0.0, 255.0, 0.0, 255.0),
        )
        .expect("Не получилось нарисовать на изображении углы Charuco");

        let Ok((
            marker_corners_4,
            marker_ids_4,
            charuco_corners_4,
            charuco_ids_4,
            obj_points_4,
            img_points_4,
        )) = get_charuco(&charuco_board, &img_4)
        else {
            eprintln!("Ошибка при извлечении Charuco углов");
            continue;
        };
        let mut edited_4 = img_4.clone();
        draw_detected_corners_charuco(
            &mut edited_4,
            &charuco_corners_4,
            &charuco_ids_4,
            Scalar::new(0.0, 255.0, 0.0, 255.0),
        )
        .expect("Не получилось нарисовать на изображении углы Charuco");

        let Ok(edited_combined) = combine_quadrants(&edited_1, &edited_2, &edited_3, &edited_4)
        else {
            eprintln!("Ошибка в сшивании 4 изображений");
            continue;
        };

        highgui::imshow("Charuco Доска", &edited_combined).unwrap();

        let key = highgui::wait_key(0).unwrap();
        match key {
            83 => {
                current_i += 1;
            }
            81 => {
                current_i -= 1;
            }
            32 => {
                let timestamp = current_i.to_string();
                imgcodecs::imwrite(
                    &format!("{}/img_1_{}.png", IMAGE_PATH, timestamp),
                    &img_1,
                    &opencv::core::Vector::new(),
                )
                .unwrap();
                imgcodecs::imwrite(
                    &format!("{}/img_2_{}.png", IMAGE_PATH, timestamp),
                    &img_2,
                    &Vector::new(),
                )
                .unwrap();
                imgcodecs::imwrite(
                    &format!("{}/img_3_{}.png", IMAGE_PATH, timestamp),
                    &img_3,
                    &Vector::new(),
                )
                .unwrap();
                imgcodecs::imwrite(
                    &format!("{}/img_4_{}.png", IMAGE_PATH, timestamp),
                    &img_4,
                    &Vector::new(),
                )
                .unwrap();
                println!("Изображения сохранены с timestamp: {}", timestamp);
            }
            27 => break, // Escape key
            _ => {}
        }
    }
    perform_calibration(IMAGE_PATH, &charuco_board, 4);
}

fn perform_calibration(image_path: &str, charuco_board: &CharucoBoard, num_cameras: usize) {
    println!("Поиск калибровочных изображений в: {}", image_path);

    // Собираем все файлы в директории
    let dir_entries = match fs::read_dir(image_path) {
        Ok(entries) => entries,
        Err(e) => {
            eprintln!("Ошибка чтения директории: {}", e);
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

        if file_name.starts_with("img_") && file_name.ends_with(".png") {
            let parts: Vec<&str> = file_name.split('_').collect();
            if parts.len() == 3 {
                if let Ok(cam_num) = parts[1].parse::<usize>() {
                    if let Ok(frame_num) = parts[2].trim_end_matches(".png").parse::<usize>() {
                        if let Ok(img) = imgcodecs::imread(
                            &entry.path().to_string_lossy(),
                            imgcodecs::IMREAD_COLOR,
                        ) {
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

    println!("Найдено {} наборов(сцен) изображений", frame_numbers.len());

    // Выполняем калибровку
    match calibration::calibrate_multiple_with_charuco(&camera_images, charuco_board) {
        Ok(cameras) => {
            println!(
                "Калибровка успешно завершена. Получено {} камер:",
                cameras.len()
            );
            for (i, cam) in cameras.iter().enumerate() {
                println!("\nКамера {}:", i + 1);
                println!("Матрица внутренних параметров:");
                println!("{:?}", cam.intrinsic);
                println!("Коэффициенты искажения:");
                println!("{:?}", cam.distortion);

                if i > 0 {
                    println!("Матрица вращения относительно основной камеры:");
                    println!("{:?}", cam.rotation);
                    println!("Вектор трансляции относительно основной камеры:");
                    println!("{:?}", cam.translation);
                }
            }

            // Сохранение параметров в файл (опционально)
            if let Err(e) =
                save_camera_parameters(&cameras, &format!("{}calibration_params.yml", image_path))
            {
                eprintln!("Ошибка при сохранении параметров: {}", e);
            }
        }
        Err(e) => eprintln!("Ошибка при калибровке: {:?}", e),
    }
}

fn save_camera_parameters(cameras: &[CameraParameters], path: &str) -> opencv::Result<()> {
    use opencv::core::FileStorage;
    use opencv::core::FileStorage_Mode;

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
