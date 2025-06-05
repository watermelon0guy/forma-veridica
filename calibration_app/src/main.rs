use std::path::Path;

use lib_cv::calibration::{get_charuco, perform_calibration};
use lib_cv::utils::{combine_quadrants, split_image_into_quadrants, video_to_frames};
use log::info;
use opencv::core::{Scalar, Vector};
use opencv::highgui;
use opencv::imgcodecs;
use opencv::objdetect::draw_detected_corners_charuco;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    const PICKED_IMAGE_PATH: &str =
        "/home/watermelon0guy/Изображения/Experiments/raspberry_pi_cardboard/calibration/picked";
    const PARSED_IMAGE_PATH: &str =
        "/home/watermelon0guy/Изображения/Experiments/raspberry_pi_cardboard/calibration/parsed";
    const VIDEO_PATH: &str =
        "/home/watermelon0guy/Видео/Experiments/raspberry_pi_cardboard/20250603_113751_hires.mp4";
    const CAMERAS_PARAMS_PATH: &str =
        "/home/watermelon0guy/Изображения/Experiments/raspberry_pi_cardboard/calibration";
    highgui::named_window("Charuco Доска", highgui::WINDOW_KEEPRATIO).unwrap();

    video_to_frames(Path::new(VIDEO_PATH), Path::new(PARSED_IMAGE_PATH)).unwrap();

    let dictionary = opencv::objdetect::get_predefined_dictionary(
        opencv::objdetect::PredefinedDictionaryType::DICT_4X4_50,
    )
    .unwrap();
    let charuco_board = opencv::objdetect::CharucoBoard::new_def(
        opencv::core::Size::new(10, 5),
        13.0,
        9.1,
        &dictionary,
    )
    .unwrap();

    let mut current_i = 0;
    loop {
        let current_frame = match imgcodecs::imread(
            &format!("{}/{}.png", PARSED_IMAGE_PATH, current_i),
            imgcodecs::IMREAD_COLOR,
        ) {
            Ok(frame) => frame,
            Err(_) => {
                eprintln!("Не получилось считать кадр");
                continue;
            }
        };

        let Ok(quadrants) = split_image_into_quadrants(&current_frame) else {
            eprintln!("Не получилось разбить изображение");
            continue;
        };

        let img_1 = quadrants[0].clone();
        let img_2 = quadrants[1].clone();
        let img_3 = quadrants[2].clone();
        let img_4 = quadrants[3].clone();

        let Ok((
            _marker_corners_1,
            _marker_ids_1,
            charuco_corners_1,
            charuco_ids_1,
            _obj_points_1,
            _img_points_1,
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
            _marker_corners_2,
            _marker_ids_2,
            charuco_corners_2,
            charuco_ids_2,
            _obj_points_2,
            _img_points_2,
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
            _marker_corners_3,
            _marker_ids_3,
            charuco_corners_3,
            charuco_ids_3,
            _obj_points_3,
            _img_points_3,
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
            _marker_corners_4,
            _marker_ids_4,
            charuco_corners_4,
            charuco_ids_4,
            _obj_points_4,
            _img_points_4,
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
                    &format!("{}/img_1_{}.png", PICKED_IMAGE_PATH, timestamp),
                    &img_1,
                    &opencv::core::Vector::new(),
                )
                .unwrap();
                imgcodecs::imwrite(
                    &format!("{}/img_2_{}.png", PICKED_IMAGE_PATH, timestamp),
                    &img_2,
                    &Vector::new(),
                )
                .unwrap();
                imgcodecs::imwrite(
                    &format!("{}/img_3_{}.png", PICKED_IMAGE_PATH, timestamp),
                    &img_3,
                    &Vector::new(),
                )
                .unwrap();
                imgcodecs::imwrite(
                    &format!("{}/img_4_{}.png", PICKED_IMAGE_PATH, timestamp),
                    &img_4,
                    &Vector::new(),
                )
                .unwrap();
                info!("Изображения сохранены с timestamp: {}", timestamp);
            }
            27 => break, // Escape key
            _ => {}
        }
    }
    perform_calibration(
        &PICKED_IMAGE_PATH,
        &Path::new(CAMERAS_PARAMS_PATH),
        &charuco_board,
        4,
    );
}
