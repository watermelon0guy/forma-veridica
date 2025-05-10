use lib_cv::calibration::{self, get_charuco};
use lib_cv::utils::{combine_quadrants, split_image_into_quadrants};
use opencv::core::{Scalar, Size};
use opencv::features2d::{DrawMatchesFlags, draw_keypoints, draw_matches_knn_def};
use opencv::gapi::crop;
use opencv::imgcodecs;
use opencv::objdetect::{CharucoBoard, draw_detected_corners_charuco, draw_detected_markers};
use opencv::videoio::{CAP_PROP_POS_FRAMES, VideoCapture};
use opencv::{highgui, prelude::*};

fn main() {
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
    while true {
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
            32 => {}
            27 => break, // Escape key
            _ => {}
        }
        println!("obj_points_1: {:?}", obj_points_1);
        println!("charuco_corners_1: {:?}", charuco_corners_1);
        println!("img_points_1: {:?}", img_points_1);
        println!("charuco_ids_1: {:?}", charuco_ids_1);

        // let mut image_with_markers = cropped.clone();
        // draw_detected_markers(
        //     &mut image_with_markers,
        //     &marker_corners,
        //     &marker_ids,
        //     Scalar::new(255.0, 0.0, 0.0, 255.0),
        // )
        // .unwrap();

        // highgui::imshow("Charuco Доска", &image_with_markers).unwrap();
        // highgui::wait_key(0).unwrap();
    }
}
