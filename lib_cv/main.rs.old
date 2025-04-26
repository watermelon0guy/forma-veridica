mod lib_cv;
use opencv::core::Scalar;
use opencv::features2d::{draw_keypoints, draw_matches_knn_def, DrawMatchesFlags};
use opencv::imgcodecs;
use opencv::{highgui, prelude::*};

use lib_cv::correspondence::{self, bf_match_knn};

fn main() {
    highgui::named_window("Ключевые точки", highgui::WINDOW_KEEPRATIO).unwrap();

    let image_1 = imgcodecs::imread(
        "/home/watermelon0guy/Изображения/Image Dataset/test/1.png",
        imgcodecs::IMREAD_COLOR,
    )
    .unwrap();
    let image_2 = imgcodecs::imread(
        "/home/watermelon0guy/Изображения/Image Dataset/test/2.png",
        imgcodecs::IMREAD_COLOR,
    )
    .unwrap();
    let mut image_with_keypoints = Mat::default();
    let (keypoints_1, descriptors_1) = correspondence::sift(&image_1).unwrap();
    let (keypoints_2, descriptors_2) = correspondence::sift(&image_2).unwrap();

    let _ = draw_keypoints(
        &image_1,
        &keypoints_1,
        &mut image_with_keypoints,
        Scalar::all(-1.0),
        DrawMatchesFlags::DRAW_RICH_KEYPOINTS,
    );

    highgui::imshow("Ключевые точки", &image_with_keypoints).unwrap();
    highgui::wait_key(0).unwrap();

    let _ = draw_keypoints(
        &image_2,
        &keypoints_2,
        &mut image_with_keypoints,
        Scalar::all(-1.0),
        DrawMatchesFlags::DRAW_RICH_KEYPOINTS,
    );

    highgui::imshow("Ключевые точки", &image_with_keypoints).unwrap();
    highgui::wait_key(0).unwrap();

    let matched_descriptors = match bf_match_knn(&descriptors_1, &descriptors_2, 2, 0.75) {
        Ok(val) => val,
        Err(_) => {
            println!("Ошибка в bf_match");
            return;
        }
    };
    let mut image_matched = Mat::default();

    let _ = draw_matches_knn_def(
        &image_1,
        &keypoints_1,
        &image_2,
        &keypoints_2,
        &matched_descriptors,
        &mut image_matched,
    );

    highgui::imshow("Ключевые точки", &image_matched).unwrap();
    highgui::wait_key(0).unwrap();
}
