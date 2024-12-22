mod lib_cv;
use opencv::{features2d::draw_keypoints_def, imgcodecs};
use opencv::{highgui, prelude::*};

use lib_cv::correspondence;

fn main() {
    let image = imgcodecs::imread(
        "/home/watermelon0guy/Изображения/Image Dataset/images/DSCN4698.JPG",
        imgcodecs::IMREAD_COLOR,
    )
    .unwrap();
    let mut image_with_keypoints = Mat::default();

    let (keypoints, descriptors) = correspondence::sift(&image).unwrap();

    draw_keypoints_def(&image, &keypoints, &mut image_with_keypoints).unwrap();

    highgui::named_window("Ключевые точки", highgui::WINDOW_KEEPRATIO).unwrap();
    highgui::imshow("Ключевые точки", &image_with_keypoints).unwrap();
    highgui::wait_key(0).unwrap();
}
