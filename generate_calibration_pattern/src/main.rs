use opencv::{
    core::{Mat, Size2d},
    prelude::BoardTraitConst,
};

fn main() {
    let dictionary = opencv::objdetect::get_predefined_dictionary(
        opencv::objdetect::PredefinedDictionaryType::DICT_4X4_50,
    )
    .unwrap();
    let charuco_board = opencv::objdetect::CharucoBoard::new_def(
        opencv::core::Size::new(10, 5),
        10.0,
        7.0,
        &dictionary,
    )
    .unwrap();
    let mut img = Mat::default();
    charuco_board
        .generate_image(opencv::core::Size::new(1000, 500), &mut img, 20, 1)
        .unwrap();
    opencv::imgcodecs::imwrite("charuco_board.png", &img, &opencv::core::Vector::new()).unwrap();
}
