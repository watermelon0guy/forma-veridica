use opencv::{Error, prelude::*};

pub fn split_image_into_quadrants(img: &Mat) -> Result<(Mat, Mat, Mat, Mat), Error> {
    let roi_1 = Mat::roi(
        img,
        opencv::core::Rect::new(0, 0, img.cols() / 2, img.rows() / 2),
    )?;
    let roi_2 = Mat::roi(
        img,
        opencv::core::Rect::new(img.cols() / 2, 0, img.cols() / 2, img.rows() / 2),
    )?;
    let roi_3 = Mat::roi(
        img,
        opencv::core::Rect::new(0, img.rows() / 2, img.cols() / 2, img.rows() / 2),
    )?;
    let roi_4 = Mat::roi(
        img,
        opencv::core::Rect::new(
            img.cols() / 2,
            img.rows() / 2,
            img.cols() / 2,
            img.rows() / 2,
        ),
    )?;
    let mut cropped_1 = Mat::default();
    roi_1.copy_to(&mut cropped_1).unwrap();
    let mut cropped_2 = Mat::default();
    roi_2.copy_to(&mut cropped_2).unwrap();
    let mut cropped_3 = Mat::default();
    roi_3.copy_to(&mut cropped_3).unwrap();
    let mut cropped_4 = Mat::default();
    roi_4.copy_to(&mut cropped_4).unwrap();
    Ok((cropped_1, cropped_2, cropped_3, cropped_4))
}
