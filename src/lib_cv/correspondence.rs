use opencv::core::{KeyPoint, Vector};
use opencv::features2d::{draw_keypoints, draw_keypoints_def, SIFT};
use opencv::prelude::*;
use opencv::{self, Error};

pub fn sift(image_1: &Mat) -> Result<(Vector<KeyPoint>, Mat), Error> {
    let mut sift = SIFT::create(0, 3, 0.04, 10f64, 1.6, false).unwrap();

    let mut keypoints_1 = Vector::<KeyPoint>::default();

    let mut descriptors_1 = Mat::default();

    let mask = Mat::default();

    sift.detect_and_compute_def(&image_1, &mask, &mut keypoints_1, &mut descriptors_1)
        .unwrap();

    Ok((keypoints_1, descriptors_1))
}
