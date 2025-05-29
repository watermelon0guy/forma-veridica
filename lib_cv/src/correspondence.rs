use std::time::Instant;

use opencv::core::{DMatch, KeyPoint, NORM_L2, Vector};
use opencv::features2d::{BFMatcher, SIFT};
use opencv::prelude::*;
use opencv::{self, Error};

pub fn sift(image_1: &Mat) -> Result<(Vector<KeyPoint>, Mat), Error> {
    let mut sift = SIFT::create(0, 10, 0.04, 10f64, 1.6, false).unwrap();

    let mut keypoints_1 = Vector::<KeyPoint>::default();

    let mut descriptors_1 = Mat::default();

    let mask = Mat::default();
    let stopwatch = Instant::now();
    sift.detect_and_compute_def(&image_1, &mask, &mut keypoints_1, &mut descriptors_1)
        .unwrap();
    println!(
        "Нахождение признаков методом SIFT заняло {:?}",
        stopwatch.elapsed()
    );
    Ok((keypoints_1, descriptors_1))
}

pub fn bf_match(
    descriptors_1: &Mat,
    descriptors_2: &Mat,
    threshold: f32,
) -> Result<Vector<DMatch>, Error> {
    let mut bf_matcher = BFMatcher::create(NORM_L2, false)?;
    let mut matched_descriptors = Vector::<DMatch>::default();
    bf_matcher.add(&descriptors_1)?;
    bf_matcher.match__def(&descriptors_2, &mut matched_descriptors)?;

    let filtered_matches: Vector<DMatch> = matched_descriptors
        .into_iter()
        .filter(|m| m.distance < threshold)
        .collect();
    Ok(filtered_matches)
}

pub fn bf_match_knn(
    descriptors_1: &Mat,
    descriptors_2: &Mat,
    neighbours_amount: i32,
    ratio: f32,
) -> Result<Vector<Vector<DMatch>>, Error> {
    let bf_matcher = BFMatcher::create(NORM_L2, false)?;
    let mut matched_descriptors = Vector::<Vector<DMatch>>::default();
    bf_matcher.knn_train_match_def(
        &descriptors_1,
        &descriptors_2,
        &mut matched_descriptors,
        neighbours_amount,
    )?;

    let stopwatch = Instant::now();
    let filtered_matches: Vector<Vector<DMatch>> = matched_descriptors
        .into_iter()
        .filter(|n| {
            n.len() >= 2
                && n.get(0)
                    .expect("Ошибка при считывании дескриптора из массива соседей")
                    .distance
                    < ratio
                        * n.get(1)
                            .expect("Ошибка при считывании дескриптора из массива соседей")
                            .distance
        })
        .collect();
    println!(
        "Сопоставление признаков методом Brute Force заняло {:?}",
        stopwatch.elapsed()
    );

    Ok(filtered_matches)
}
