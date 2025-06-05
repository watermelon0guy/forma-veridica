use log::debug;
use opencv::core::{DMatch, KeyPoint, NORM_L2, Vector};
use opencv::features2d::{BFMatcher, SIFT};
use opencv::prelude::*;
use opencv::{self, Error};

pub fn sift(
    image_1: &Mat,
    nfeatures: i32,
    n_octave_layers: i32,
    contrast_threshold: f64,
    edge_threshold: f64,
    sigma: f64,
    use_provided_keypoints: bool,
) -> Result<(Vector<KeyPoint>, Mat), Error> {
    let mut sift = SIFT::create(
        nfeatures,
        n_octave_layers,
        contrast_threshold,
        edge_threshold,
        sigma,
        use_provided_keypoints,
    )?;

    let mut keypoints_1 = Vector::<KeyPoint>::default();

    let mut descriptors_1 = Mat::default();

    let mask = Mat::default();
    sift.detect_and_compute_def(&image_1, &mask, &mut keypoints_1, &mut descriptors_1)?;
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

    Ok(filtered_matches)
}

pub fn gather_points_2d_from_matches(
    all_matches: &Vec<Vector<Vector<DMatch>>>,
    all_keypoints: &Vec<Vector<KeyPoint>>,
) -> Result<Vector<Mat>, Error> {
    // Создаем матрицы с 2D точками для всех камер
    let mut points_2d = Vector::<Mat>::default();

    // Для первой (референсной) камеры
    let num_matches = all_matches[0].len();
    debug!("Общее количество сопоставленных точек: {}", num_matches);
    let mut points_cam_1 = Mat::zeros(num_matches as i32, 2, opencv::core::CV_64F)?.to_mat()?;

    for (j, matches) in all_matches[0].iter().enumerate() {
        let match_ref = matches.get(0)?;
        let kp = all_keypoints[0].get(match_ref.query_idx as usize)?;
        *points_cam_1.at_2d_mut::<f64>(j as i32, 0)? = kp.pt().x as f64;
        *points_cam_1.at_2d_mut::<f64>(j as i32, 1)? = kp.pt().y as f64;
    }
    points_2d.push(points_cam_1);

    for i in 1..all_matches.len() + 1 {
        let mut points_cam = Mat::zeros(num_matches as i32, 2, opencv::core::CV_64F)?.to_mat()?;

        for (j, matches) in all_matches[i - 1].iter().enumerate() {
            let match_ref = matches.get(0)?;
            let kp = all_keypoints[i].get(match_ref.train_idx as usize)?;
            *points_cam.at_2d_mut::<f64>(j as i32, 0)? = kp.pt().x as f64;
            *points_cam.at_2d_mut::<f64>(j as i32, 1)? = kp.pt().y as f64;
        }
        points_2d.push(points_cam);
    }

    Ok(points_2d)
}
