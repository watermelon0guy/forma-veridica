use std::collections::HashMap;

use calib_targets::{
    aruco::Dictionary,
    charuco::CharucoBoard,
    core::{Homography, homography_from_4pt},
};
use image::{RgbImage, RgbaImage};
use log::debug;

use aruco_rs::{Marker, core::dictionary::DictionaryConfig};
use nalgebra::Point2;

/// Временная функция для изучения геометрии доски
pub fn dump_board_info(board: &CharucoBoard) {
    let spec = board.spec();
    debug!(
        "Board spec: rows={}, cols={}, cell_size={}, marker_size_rel={}",
        spec.rows, spec.cols, spec.cell_size, spec.marker_size_rel
    );

    debug!(
        "Expected inner corners: {} rows × {} cols = {} total",
        board.expected_inner_rows(),
        board.expected_inner_cols(),
        board.expected_inner_rows() * board.expected_inner_cols(),
    );

    // Перебери все маркеры
    for (id, pos) in board.iter_marker_positions() {
        debug!("  Marker {id}: cell=({},{})", pos.i, pos.j);
        // Какие углы доски окружают этот маркер?
        if let Some(corners) = board.marker_surrounding_charuco_corners(id as i32) {
            debug!("    surrounds corners: {corners:?}");
        }
    }

    // Перебери все углы доски
    for i in 0..board.expected_inner_rows() as i32 {
        for j in 0..board.expected_inner_cols() as i32 {
            if let Some(corner_id) = board.charuco_corner_id_from_board_corner(i, j) {
                if let Some(xy) = board.charuco_object_xy(corner_id) {
                    debug!(
                        "  Corner ({i},{j}) → id={corner_id} → 3D=({},{})",
                        xy.x, xy.y
                    );
                }
            }
        }
    }
}

pub fn detect_aruco_markers(
    img: &RgbaImage,
    dict: &aruco_rs::core::dictionary::Dictionary,
) -> Vec<Marker> {
    let img_buf = aruco_rs::ImageBuffer {
        data: img.as_raw(),
        width: img.width(),
        height: img.height(),
    };

    let cv = aruco_rs::cv::scalar::ScalarCV;
    let mut detector = aruco_rs::core::detector::Detector::new(&dict, cv);
    detector.adaptive_th_size = 15;
    detector.min_edge_length = 15.0;
    let markers = detector.detect(&img_buf);
    debug!("Обнаружено: {}", markers.len());
    markers
}

/// Переворачивает порядок битов в коде (n_bits младших бит)
fn reverse_bits(mut code: u64, n_bits: usize) -> u64 {
    let mut result = 0u64;
    for _ in 0..n_bits {
        result = (result << 1) | (code & 1);
        code >>= 1;
    }
    result
}

pub fn make_aruco_dict(calib: &Dictionary) -> &'static DictionaryConfig {
    let n_bits = calib.marker_size * calib.marker_size;
    let reversed: Vec<u64> = calib
        .codes
        .iter()
        .map(|&c| reverse_bits(c, n_bits))
        .collect();
    let config = DictionaryConfig {
        n_bits,
        tau: calib.max_correction_bits as usize,
        code_list: Vec::leak(reversed), // 'static
    };
    Box::leak(Box::new(config))
}

pub fn build_marker_homographies(
    board: &CharucoBoard,
    markers: &[Marker],
) -> HashMap<i32, Homography> {
    let spec = board.spec();
    let cell_size = spec.cell_size;
    let marker_size = cell_size * spec.marker_size_rel;
    let half = marker_size / 2.0;

    let mut transforms = HashMap::new();

    for marker in markers {
        let (sx, sy) = match board.marker_cell(marker.id) {
            Some(cell) => cell,
            None => continue,
        };

        let cx = (sx as f32 + 0.5) * cell_size;
        let cy = (sy as f32 + 0.5) * cell_size;
        let src: [Point2<f32>; 4] = [
            Point2::new(cx - half, cy - half), // TL
            Point2::new(cx + half, cy - half), // TR
            Point2::new(cx + half, cy + half), // BR
            Point2::new(cx - half, cy + half), // BL
        ];

        let dst: [nalgebra::Point2<f32>; 4] = marker
            .corners
            .map(|p| nalgebra::Point2::new(p.x as f32, p.y as f32));

        if let Some(h) = homography_from_4pt(&src, &dst) {
            if h.h.determinant().abs() > 1e-6 {
                transforms.insert(marker.id, Homography::from(h));
            }
        }
    }
    transforms
}

pub fn interpolate_charuco_corners(
    board: &CharucoBoard,
    transforms: &HashMap<i32, Homography>,
    min_markers: usize,
) -> Vec<(usize, Point2<f32>)> {
    let mut corners = Vec::new();

    for i in 0..board.expected_inner_rows() as i32 {
        for j in 0..board.expected_inner_cols() as i32 {
            let corner_id = match board.charuco_corner_id_from_board_corner(i, j) {
                Some(id) => id as usize,
                None => continue,
            };

            let obj_xy = match board.charuco_object_xy(corner_id as u32) {
                Some(pt) => Point2::new(pt.x, pt.y),
                None => continue,
            };

            let mut projections = Vec::new();

            for (marker_id, h) in transforms {
                // Проверяем, окружает ли этот маркер данный угол
                let surrounding = board.marker_surrounding_charuco_corners(*marker_id);
                if let Some(ids) = surrounding {
                    if ids.contains(&corner_id) {
                        let proj = h.apply(obj_xy);
                        projections.push(proj);
                    }
                }
            }

            if projections.len() >= min_markers {
                let sum: Point2<f32> = projections
                    .iter()
                    .fold(Point2::origin(), |acc, p| acc + p.coords);
                let avg = Point2::new(
                    sum.x / projections.len() as f32,
                    sum.y / projections.len() as f32,
                );
                corners.push((corner_id, avg));
            }
        }
    }
    corners
}
