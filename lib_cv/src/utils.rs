use calib_targets::charuco::CharucoDetectionResult;
use image::{DynamicImage, GenericImageView, Rgba};
use imageproc::{
    drawing::{draw_cross_mut, draw_hollow_circle_mut, draw_hollow_polygon_mut},
    point::Point,
};

pub fn split_image_into_quadrants(
    img: &DynamicImage,
) -> Result<Vec<DynamicImage>, Box<dyn std::error::Error>> {
    let roi_1 = img.view(0, 0, img.width() / 2, img.height() / 2);
    let roi_2 = img.view(img.width() / 2, 0, img.width() / 2, img.height() / 2);
    let roi_3 = img.view(0, img.height() / 2, img.width() / 2, img.height() / 2);
    let roi_4 = img.view(
        img.width() / 2,
        img.height() / 2,
        img.width() / 2,
        img.height() / 2,
    );
    Ok(vec![
        DynamicImage::ImageRgba8(roi_1.to_image()),
        DynamicImage::ImageRgba8(roi_2.to_image()),
        DynamicImage::ImageRgba8(roi_3.to_image()),
        DynamicImage::ImageRgba8(roi_4.to_image()),
    ])
}

pub fn draw_charuco_detection(img: &DynamicImage, result: &CharucoDetectionResult) -> DynamicImage {
    let mut img = img.to_rgba8();

    // Рисуем углы Charuco зелёным крестиком
    for corner in &result.corners {
        let x = corner.position.x as i32;
        let y = corner.position.y as i32;
        draw_hollow_circle_mut(&mut img, (x, y), 10, Rgba([255, 0, 0, 255]));
        draw_cross_mut(&mut img, Rgba([255, 0, 0, 255]), x, y);
    }

    // Рисуем маркеры ArUco синим прямоугольником
    for marker in &result.markers {
        if let Some(corners) = marker.corners_img {
            let points: Vec<_> = corners.iter().map(|p| Point::new(p.x, p.y)).collect();
            draw_hollow_polygon_mut(&mut img, &points, Rgba([255, 0, 0, 255]));
        }
    }

    DynamicImage::ImageRgba8(img)
}
