use calib_targets::charuco::CharucoDetectionResult;
use image::{DynamicImage, GenericImageView, Rgba};
use imageproc::{
    drawing::{draw_cross_mut, draw_polygon_mut},
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

pub fn draw_charuco_detection(
    img: &mut DynamicImage,
    result: &CharucoDetectionResult,
) -> DynamicImage {
    let mut img = img.to_rgba8();

    // Рисуем углы Charuco зелёным крестиком
    for corner in &result.detection.corners {
        let x = corner.position.x as i32;
        let y = corner.position.y as i32;
        draw_cross_mut(&mut img, Rgba([0, 255, 0, 255]), x, y);
    }

    // Рисуем маркеры ArUco синим прямоугольником
    for marker in &result.markers {
        if let Some(corners) = marker.corners_img {
            let points: Vec<_> = corners
                .iter()
                .map(|p| Point::new(p.x as i32, p.y as i32))
                .collect();
            draw_polygon_mut(&mut img, &points, Rgba([255, 0, 0, 255]));
        }
    }

    DynamicImage::ImageRgba8(img)
}

// DEPRECATED but may be nessecary in the future.
// pub fn split_video_into_quadrants(
//     path_to_video: &Path,
//     path_to_save: &Path,
//     file_name: &str,
// ) -> Result<Vec<PathBuf>, Error> {
//     let mut cap = VideoCapture::from_file(
//         path_to_video
//             .to_str()
//             .ok_or_else(|| Error::new(-1, "Неправильный путь к видео"))?,
//         opencv::videoio::CAP_ANY,
//     )?;
//     let mut frame = opencv::core::Mat::default();
//     let mut frame_index = 0;

//     let fourcc = opencv::videoio::VideoWriter::fourcc('m', 'p', '4', 'v')?;
//     let fps = cap.get(opencv::videoio::CAP_PROP_FPS)?;
//     let width = cap.get(opencv::videoio::CAP_PROP_FRAME_WIDTH)? as i32;
//     let height = cap.get(opencv::videoio::CAP_PROP_FRAME_HEIGHT)? as i32;

//     let quadrant_width = width / 2;
//     let quadrant_height = height / 2;

//     let mut writers = Vec::new();
//     let mut paths = Vec::new();
//     for i in 0..4 {
//         let output_path = path_to_save.join(format!("{}_{}.mp4", file_name, i));
//         let writer = opencv::videoio::VideoWriter::new(
//             output_path
//                 .to_str()
//                 .ok_or_else(|| Error::new(-1, "Неправильный путь для сохранения"))?,
//             fourcc,
//             fps,
//             opencv::core::Size::new(quadrant_width, quadrant_height),
//             true,
//         )?;
//         writers.push(writer);
//         paths.push(output_path);
//     }

//     while cap.read(&mut frame)? {
//         let quadrants = split_image_into_quadrants(&frame)?;
//         for (i, quadrant) in quadrants.into_iter().enumerate() {
//             writers[i].write(&quadrant)?;
//         }

//         frame_index += 1;
//         debug!("Обработан кадр {}", frame_index);
//     }

//     for mut writer in writers {
//         writer.release()?;
//     }

//     Ok(paths)
// }

// pub fn combine_quadrants(
//     img_1: &Mat,
//     img_2: &Mat,
//     img_3: &Mat,
//     img_4: &Mat,
// ) -> opencv::Result<Mat> {
//     // Соединяем верхние два изображения горизонтально
//     let mut top_row = Mat::default();
//     let mut tops = Vector::<Mat>::default();
//     tops.push(img_1.clone());
//     tops.push(img_2.clone());
//     hconcat(&tops, &mut top_row)?;

//     // Соединяем нижние два изображения горизонтально
//     let mut bottom_row = Mat::default();
//     let mut bottoms = Vector::<Mat>::default();
//     bottoms.push(img_3.clone());
//     bottoms.push(img_4.clone());
//     hconcat(&bottoms, &mut bottom_row)?;

//     // Соединяем верхний и нижний ряды вертикально
//     let mut combined = Mat::default();
//     let mut all = Vector::<Mat>::default();
//     all.push(top_row);
//     all.push(bottom_row);
//     vconcat(&all, &mut combined)?;

//     Ok(combined)
// }

// pub fn video_to_frames(path_to_video: &Path, parsed_image_folder_path: &Path) -> Result<(), Error> {
//     let mut cap = VideoCapture::from_file(
//         path_to_video
//             .to_str()
//             .ok_or_else(|| Error::new(-1, "Неправильный путь к видео"))?,
//         opencv::videoio::CAP_ANY,
//     )?;
//     let mut frame = opencv::core::Mat::default();
//     let mut frame_index = 0;

//     while cap.read(&mut frame)? {
//         let filename = format!(
//             "{}/{}.png",
//             parsed_image_folder_path
//                 .to_str()
//                 .ok_or_else(|| Error::new(-1, "Неправильный путь к папке для изображений"))?,
//             frame_index
//         );
//         opencv::imgcodecs::imwrite(&filename, &frame, &opencv::core::Vector::new())?;
//         frame_index += 1;
//         debug!("Обработано {}", frame_index);
//     }
//     Ok(())
// }

// pub fn vector_point2f_to_mat(points: &Vector<Point2f>) -> Result<Mat, Error> {
//     let num_points = points.len() as i32;
//     let mut mat = Mat::zeros(num_points, 2, opencv::core::CV_64F)?.to_mat()?;
//     for i in 0..num_points {
//         let p = points.get(i as usize)?;
//         *mat.at_2d_mut::<f64>(i, 0)? = p.x as f64;
//         *mat.at_2d_mut::<f64>(i, 1)? = p.y as f64;
//     }
//     Ok(mat)
// }

// pub fn open_video_captures(
//     caps: &mut Vec<VideoCapture>,
//     video_files: &Vec<Option<PathBuf>>,
// ) -> Result<(), Error> {
//     Ok(for video_file in video_files.iter() {
//         let cap = VideoCapture::from_file(
//             video_file
//                 .as_ref()
//                 .ok_or_else(|| Error::new(-1, "Неправильный путь к видео"))?
//                 .to_str()
//                 .ok_or_else(|| Error::new(-1, "Путь к видео не является валидной UTF-8 строкой"))?,
//             opencv::videoio::CAP_ANY,
//         )?;
//         caps.push(cap);
//     })
// }

// pub fn read_frames(caps: &mut Vec<VideoCapture>, frames: &mut Vec<Mat>) -> Result<(), Error> {
//     for (i, cap) in caps.iter_mut().enumerate() {
//         let mut frame = &mut frames[i];
//         cap.read(&mut frame)?;
//     }
//     Ok(())
// }

// pub fn get_video_frame_count(video_file: &PathBuf) -> Result<usize, Error> {
//     let cap = VideoCapture::from_file(&video_file.to_string_lossy(), CAP_ANY)?;
//     Ok(cap.get(CAP_PROP_FRAME_COUNT)? as usize)
// }
