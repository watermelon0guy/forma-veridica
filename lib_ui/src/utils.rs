use eframe::egui::{ColorImage, TextureHandle, TextureOptions};
use image::DynamicImage;

pub fn set_color_image_to_texture_handle(
    color_image: &DynamicImage,
    texture_handle: &mut TextureHandle,
) {
    texture_handle.set(
        dynamic_image_to_color_image(color_image),
        TextureOptions::default(),
    );
}

pub fn dynamic_image_to_color_image(img: &DynamicImage) -> ColorImage {
    let size = [img.width() as usize, img.height() as usize];
    let rgb = img.to_rgb8();
    ColorImage::from_rgb(size, rgb.as_raw())
}
