use std::path::Path;

use eframe::egui::{ColorImage, TextureHandle, TextureOptions};
use image::DynamicImage;
use video_rs::{Decoder, Time, frame::Frame};

pub(crate) struct VideoPlayer {
    decoder: Decoder,
    current_frame: u64,
    pub(crate) current_time_in_seconds: f64,
    total_frames: u64,
    duration: Time,
    frame_rate: f32,
    color_image: DynamicImage,
}

impl VideoPlayer {
    pub(crate) fn new(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let decoder = Decoder::new(path)?;

        let frame_rate = decoder.frame_rate();

        let total_frames = decoder.frames()?;
        let duration = decoder.duration()?;
        let mut player = Self {
            decoder,
            current_frame: 0,
            current_time_in_seconds: 0.0,
            duration,
            total_frames,
            frame_rate,
            color_image: DynamicImage::default(),
        };

        player.update_frame()?;
        Ok(player)
    }

    pub(crate) fn rewind_forward(&mut self, amount: u64) -> Result<(), Box<dyn std::error::Error>> {
        if self.current_frame + amount < self.total_frames - 1 {
            self.current_frame += amount;
            self.seek_to_frame(self.current_frame as u64)?;
            self.update_current_time_from_frame();
        } else {
            return Err("Не получается получить следующий кадр".into());
        }
        Ok(())
    }

    pub(crate) fn rewind_backward(
        &mut self,
        amount: u64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.current_frame - amount > 0 {
            self.current_frame -= amount;
            self.seek_to_frame(self.current_frame as u64)?;
            self.update_current_time_from_frame();
        } else {
            return Err("Не получается получить предыдущий кадр".into());
        }
        Ok(())
    }

    pub(crate) fn seek_to_time(&mut self, seconds: f64) -> Result<(), Box<dyn std::error::Error>> {
        let duration = self.duration;
        if seconds < duration.as_secs_f64() {
            self.decoder.seek((seconds * 1000.0) as i64)?;
            self.update_frame()?;
        }
        Ok(())
    }

    pub(crate) fn seek_to_frame(&mut self, frame: u64) -> Result<(), Box<dyn std::error::Error>> {
        if frame < self.total_frames {
            let time = (frame as f64 / self.frame_rate as f64 * 1000.0) as i64;
            self.decoder.seek(time)?;
            // self.decoder.seek_to_frame(frame as i64)?;
            self.update_frame()?;
        }
        Ok(())
    }

    pub(crate) fn update_frame(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let (_, frame) = match self.decoder.decode() {
            Ok(f) => f,
            Err(e) => return Err(format!("Проблема в декодировании кадра: {e}").into()),
        };

        self.color_image = frame_to_color_image(frame, self.size())?;
        Ok(())
    }

    pub(crate) fn size(&self) -> (u32, u32) {
        self.decoder.size()
    }

    pub(crate) fn length_in_seconds(&self) -> f64 {
        self.duration.as_secs_f64()
    }

    pub(crate) fn update_current_time_from_frame(&mut self) {
        self.current_time_in_seconds = self.current_frame as f64 / self.frame_rate as f64;
    }

    pub(crate) fn update_current_frame_from_time_in_seconds(&mut self, seconds: f64) {
        self.current_frame = (seconds as f64 * self.frame_rate as f64) as u64;
        if self.current_frame >= self.total_frames {
            self.current_frame = self.total_frames.saturating_sub(1);
        }
    }

    pub(crate) fn current_frame(&self) -> u64 {
        self.current_frame
    }

    pub(crate) fn dynamic_image(&self) -> &DynamicImage {
        &self.color_image
    }
}

fn frame_to_color_image(
    frame: Frame,
    size: (u32, u32),
) -> Result<DynamicImage, Box<dyn std::error::Error>> {
    let raw_frame = frame
        .as_slice()
        .ok_or("Проблема при конвертации кадра в байты для egui")?;

    let rgb_image = image::RgbImage::from_raw(size.0, size.1, raw_frame.to_vec())
        .ok_or("Проблема при создании RgbImage из raw данных")?;

    Ok(DynamicImage::ImageRgb8(rgb_image))
}

pub(crate) fn set_color_image_to_texture_handle(
    color_image: &DynamicImage,
    texture_handle: &mut TextureHandle,
) {
    texture_handle.set(
        dynamic_image_to_color_image(color_image),
        TextureOptions::default(),
    );
}

pub(crate) fn dynamic_image_to_color_image(img: &DynamicImage) -> ColorImage {
    let size = [img.width() as usize, img.height() as usize];
    let rgb = img.to_rgb8();
    ColorImage::from_rgb(size, rgb.as_raw())
}
