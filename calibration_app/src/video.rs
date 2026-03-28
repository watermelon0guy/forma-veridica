use std::path::Path;

use eframe::egui::{ColorImage, Context, TextureHandle, TextureOptions};
use video_rs::{Frame, Time, decode::Decoder};

pub(crate) struct VideoPlayer {
    decoder: Decoder,
    current_frame: u64,
    pub(crate) current_time_in_seconds: f64,
    total_frames: u64,
    duration: Time,
    frame_rate: f32,
    video_texture_handler: Option<TextureHandle>,
}

impl VideoPlayer {
    pub(crate) fn new(ctx: &Context, path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
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
            video_texture_handler: None,
        };

        player.update_frame(ctx)?;
        Ok(player)
    }

    pub(crate) fn rewind_forward(
        &mut self,
        ctx: &Context,
        amount: u64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.current_frame + amount < self.total_frames - 1 {
            self.current_frame += amount;
            self.seek_to_frame(ctx, self.current_frame as u64)?;
            self.update_current_time_from_frame();
        } else {
            return Err("Не получается получить следующий кадр".into());
        }
        Ok(())
    }

    pub(crate) fn rewind_backward(
        &mut self,
        ctx: &Context,
        amount: u64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.current_frame - amount > 0 {
            self.current_frame -= amount;
            self.seek_to_frame(ctx, self.current_frame as u64)?;
            self.update_current_time_from_frame();
        } else {
            return Err("Не получается получить предыдущий кадр".into());
        }
        Ok(())
    }

    pub(crate) fn seek_to_time(
        &mut self,
        ctx: &Context,
        seconds: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let duration = self.duration;
        if seconds < duration.as_secs_f64() {
            self.decoder.seek((seconds * 1000.0) as i64)?;
            self.update_frame(ctx)?;
        }
        Ok(())
    }

    pub(crate) fn seek_to_frame(
        &mut self,
        ctx: &Context,
        frame: u64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if frame < self.total_frames {
            let time = (frame as f64 / self.frame_rate as f64 * 1000.0) as i64;
            self.decoder.seek(time)?;
            // self.decoder.seek_to_frame(frame as i64)?;
            self.update_frame(ctx)?;
        }
        Ok(())
    }

    pub(crate) fn update_frame(&mut self, ctx: &Context) -> Result<(), Box<dyn std::error::Error>> {
        let (_, frame) = match self.decoder.decode() {
            Ok(f) => f,
            Err(e) => return Err(format!("Проблема в декодировании кадра: {e}").into()),
        };

        let color_image = frame_to_color_image(frame, self.size())?;

        match &mut self.video_texture_handler {
            Some(t) => t.set(color_image, TextureOptions::default()),
            None => {
                self.video_texture_handler =
                    Some(ctx.load_texture("video_frame", color_image, TextureOptions::default()))
            }
        }
        Ok(())
    }

    pub fn texture(&self) -> Option<&TextureHandle> {
        self.video_texture_handler.as_ref()
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
}

fn frame_to_color_image(
    frame: Frame,
    size: (u32, u32),
) -> Result<ColorImage, Box<dyn std::error::Error>> {
    let raw_frame = frame
        .as_slice()
        .ok_or("Проблема при конвертации кадра в байты для egui")?;

    Ok(ColorImage::from_rgb(
        [size.0 as usize, size.1 as usize],
        raw_frame,
    ))
}
