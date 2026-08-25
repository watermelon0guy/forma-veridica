use std::path::Path;

use ffmpeg::codec::{Context as CodecContext, decoder::Video as VideoDecoder};
use ffmpeg::format::{context::Input, pixel::Pixel};
use ffmpeg::frame::Video as VideoFrame;
use ffmpeg::media::Type;
use ffmpeg::packet::Packet;
use ffmpeg::software::scaling::{Context as ScalingContext, flag::Flags as ScalingFlags};
use ffmpeg::util::error::EAGAIN;
use image::DynamicImage;

// Crate называется `ffmpeg-next`, в коде используем короткий алиас `ffmpeg`.
extern crate ffmpeg_next as ffmpeg;

/// Seek позиционирует голову чтения в пределах одной секунды от цели.
const SEEK_LEEWAY_MICROSECONDS: i64 = 1_000_000;

pub struct VideoPlayer {
    input: Input,
    video_stream_index: usize,
    decoder: VideoDecoder,
    scaler: ScalingContext,
    size: (u32, u32),
    current_frame: u64,
    pub current_time_in_seconds: f64,
    total_frames: u64,
    duration_seconds: f64,
    frame_rate: f32,
    color_image: DynamicImage,
}

impl VideoPlayer {
    pub fn new(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let input = ffmpeg::format::input(path)?;
        let stream = input
            .streams()
            .best(Type::Video)
            .ok_or("Не удалось найти видеопоток")?;
        let video_stream_index = stream.index();

        let decoder = CodecContext::from_parameters(stream.parameters())?
            .decoder()
            .video()?;

        let size = (decoder.width(), decoder.height());

        let scaler = ScalingContext::get(
            decoder.format(),
            size.0,
            size.1,
            Pixel::RGB24,
            size.0,
            size.1,
            ScalingFlags::AREA,
        )?;

        let frame_rate = {
            let rate = stream.rate();
            let rate = if rate.numerator() > 0 && rate.denominator() > 0 {
                rate
            } else {
                stream.avg_frame_rate()
            };
            if rate.denominator() > 0 {
                rate.numerator() as f32 / rate.denominator() as f32
            } else {
                0.0
            }
        };

        let total_frames = stream.frames().max(0) as u64;
        // У MKV/WebM и некоторых MP4 duration потока не заполнен — берём из контейнера.
        let duration_seconds = if stream.duration() > 0 {
            stream.duration() as f64 * f64::from(stream.time_base())
        } else {
            input.duration() as f64 / 1_000_000.0
        };

        let mut player = Self {
            input,
            video_stream_index,
            decoder,
            scaler,
            size,
            current_frame: 0,
            current_time_in_seconds: 0.0,
            total_frames,
            duration_seconds,
            frame_rate,
            color_image: DynamicImage::default(),
        };

        player.update_frame()?;
        Ok(player)
    }

    pub fn rewind_forward(&mut self, amount: u64) -> Result<(), Box<dyn std::error::Error>> {
        if self.total_frames == 0 {
            return Err("Не удалось определить количество кадров в видео".into());
        }
        if self.current_frame + amount < self.total_frames - 1 {
            self.current_frame += amount;
            self.seek_to_frame(self.current_frame)?;
            self.update_current_time_from_frame();
        } else {
            return Err("Не получается получить следующий кадр".into());
        }
        Ok(())
    }

    pub fn rewind_backward(&mut self, amount: u64) -> Result<(), Box<dyn std::error::Error>> {
        if amount < self.current_frame {
            self.current_frame -= amount;
            self.seek_to_frame(self.current_frame)?;
            self.update_current_time_from_frame();
        } else {
            return Err("Не получается получить предыдущий кадр".into());
        }
        Ok(())
    }

    pub fn seek_to_time(&mut self, seconds: f64) -> Result<(), Box<dyn std::error::Error>> {
        if !(0.0..self.duration_seconds).contains(&seconds) {
            return Err(format!("Время {seconds} находится за пределами видео").into());
        }

        self.seek((seconds * 1_000_000.0) as i64)?;
        self.update_frame()?;
        self.current_time_in_seconds = seconds;
        self.update_current_frame_from_time_in_seconds(seconds);
        Ok(())
    }

    pub fn seek_to_frame(&mut self, frame: u64) -> Result<(), Box<dyn std::error::Error>> {
        if frame < self.total_frames {
            let time = (frame as f64 / self.frame_rate as f64 * 1_000_000.0) as i64;
            self.seek(time)?;
            self.update_frame()?;
        }
        Ok(())
    }

    pub fn update_frame(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let rgb_frame = self.decode_next_frame()?;
        self.color_image = rgb_frame_to_dynamic_image(&rgb_frame)?;
        Ok(())
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    pub fn length_in_seconds(&self) -> f64 {
        self.duration_seconds
    }

    pub fn update_current_time_from_frame(&mut self) {
        self.current_time_in_seconds = self.current_frame as f64 / self.frame_rate as f64;
    }

    pub fn update_current_frame_from_time_in_seconds(&mut self, seconds: f64) {
        self.current_frame = (seconds * self.frame_rate as f64) as u64;
        if self.current_frame >= self.total_frames {
            self.current_frame = self.total_frames.saturating_sub(1);
        }
    }

    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }

    pub fn dynamic_image(&self) -> &DynamicImage {
        &self.color_image
    }

    pub fn _frame_rate(&self) -> f32 {
        self.frame_rate
    }

    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Перейти к timestamp в микросекундах и сбросить буферы декодера.
    fn seek(&mut self, timestamp_micros: i64) -> Result<(), Box<dyn std::error::Error>> {
        let range = timestamp_micros - SEEK_LEEWAY_MICROSECONDS
            ..timestamp_micros + SEEK_LEEWAY_MICROSECONDS;
        let result = self.input.seek(timestamp_micros, range);
        // avformat_seek_file может сбросить демуксер даже при ошибке,
        // поэтому буферы декодера сбрасываем всегда.
        self.decoder.flush();
        result?;
        Ok(())
    }

    /// Декодировать следующий кадр и переконвертировать его в RGB24.
    fn decode_next_frame(&mut self) -> Result<VideoFrame, Box<dyn std::error::Error>> {
        let mut decoded = VideoFrame::empty();
        loop {
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => {
                    let mut rgb_frame = VideoFrame::empty();
                    self.scaler.run(&decoded, &mut rgb_frame)?;
                    return Ok(rgb_frame);
                }
                // Декодеру нужны данные: читаем следующий пакет видеопотока.
                Err(ffmpeg::Error::Other { errno }) if errno == EAGAIN => {}
                Err(e) => return Err(format!("Проблема в декодировании кадра: {e}").into()),
            }

            match self.next_video_packet()? {
                Some(packet) => self.decoder.send_packet(&packet)?,
                // Видеопоток закончился — дренируем декодер от оставшихся кадров.
                None => self.decoder.send_eof()?,
            }
        }
    }

    /// Прочитать следующий пакет, принадлежащий нашему видеопотоку.
    fn next_video_packet(&mut self) -> Result<Option<Packet>, Box<dyn std::error::Error>> {
        for (stream, packet) in self.input.packets() {
            if stream.index() == self.video_stream_index {
                return Ok(Some(packet));
            }
        }
        Ok(None)
    }
}

/// Копирует данные RGB24-кадра в DynamicImage с учётом stride (linesize)
/// строки, который может быть больше `width * 3`.
fn rgb_frame_to_dynamic_image(
    frame: &VideoFrame,
) -> Result<DynamicImage, Box<dyn std::error::Error>> {
    let width = frame.width();
    let height = frame.height();
    let stride = frame.stride(0);
    let bytes_per_row = width as usize * 3;

    let mut pixels = Vec::with_capacity(bytes_per_row * height as usize);
    for row in 0..height as usize {
        let start = row * stride;
        pixels.extend_from_slice(&frame.data(0)[start..start + bytes_per_row]);
    }

    let rgb_image = image::RgbImage::from_raw(width, height, pixels)
        .ok_or("Проблема при создании RgbImage из raw данных")?;

    Ok(DynamicImage::ImageRgb8(rgb_image))
}
