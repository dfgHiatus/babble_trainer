use std::{collections::HashMap, fs::File, io::Read};

use crate::{
    ImageData, ImageLabel, corruption_detector::FastCorruptionDetector, utils::decode_jpeg,
};
use anyhow::Result;
use byteorder::{LittleEndian, ReadBytesExt};
use log::{debug, info};

#[derive(Debug, Clone)]
struct FileData {
    routine_pitch: f32,
    routine_yaw: f32,
    routine_distance: f32,
    routine_convergence: f32,
    fov_adjust_distance: f32,
    left_eye_pitch: f32,
    left_eye_yaw: f32,
    right_eye_pitch: f32,
    right_eye_yaw: f32,
    routine_left_lid: f32,
    routine_right_lid: f32,
    routine_brow_raise: f32,
    routine_brow_angry: f32,
    routine_widen: f32,
    routine_squint: f32,
    routine_dilate: f32,
    timestamp: u64,
    video_timestamp_left: u64,
    video_timestamp_right: u64,
    routine_state: i32,
    jpeg_data_left_length: i32,
    jpeg_data_right_length: i32,
}

impl FileData {
    pub fn from_reader(rdr: &mut impl std::io::Read) -> Result<Self, std::io::Error> {
        Ok(FileData {
            routine_pitch: rdr.read_f32::<LittleEndian>()?,
            routine_yaw: rdr.read_f32::<LittleEndian>()?,
            routine_distance: rdr.read_f32::<LittleEndian>()?,
            routine_convergence: rdr.read_f32::<LittleEndian>()?,
            fov_adjust_distance: rdr.read_f32::<LittleEndian>()?,
            left_eye_pitch: rdr.read_f32::<LittleEndian>()?,
            left_eye_yaw: rdr.read_f32::<LittleEndian>()?,
            right_eye_pitch: rdr.read_f32::<LittleEndian>()?,
            right_eye_yaw: rdr.read_f32::<LittleEndian>()?,
            routine_left_lid: rdr.read_f32::<LittleEndian>()?,
            routine_right_lid: rdr.read_f32::<LittleEndian>()?,
            routine_brow_raise: rdr.read_f32::<LittleEndian>()?,
            routine_brow_angry: rdr.read_f32::<LittleEndian>()?,
            routine_widen: rdr.read_f32::<LittleEndian>()?,
            routine_squint: rdr.read_f32::<LittleEndian>()?,
            routine_dilate: rdr.read_f32::<LittleEndian>()?,
            timestamp: rdr.read_u64::<LittleEndian>()?,
            video_timestamp_left: rdr.read_u64::<LittleEndian>()?,
            video_timestamp_right: rdr.read_u64::<LittleEndian>()?,
            routine_state: rdr.read_i32::<LittleEndian>()?,
            jpeg_data_left_length: rdr.read_i32::<LittleEndian>()?,
            jpeg_data_right_length: rdr.read_i32::<LittleEndian>()?,
        })
    }

    fn to_image_label(&self) -> ImageLabel {
        ImageLabel {
            routine_pitch: self.routine_pitch,
            routine_yaw: self.routine_yaw,
            routine_distance: self.routine_distance,
            routine_convergence: self.routine_convergence,
            fov_adjust_distance: self.fov_adjust_distance,
            left_eye_pitch: self.left_eye_pitch,
            left_eye_yaw: self.left_eye_yaw,
            right_eye_pitch: self.right_eye_pitch,
            right_eye_yaw: self.right_eye_yaw,
            routine_left_lid: self.routine_left_lid,
            routine_right_lid: self.routine_right_lid,
            routine_brow_raise: self.routine_brow_raise,
            routine_brow_angry: self.routine_brow_angry,
            routine_widen: self.routine_widen,
            routine_squint: self.routine_squint,
            routine_dilate: self.routine_dilate,
            routine_state: self.routine_state,
        }
    }
}

pub struct FileReader {
    pub left_eye_frames: HashMap<u64, ImageData>,
    pub right_eye_frames: HashMap<u64, ImageData>,
    pub label_frames: HashMap<u64, ImageLabel>,

    pub raw_frames: u64,
    pub skipped_frames: u64,
    pub total_bad_frames: u64,

    detector: FastCorruptionDetector,
}

impl FileReader {
    pub fn new() -> Self {
        Self {
            left_eye_frames: HashMap::new(),
            right_eye_frames: HashMap::new(),
            label_frames: HashMap::new(),
            raw_frames: 0,
            skipped_frames: 0,
            total_bad_frames: 0,
            detector: FastCorruptionDetector::new(0.022669, true, 100),
        }
    }

    pub fn read_capture_file(
        &mut self,
        filename: &str,
        do_glitch_detection: bool,
        equalize_histogram: bool,
        exclude_after: u64,
        exclude_before: u64,
    ) -> Result<()> {
        let mut file = File::open(filename)?;

        // Read the raw data from file
        // We don't read the file just to calculate the frame count, we will read it in one pass and process frames as we go

        let start_time = std::time::Instant::now();
        loop {
            let data = match FileData::from_reader(&mut file) {
                Ok(d) => d,
                Err(e) => {
                    if e.kind() == std::io::ErrorKind::UnexpectedEof {
                        info!("Reached end of file");
                        break;
                    } else {
                        return Err(e.into());
                    }
                }
            };

            if data.jpeg_data_left_length < 0 || data.jpeg_data_right_length < 0 {
                info!(
                    "Invalid JPEG data lengths: left={}, right={}",
                    data.jpeg_data_left_length, data.jpeg_data_right_length
                );
                break;
            }

            if data.jpeg_data_left_length > 10 * 1024 * 1024
                || data.jpeg_data_right_length > 10 * 1024 * 1024
            {
                info!(
                    "JPEG data lengths too large: left={}, right={}",
                    data.jpeg_data_left_length, data.jpeg_data_right_length
                );
                break;
            }

            // Read the image data
            let mut image_left_data = vec![0u8; data.jpeg_data_left_length as usize];
            file.read_exact(&mut image_left_data)?;

            let mut image_right_data = vec![0u8; data.jpeg_data_right_length as usize];
            file.read_exact(&mut image_right_data)?;

            self.raw_frames += 1;

            let left_image = decode_jpeg(&image_left_data, equalize_histogram)?;
            let right_image = decode_jpeg(&image_right_data, equalize_histogram)?;

            let bad = if do_glitch_detection {
                let detection = self.detector.process_frame_pair(&left_image, &right_image);

                if detection.left_corrupted || detection.right_corrupted {
                    self.total_bad_frames += 1;
                    debug!(
                        "Detected bad frame at timestamp {}: bad_left={}, bad_right={}",
                        data.timestamp, detection.left_corrupted, detection.right_corrupted
                    );
                }

                detection.left_corrupted || detection.right_corrupted
            } else {
                false
            };

            if self.skipped_frames < exclude_before {
                self.skipped_frames += 1;
                continue;
            }
            if exclude_after != 0 && self.raw_frames > exclude_after {
                break;
            }

            if !bad {
                self.left_eye_frames
                    .insert(data.video_timestamp_left, left_image);
                self.right_eye_frames
                    .insert(data.video_timestamp_right, right_image);
                self.label_frames
                    .insert(data.timestamp, data.to_image_label());
            }
        }

        info!(
            "Finished reading file in {:.2?} ({} per second)",
            std::time::Instant::now() - start_time,
            self.raw_frames as f32
                / std::time::Instant::now()
                    .duration_since(start_time)
                    .as_secs_f32()
                    .max(1.0)
        );

        info!("Detected {} raw frames", self.raw_frames);
        info!("Unique left eye frames: {}", self.left_eye_frames.len());
        info!("Unique right eye frames: {}", self.right_eye_frames.len());
        info!("Unique label frames: {}", self.label_frames.len());
        info!(
            "Excluded {} bad frames (bsb glitch detector)",
            self.total_bad_frames
        );

        Ok(())
    }
}
