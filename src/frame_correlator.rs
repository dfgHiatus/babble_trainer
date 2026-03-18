use crate::{
    ImageData, ImageLabel,
    batcher::{DatasetInfo, WindowedFrame},
};
use burn::data::dataset::transform::Mapper;
use log::info;
use pearson::correlate;

const WIN_SIZE_MUL: usize = 10;

fn find_pattern_based_offset(label_timestamps: &[u64], eye_timestamps: &[u64]) -> i64 {
    if label_timestamps.len() < 10 || eye_timestamps.len() < 10 {
        return 0; // Python version tried global time offset here, but for simplicity we will just return 0 if we don't have enough data to analyze
    }

    let label_intervals: Vec<f64> = label_timestamps
        .windows(2)
        .map(|w| (w[1] - w[0]) as f64)
        .collect();
    let eye_intervals: Vec<f64> = eye_timestamps
        .windows(2)
        .map(|w| (w[1] - w[0]) as f64)
        .collect();

    let mut best_offset = 0;
    let mut best_correlation = -1.0;

    // ...I'm pretty sure these are always the same length because we serialize them in the same object... but uhhh, python does this and it takes 0 seconds if it is soooo
    for start_pos in 0..1.max(eye_intervals.len() - label_intervals.len()) {
        let end_pos = start_pos + label_intervals.len();
        if end_pos > eye_intervals.len() {
            break;
        }

        let eye_segment = &eye_intervals[start_pos..end_pos];

        if eye_segment.len() == label_intervals.len() {
            let correlation = correlate(&label_intervals, eye_segment);
            if correlation > best_correlation {
                best_correlation = correlation;

                let label_start_time = label_timestamps[0];
                let eye_start_time = eye_timestamps[start_pos];
                best_offset = eye_start_time as i64 - label_start_time as i64;
            }
        }
    }

    info!(
        "Pattern correlation: {:.2} ({:.4})",
        best_offset, best_correlation
    );
    best_offset
}

/// Finds the best unused neighbor index within a specified window size around a target timestamp
fn find_best_unused_neighbor(
    timestamps: &[u64],
    idx: usize,
    target: i64,
    used_indices: &std::collections::HashSet<usize>,
    window_size: usize,
) -> (Option<usize>, u64) {
    let mut best_idx = None;
    let mut best_dev = u64::MAX;

    let window_size = window_size * WIN_SIZE_MUL;

    let start = idx.saturating_sub(window_size);
    let end = (idx + window_size).min(timestamps.len());

    for i in start..end {
        if used_indices.contains(&i) {
            continue;
        }

        let dev = (timestamps[i] as i64 - target).unsigned_abs();
        if dev < best_dev {
            best_dev = dev;
            best_idx = Some(i);
        }
    }

    (best_idx, best_dev)
}

#[derive(Debug, Clone)]
pub struct AlignedFrame {
    pub label: ImageLabel,
    pub left_eye: ImageData,
    pub right_eye: ImageData,
    pub timestamp: u64,
}

pub fn align_frames(
    left_frames: Vec<(u64, ImageData)>,
    right_frames: Vec<(u64, ImageData)>,
    label_frames: Vec<(u64, ImageLabel)>,
) -> Vec<AlignedFrame> {
    let left_timestamps: Vec<u64> = left_frames.iter().map(|(ts, _)| *ts).collect();
    let right_timestamps: Vec<u64> = right_frames.iter().map(|(ts, _)| *ts).collect();
    let label_timestamps: Vec<u64> = label_frames.iter().map(|(ts, _)| *ts).collect();

    info!("Advanced Phase 1: Cross-correlation offset detection...");

    fn estimate_frame_intervals(timestamps: &[u64]) -> Vec<u64> {
        timestamps.windows(2).map(|w| w[1] - w[0]).collect()
    }

    let label_intervals = estimate_frame_intervals(&label_timestamps[..(3000.min(label_timestamps.len()))]);
    let left_intervals = estimate_frame_intervals(&left_timestamps[..(3000.min(left_timestamps.len()))]);
    // let right_intervals = estimate_frame_intervals(&right_timestamps[..(3000.min(right_timestamps.len()))]);

    if !label_intervals.is_empty() && !left_intervals.is_empty() {
        let avg_label_fps =
            1000.0 / (label_intervals.iter().sum::<u64>() as f64 / label_intervals.len() as f64);
        let avg_left_fps =
            1000.0 / (left_intervals.iter().sum::<u64>() as f64 / left_intervals.len() as f64);
        info!(
            "Estimated frame rates: Label={:.1}fps, Left={:.1}fps",
            avg_label_fps, avg_left_fps
        );
    }

    let left_offset = find_pattern_based_offset(&label_timestamps, &left_timestamps);
    let right_offset = find_pattern_based_offset(&label_timestamps, &right_timestamps);

    info!(
        "Pattern-based offsets: left={}ms, right={}ms",
        left_offset, right_offset
    );

    struct PotentialMatch {
        label_data: ImageLabel,
        left_img: ImageData,
        right_img: ImageData,
        label_ts: u64,
        quality: u64,
        left_idx: usize,
        right_idx: usize,
    }

    let mut potential_matches = label_frames
        .iter()
        .filter_map(|(label_ts, label_data)| {
            let adjusted_left_target = *label_ts as i64 + left_offset;
            let adjusted_right_target = *label_ts as i64 + right_offset;

            let left_idx = left_timestamps
                .binary_search(&adjusted_left_target.unsigned_abs())
                .unwrap_or_else(|x| x);
            let (best_left_idx, _) = find_best_unused_neighbor(
                &left_timestamps,
                left_idx,
                adjusted_left_target,
                &std::collections::HashSet::new(),
                5,
            );

            let right_idx = right_timestamps
                .binary_search(&adjusted_right_target.unsigned_abs())
                .unwrap_or_else(|x| x);
            let (best_right_idx, _) = find_best_unused_neighbor(
                &right_timestamps,
                right_idx,
                adjusted_right_target,
                &std::collections::HashSet::new(),
                5,
            );

            if let (Some(left_idx), Some(right_idx)) = (best_left_idx, best_right_idx) {
                let actual_left_dev =
                    (left_timestamps[left_idx] as i64 - *label_ts as i64).unsigned_abs();
                let actual_right_dev =
                    (right_timestamps[right_idx] as i64 - *label_ts as i64).unsigned_abs();

                Some(PotentialMatch {
                    label_data: label_data.clone(),
                    left_img: left_frames[left_idx].1.clone(),
                    right_img: right_frames[right_idx].1.clone(),
                    label_ts: *label_ts,
                    quality: actual_left_dev + actual_right_dev,
                    left_idx,
                    right_idx,
                })
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    info!(
        "Found {} potential matches before conflict resolution, but dropped {}",
        potential_matches.len(),
        potential_matches.len() - potential_matches.len()
    );

    potential_matches.sort_by_key(|x| x.quality); // sort by quality

    let mut final_matches = Vec::new();
    let mut used_left = std::collections::HashSet::new();
    let mut used_right = std::collections::HashSet::new();

    let potential_matches_len = potential_matches.len();
    for match_ in potential_matches {
        if !used_left.contains(&match_.left_idx) && !used_right.contains(&match_.right_idx) {
            used_left.insert(match_.left_idx);
            used_right.insert(match_.right_idx);
            final_matches.push(AlignedFrame {
                label: match_.label_data,
                left_eye: match_.left_img,
                right_eye: match_.right_img,
                timestamp: match_.label_ts,
            });
        }
    }

    info!(
        "Dropped {} potential matches due to conflicts",
        potential_matches_len - final_matches.len()
    );

    final_matches.sort_by_key(|x| x.timestamp); // sort by label timestamp

    final_matches
}

pub struct ImageToTensor;

impl Mapper<Vec<AlignedFrame>, WindowedFrame> for ImageToTensor {
    /// Converts a windowed vector of frames into a
    fn map(&self, item: &Vec<AlignedFrame>) -> WindowedFrame {
        WindowedFrame {
            label: item.last().unwrap().label.clone(),
            left_eye: [
                item[0].left_eye.clone(),
                item[1].left_eye.clone(),
                item[2].left_eye.clone(),
                item[3].left_eye.clone(),
            ],
            right_eye: [
                item[0].right_eye.clone(),
                item[1].right_eye.clone(),
                item[2].right_eye.clone(),
                item[3].right_eye.clone(),
            ],
            timestamp: item.last().unwrap().timestamp,
        }
    }
}

pub fn create_dataset(frames: &[AlignedFrame]) -> DatasetInfo {
    let mut pitch_min_l = f32::INFINITY;
    let mut pitch_max_l = f32::NEG_INFINITY;
    let mut yaw_min_l = f32::INFINITY;
    let mut yaw_max_l = f32::NEG_INFINITY;

    let mut pitch_min_r = f32::INFINITY;
    let mut pitch_max_r = f32::NEG_INFINITY;
    let mut yaw_min_r = f32::INFINITY;
    let mut yaw_max_r = f32::NEG_INFINITY;

    for frame in frames {
        let label = &frame.label;

        if label.left_eye_pitch < pitch_min_l {
            pitch_min_l = label.left_eye_pitch;
        }
        if label.left_eye_pitch > pitch_max_l {
            pitch_max_l = label.left_eye_pitch;
        }
        if label.left_eye_yaw < yaw_min_l {
            yaw_min_l = label.left_eye_yaw;
        }
        if label.left_eye_yaw > yaw_max_l {
            yaw_max_l = label.left_eye_yaw;
        }

        if label.right_eye_pitch < pitch_min_r {
            pitch_min_r = label.right_eye_pitch;
        }
        if label.right_eye_pitch > pitch_max_r {
            pitch_max_r = label.right_eye_pitch;
        }
        if label.right_eye_yaw < yaw_min_r {
            yaw_min_r = label.right_eye_yaw;
        }
        if label.right_eye_yaw > yaw_max_r {
            yaw_max_r = label.right_eye_yaw;
        }
    }

    DatasetInfo {
        pitch_min_l,
        pitch_max_l,
        yaw_min_l,
        yaw_max_l,
        pitch_min_r,
        pitch_max_r,
        yaw_min_r,
        yaw_max_r,
        label_count: frames.len(),
    }
}
