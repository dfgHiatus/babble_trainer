use std::collections::VecDeque;

use burn::{
    prelude::ToElement,
    tensor::{Tensor, backend::Backend},
};
use log::error;

use crate::ImageData;

/// Takes in a grayscale image tensor with values 0-1 and calculates the consistency (standard deviation of row differences)
pub fn calculate_row_pattern_consistency_tensor<B: Backend>(image: &Tensor<B, 2>) -> Tensor<B, 1> {
    // Calculate row means
    let row_means = image.clone().mean_dim(1);

    // Calculate consistency (standard deviation of row differences)
    if row_means.dims().len() > 1 {
        let n = row_means.dims()[0];

        if n < 2 {
            error!(
                "Not enough rows to calculate consistency, Dims: {:?}",
                row_means.dims()
            );
            return Tensor::from([0.0]);
        }

        let diffs = row_means.clone().narrow(0, 1, n - 1) - row_means.narrow(0, 0, n - 1);

        diffs.var(0).flatten(0, 1)
    } else {
        Tensor::from([0.0])
    }
}

/// Takes in a grayscale image tensor with values 0-1 and calculates the consistency (standard deviation of row differences)
pub fn calculate_row_pattern_consistency_image(image: &ImageData) -> f32 {
    // Calculate row means
    let row_means = image
        .rows()
        .map(|row| {
            let len = row.len() as f32;
            let sum: u32 = row.map(|pixel| pixel[0] as u32).sum();
            sum as f32 / len / 255.0 // Normalize to 0-1
        })
        .collect::<Vec<f32>>();

    // Calculate consistency (standard deviation of row differences)
    if row_means.len() > 1 {
        let n = row_means.len();

        if n < 2 {
            error!(
                "Not enough rows to calculate consistency, Dims: {:?}",
                row_means.len()
            );
            return 0.0;
        }

        let diffs = row_means
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect::<Vec<f32>>();

        let diff_mean = diffs.iter().sum::<f32>() / (diffs.len() as f32);

        diffs.iter().map(|d| (d - diff_mean).powi(2)).sum::<f32>() / (diffs.len() as f32)
    } else {
        0.0
    }
}

/// Pulling apart the deque so often probably isn't good for performance
/// so this could probably be optimized with some kind of rolling statistic?
fn median(values: &mut [f32]) -> f32 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

#[derive(Debug, Clone, Default)]
pub struct CorruptionDetectionResult {
    pub left_corrupted: bool,
    pub right_corrupted: bool,
    pub left_value: f32,
    pub right_value: f32,
    pub left_threshold: f32,
    pub right_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct CorruptionDetectionStats {
    pub total_frames: usize,
    pub corrupted_left: usize,
    pub corrupted_right: usize,
    pub corruption_rate_left: f32,
    pub corruption_rate_right: f32,
    pub base_threshold: f32,
    pub current_threshold: f32,
    pub threshold_updates: usize,
    pub adaptive_enabled: bool,
}

pub struct FastCorruptionDetector {
    pub base_threshold: f32,
    pub current_threshold: f32,
    pub use_adaptive: bool,

    recent_values: VecDeque<f32>,

    pub total_frames: usize,
    pub detected_corrupted_left: usize,
    pub detected_corrupted_right: usize,
    pub threshold_updates: usize,
}

impl FastCorruptionDetector {
    pub fn new(threshold: f32, use_adaptive: bool, adaption_window: usize) -> Self {
        Self {
            base_threshold: threshold,
            current_threshold: threshold,
            use_adaptive,
            recent_values: VecDeque::with_capacity(adaption_window),
            total_frames: 0,
            detected_corrupted_left: 0,
            detected_corrupted_right: 0,
            threshold_updates: 0,
        }
    }

    pub fn update_adaptive_threshold(&mut self, value: f32) {
        if !self.use_adaptive {
            return;
        }

        // Add current value to history
        self.recent_values.push_back(value);

        // Need enough history to compute adaptive threshold
        if self.recent_values.len() < 20 {
            return;
        }

        // Use robust statistics (median + k*MAD) to set threshold
        // Assumes most frames are clean, so this gives threshold for outliers
        // We don't use a burn vector here, hoping that SIMD handles this one, plus only 100 or so values
        let mut values: Vec<f32> = self.recent_values.iter().cloned().collect();
        let median_val = median(&mut values);
        let mut mad = values
            .iter()
            .map(|v| (v - median_val).abs())
            .collect::<Vec<f32>>();
        let mad = median(&mut mad);

        // Set threshold as median + 3*MAD (robust outlier detection)
        let adaptive_threshold = median_val + 3.0 * mad;

        // Don't let adaptive threshold go too far from base threshold
        let min_threshold = self.base_threshold * 0.5;
        let max_threshold = self.base_threshold * 3.0;

        self.current_threshold = adaptive_threshold.clamp(min_threshold, max_threshold);
        self.threshold_updates += 1;
    }

    /// Determine if frame is corrupted based on row pattern consistency.
    /// Returns (is_corrupted, metric_value, threshold_used)
    pub fn is_corrupted_tensor(&mut self, frame: &Tensor<impl Backend, 2>) -> (bool, f32, f32) {
        let metric_value = calculate_row_pattern_consistency_tensor(&frame)
            .into_scalar()
            .to_f32();

        self.total_frames += 1;

        // Update adaptive threshold
        self.update_adaptive_threshold(metric_value);

        // Check if corrupted
        let is_corrupted = metric_value > self.current_threshold;

        (is_corrupted, metric_value, self.current_threshold)
    }

    /// Determine if frame is corrupted based on row pattern consistency.
    /// Returns (is_corrupted, metric_value, threshold_used)
    pub fn is_corrupted(&mut self, frame: &ImageData) -> (bool, f32, f32) {
        let metric_value = calculate_row_pattern_consistency_image(frame);

        self.total_frames += 1;

        // Update adaptive threshold
        self.update_adaptive_threshold(metric_value);

        // Check if corrupted
        let is_corrupted = metric_value > self.current_threshold;

        (is_corrupted, metric_value, self.current_threshold)
    }

    pub fn process_frame_pair(
        &mut self,
        left_frame: &ImageData,
        right_frame: &ImageData,
    ) -> CorruptionDetectionResult {
        self.total_frames += 1;

        let (left_corrupted, left_value, left_threshold) = self.is_corrupted(left_frame);
        let (right_corrupted, right_value, right_threshold) = self.is_corrupted(right_frame);

        if left_corrupted {
            self.detected_corrupted_left += 1;
        }
        if right_corrupted {
            self.detected_corrupted_right += 1;
        }

        CorruptionDetectionResult {
            left_corrupted,
            right_corrupted,
            left_value,
            right_value,
            left_threshold,
            right_threshold,
        }
    }

    pub fn get_stats(&self) -> CorruptionDetectionStats {
        CorruptionDetectionStats {
            total_frames: self.total_frames,
            corrupted_left: self.detected_corrupted_left,
            corrupted_right: self.detected_corrupted_right,
            corruption_rate_left: self.detected_corrupted_left as f32
                / 1f32.max(self.total_frames as f32),
            corruption_rate_right: self.detected_corrupted_right as f32
                / 1f32.max(self.total_frames as f32),
            base_threshold: self.base_threshold,
            current_threshold: self.current_threshold,
            threshold_updates: self.threshold_updates,
            adaptive_enabled: self.use_adaptive,
        }
    }
}
