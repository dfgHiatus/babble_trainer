use burn::{
    Tensor,
    data::{dataloader::batcher::Batcher, dataset::transform::Mapper},
    prelude::Backend,
};

use crate::{ImageData, frame_correlator::AlignedFrame, loader::ImageLabel};

#[derive(Clone, Debug)]
pub struct DatasetInfo {
    pub pitch_min_l: f32,
    pub pitch_max_l: f32,
    pub yaw_min_l: f32,
    pub yaw_max_l: f32,

    pub pitch_min_r: f32,
    pub pitch_max_r: f32,
    pub yaw_min_r: f32,
    pub yaw_max_r: f32,

    pub label_count: usize,
}

impl DatasetInfo {
    pub fn from_frames(frames: &[AlignedFrame]) -> Self {
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
}

#[derive(Clone)]
pub struct EyeDataBatcher {
    pub training: bool,
    pub dataset_info: DatasetInfo,
}

#[derive(Clone, Debug)]
pub struct EyeDataBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 2>,
}

#[derive(Clone, Debug)]
pub struct WindowedFrame {
    pub label: ImageLabel,
    pub left_eye: [ImageData; 4],
    pub right_eye: [ImageData; 4],
    pub timestamp: u64,
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

fn apply_spatial_transform(
    image: [ImageData; 8],
    max_shift: i32,
    max_rotation: f32,
    max_scale: f32,
) -> [ImageData; 8] {
    // Placeholder for actual spatial transformation logic
    image.clone()
}

fn apply_intensity_transformation(
    image: [ImageData; 8],
    brightness_range: f32,
    contrast_range: f32,
) -> [ImageData; 8] {
    // Placeholder for actual intensity transformation logic
    image.clone()
}

fn apply_blur(image: [ImageData; 8], max_kernel_size: i32) -> [ImageData; 8] {
    // Placeholder for actual blur transformation logic
    image.clone()
}

impl<B: Backend> Batcher<B, WindowedFrame, EyeDataBatch<B>> for EyeDataBatcher {
    fn batch(&self, mut items: Vec<WindowedFrame>, device: &B::Device) -> EyeDataBatch<B> {
        // Pull these out first so that we can pull the image buffers out without copies
        let targets = items
            .iter()
            .map(|item| {
                let left_lid_closed = item.label.routine_left_lid < 0.5;
                let right_lid_closed = item.label.routine_right_lid < 0.5;

                // ...Python trainer has this code but I'm prettyyyy sure it just gets overwritten 10 lines down
                // let norm_pitch_r = (item.label.right_eye_pitch - self.dataset_info.pitch_min_r)
                //     / (self.dataset_info.pitch_max_r - self.dataset_info.pitch_min_r);
                // let norm_yaw_r = (item.label.right_eye_yaw - self.dataset_info.yaw_min_r)
                //     / (self.dataset_info.yaw_max_r - self.dataset_info.yaw_min_r);

                // let norm_pitch_l = (item.label.left_eye_pitch - self.dataset_info.pitch_min_l)
                //     / (self.dataset_info.pitch_max_l - self.dataset_info.pitch_min_l);
                // let norm_yaw_l = (item.label.left_eye_yaw - self.dataset_info.yaw_min_l)
                //     / (self.dataset_info.yaw_max_l - self.dataset_info.yaw_min_l);

                let norm_pitch_r = ((item.label.right_eye_pitch + 45.0) / 90.0).clamp(0.0, 1.0);
                let norm_yaw_r = ((item.label.right_eye_yaw + 45.0) / 90.0).clamp(0.0, 1.0);
                let norm_pitch_l = ((item.label.left_eye_pitch + 45.0) / 90.0).clamp(0.0, 1.0);
                let norm_yaw_l = ((item.label.left_eye_yaw + 45.0) / 90.0).clamp(0.0, 1.0);

                let left_values = if left_lid_closed {
                    [0.5, 0.5, 1.0]
                } else {
                    [norm_pitch_l, norm_yaw_l, 0.0]
                };

                let right_values = if right_lid_closed {
                    [0.5, 0.5, 1.0]
                } else {
                    [norm_pitch_r, norm_yaw_r, 0.0]
                };

                return Tensor::<B, 1>::from_data(
                    [
                        left_values[0],
                        left_values[1],
                        left_values[2],
                        right_values[0],
                        right_values[1],
                        right_values[2],
                    ],
                    device,
                );
            })
            .map(|tensor| tensor.reshape([1, -1]))
            .collect();

        let images = items
            .drain(..)
            .map(|item| {
                let [l_0, l_1, l_2, l_3] = item.left_eye;
                let [r_0, r_1, r_2, r_3] = item.right_eye;

                // Intersperse Left and Right channels, with Left channels in even indices and Right channels in odd indices
                let mut image = [l_0, r_0, l_1, r_1, l_2, r_2, l_3, r_3];

                if self.training {
                    if rand::random::<f32>() < 0.2 {
                        image = apply_spatial_transform(image, 24, 10.0, 0.1);
                    }

                    if rand::random::<f32>() < 0.3 {
                        image = apply_intensity_transformation(image, 0.1, 0.5);
                    }

                    if rand::random::<f32>() < 0.2 {
                        image = apply_blur(image, 5);
                    }
                }

                Tensor::stack(
                    image
                        .iter()
                        .map(|img| {
                            let data = img.as_raw();
                            let height = img.height() as usize;
                            let width = img.width() as usize;

                            Tensor::<B, 1>::from_data(&data[..], device).reshape([width, height])
                        })
                        .collect(),
                    0,
                )
            })
            .map(|tensor| {
                let [channels, height, width] = tensor.dims();

                tensor.reshape([1, channels, height, width])
            })
            .collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        EyeDataBatch { images, targets }
    }
}
