use burn::{
    Tensor,
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, DatasetIterator},
    },
    prelude::Backend,
    tensor::backend::AutodiffBackend,
};

use crate::{frame_correlator::AlignedFrame, loader::ImageLabel};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EyeSide {
    Left,
    Right,
}

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
    pub fn from_frames<B: Backend>(frames: &[AlignedFrame<B>]) -> Self {
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
    pub side: EyeSide,
    pub dataset_info: DatasetInfo,
}

#[derive(Clone, Debug)]
pub struct EyeDataBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 2>,
}

#[derive(Clone, Debug)]
pub struct WindowedFrame<B: Backend> {
    pub label: ImageLabel,
    pub left_eye: Tensor<B, 3>,
    pub right_eye: Tensor<B, 3>,
    pub timestamp: u64,
}

impl<B: Backend> WindowedFrame<B> {
    pub fn from_aligned_frames(aligned: &[AlignedFrame<B>]) -> Vec<WindowedFrame<B>> {
        aligned
            .windows(4)
            .map(|w| {
                // Concat the 4 frames in the window along the channel dimension
                let left_eye = Tensor::cat(
                    w.iter()
                        .map(|f| {
                            let [height, width] = f.left_eye.dims();
                            f.left_eye.clone().reshape([1, height, width])
                        })
                        .collect::<Vec<_>>(),
                    0,
                );
                let right_eye = Tensor::cat(
                    w.iter()
                        .map(|f| {
                            let [height, width] = f.right_eye.dims();
                            f.right_eye.clone().reshape([1, height, width])
                        })
                        .collect::<Vec<_>>(),
                    0,
                );

                WindowedFrame {
                    label: w.last().unwrap().label.clone(),
                    left_eye,
                    right_eye,
                    timestamp: w.last().unwrap().timestamp,
                }
            })
            .collect()
    }
}

impl<B: Backend> WindowedFrame<B> {
    pub fn to_backend<C: Backend>(&self, device: &C::Device) -> WindowedFrame<C> {
        WindowedFrame {
            label: self.label.clone(),
            left_eye: Tensor::from_data(self.left_eye.to_data(), device),
            right_eye: Tensor::from_data(self.right_eye.to_data(), device),
            timestamp: self.timestamp,
        }
    }
}

fn apply_spatial_transform<B: Backend>(
    image: &Tensor<B, 3>,
    max_shift: i32,
    max_rotation: f32,
    max_scale: f32,
) -> Tensor<B, 3> {
    // Placeholder for actual spatial transformation logic
    image.clone()
}

fn apply_intensity_transformation<B: Backend>(
    image: &Tensor<B, 3>,
    brightness_range: f32,
    contrast_range: f32,
) -> Tensor<B, 3> {
    // Placeholder for actual intensity transformation logic
    image.clone()
}

fn apply_blur<B: Backend>(image: &Tensor<B, 3>, max_kernel_size: i32) -> Tensor<B, 3> {
    // Placeholder for actual blur transformation logic
    image.clone()
}

impl<B: Backend> Batcher<B, WindowedFrame<B>, EyeDataBatch<B>> for EyeDataBatcher {
    fn batch(&self, items: Vec<WindowedFrame<B>>, device: &B::Device) -> EyeDataBatch<B> {
        let images = items
            .iter()
            .map(|item| {
                let mut image = match self.side {
                    EyeSide::Left => item.left_eye.clone(),
                    EyeSide::Right => item.right_eye.clone(),
                };

                if self.training {
                    if rand::random::<f32>() < 0.2 {
                        image = apply_spatial_transform(&image, 24, 10.0, 0.1);
                    }

                    if rand::random::<f32>() < 0.3 {
                        image = apply_intensity_transformation(&image, 0.1, 0.5);
                    }

                    if rand::random::<f32>() < 0.2 {
                        image = apply_blur(&image, 5);
                    }

                    image.to_device(device)
                } else {
                    image.to_device(device)
                }
            })
            .map(|tensor| {
                let [channels, height, width] = tensor.dims();

                tensor.reshape([1, channels, height, width])
            })
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                let left_lid_closed = item.label.routine_left_lid < 0.5;
                let right_lid_closed = item.label.routine_right_lid < 0.5;

                match self.side {
                    EyeSide::Left => {
                        if left_lid_closed {
                            return Tensor::<B, 1>::from_data([0.5, 0.5, 1.0], device);
                        }
                    }
                    EyeSide::Right => {
                        if right_lid_closed {
                            return Tensor::<B, 1>::from_data([0.5, 0.5, 1.0], device);
                        }
                    }
                }

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

                match self.side {
                    EyeSide::Left => {
                        Tensor::<B, 1>::from_data([norm_pitch_l, norm_yaw_l, 0.0], device)
                    }
                    EyeSide::Right => {
                        Tensor::<B, 1>::from_data([norm_pitch_r, norm_yaw_r, 0.0], device)
                    }
                }
            })
            .map(|tensor| tensor.reshape([1, 3]))
            .collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        EyeDataBatch { images, targets }
    }
}
