use burn::{
    nn::{
        Linear, LinearConfig, PaddingConfig2d, Relu, Sigmoid,
        conv::{Conv2d, Conv2dConfig},
        loss::{CrossEntropyLossConfig, MseLoss, Reduction},
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, InferenceStep, RegressionOutput, TrainOutput, TrainStep},
};

use crate::batcher::EyeDataBatch;

#[derive(Module, Debug)]
pub struct MicroChad<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    conv5: Conv2d<B>,
    conv6: Conv2d<B>,
    fc_gaze: Linear<B>,
    pool: MaxPool2d,
    adaptive: MaxPool2d,
    act: Relu,
    sigmoid: Sigmoid,
}

#[derive(Config, Debug)]
pub struct MicroChadConfig {
    out_count: usize,
}

impl MicroChadConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MicroChad<B> {
        MicroChad {
            conv1: Conv2dConfig::new([4, 28], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            conv2: Conv2dConfig::new([28, 42], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            conv3: Conv2dConfig::new([42, 63], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            conv4: Conv2dConfig::new([63, 94], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            conv5: Conv2dConfig::new([94, 141], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            conv6: Conv2dConfig::new([141, 212], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),

            fc_gaze: LinearConfig::new(212, self.out_count).init(device),

            pool: MaxPool2dConfig::new([2, 2])
                .with_strides([2, 2])
                .with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0))
                .with_dilation([1, 1])
                .with_ceil_mode(false)
                .init(),
            adaptive: MaxPool2dConfig::new([1, 1]).init(),

            act: Relu::new(),
            sigmoid: Sigmoid::new(),
        }
    }
}

impl<B: Backend> MicroChad<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(x);
        let x = self.act.forward(x);
        let x = self.pool.forward(x);

        let x = self.conv2.forward(x);
        let x = self.act.forward(x);
        let x = self.pool.forward(x);

        let x = self.conv3.forward(x);
        let x = self.act.forward(x);
        let x = self.pool.forward(x);

        let x = self.conv4.forward(x);
        let x = self.act.forward(x);
        let x = self.pool.forward(x);

        let x = self.conv5.forward(x);
        let x = self.act.forward(x);
        let x = self.pool.forward(x);

        let x = self.conv6.forward(x);
        let x = self.act.forward(x);

        let x = self.adaptive.forward(x);

        let x = x.max_dims(&[2, 3]);
        let x = x.flatten(1, -1);

        let x = self.fc_gaze.forward(x);
        let x = self.sigmoid.forward(x);

        x
    }

    pub fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 2>,
    ) -> RegressionOutput<B> {
        let output = self.forward(images);
        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Reduction::Auto);

        RegressionOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep for MicroChad<B> {
    type Input = EyeDataBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: EyeDataBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for MicroChad<B> {
    type Input = EyeDataBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: EyeDataBatch<B>) -> RegressionOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[derive(Module, Debug)]
pub struct MultiInputMergedMicroChad<B: Backend> {
    left: MicroChad<B>,
    right: MicroChad<B>,
}

#[derive(Config, Debug)]
pub struct MultiInputMergedMicroChadConfig {
    out_count: usize,
}

impl MultiInputMergedMicroChadConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiInputMergedMicroChad<B> {
        MultiInputMergedMicroChad {
            left: MicroChadConfig {
                out_count: self.out_count,
            }
            .init(device),
            right: MicroChadConfig {
                out_count: self.out_count,
            }
            .init(device),
        }
    }

    pub fn forward<B: Backend>(
        &self,
        model: &MultiInputMergedMicroChad<B>,
        x: Tensor<B, 4>,
    ) -> Tensor<B, 2> {
        let inputs_left = x.clone().select(1, Tensor::from([0, 2, 4, 6]));
        let inputs_right = x.select(1, Tensor::from([1, 3, 5, 7]));

        let preds_left = model.left.forward(inputs_left);
        let preds_right = model.right.forward(inputs_right);

        let dim = preds_left.dims().len() - 1;
        Tensor::cat(vec![preds_left, preds_right], dim)
    }
}
