use burn::{optim::AdamConfig, prelude::Config};

use crate::models::MultiInputMergedMicroChadConfig;

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: MultiInputMergedMicroChadConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 0)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}
