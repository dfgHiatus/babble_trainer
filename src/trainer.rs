use burn::{
    data::{
        dataloader::DataLoaderBuilder,
        dataset::{
            Dataset, InMemDataset,
            transform::{MapperDataset, PartialDataset, ShuffledDataset, WindowsDataset},
        },
    },
    module::Module,
    optim::AdamConfig,
    prelude::Config,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{Learner, SupervisedTraining, metric::LossMetric},
};

use crate::{
    batcher::{DatasetInfo, EyeDataBatcher, ImageToTensor},
    frame_correlator::AlignedFrame,
    models::MultiInputMergedMicroChadConfig,
};

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

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
    frames: Vec<AlignedFrame>,
) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(&device, config.seed);

    let dataset_info = DatasetInfo::from_frames(&frames);

    let batcher = EyeDataBatcher {
        training: true,
        dataset_info,
    };

    let get_data = |frames, start, end| {
        let frames = ShuffledDataset::new(
            WindowsDataset::new(InMemDataset::new(frames), 4),
            config.seed,
        );
        let len = frames.len();

        MapperDataset::new(
            PartialDataset::new(frames, len * start / 10, len * end / 10),
            ImageToTensor,
        )
    };

    let dataset_train = get_data(frames.clone(), 0, 8);
    let dataset_test = get_data(frames, 8, 10);

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_train);

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_test);

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metrics((LossMetric::new(),))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary();

    let model = config.model.init::<B>(&device);
    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
