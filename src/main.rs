#![recursion_limit = "256"]

use babble_model::{
    ImageData, ImageLabel, batcher::EyeDataBatcher, loader::FileReader,
    models::MultiInputMergedMicroChadConfig, trainer::TrainingConfig,
};
use burn::{
    backend::Autodiff,
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
use log::{error, info};

use crate::frame_correlator::{AlignedFrame, ImageToTensor, align_frames, create_dataset};

mod frame_correlator;

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

    let dataset_info = create_dataset(&frames);

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

fn main() {
    // env_logger::builder()
    //     .filter_level(log::LevelFilter::Info)
    //     .init();

    type Backend = burn::backend::Cuda;
    type AutodiffBackend = Autodiff<Backend>;
    let device = Default::default();
    let mut reader = FileReader::new();

    let aligned_frames = match reader.read_capture_file("./data/user_cal.bin", false, true, 0, 0) {
        Ok(_) => {
            info!("Finished processing capture file");

            let mut left_frames: Vec<(u64, ImageData)> = reader
                .left_eye_frames
                .iter()
                .map(|(ts, img)| (*ts, img.clone()))
                .collect();
            let mut right_frames: Vec<(u64, ImageData)> = reader
                .right_eye_frames
                .iter()
                .map(|(ts, img)| (*ts, img.clone()))
                .collect();
            let mut label_frames: Vec<(u64, ImageLabel)> = reader
                .label_frames
                .iter()
                .map(|(ts, label)| (*ts, label.clone()))
                .collect();

            left_frames.sort_by_key(|(ts, _)| *ts);
            right_frames.sort_by_key(|(ts, _)| *ts);
            label_frames.sort_by_key(|(ts, _)| *ts);

            align_frames(left_frames, right_frames, label_frames)
        }
        Err(e) => {
            error!("Failed to read capture file: {e}");
            return;
        }
    };

    train::<AutodiffBackend>(
        "./artifacts",
        TrainingConfig::new(MultiInputMergedMicroChadConfig::new(3), AdamConfig::new()),
        device,
        aligned_frames,
    );
}
