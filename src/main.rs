#![recursion_limit = "256"]

use crate::{
    frame_correlator::{AlignedFrame, align_frames},
    loader::FileReader,
    models::{MicroChadConfig, MultiInputMergedMicroChadConfig}, trainer::{TrainingConfig, train},
};
use burn::{
    backend::Autodiff, optim::AdamConfig, prelude::Backend, record::{FullPrecisionSettings, Recorder}
};
use log::{error, info};
// use burn_import::pytorch::PyTorchFileRecorder;

mod batcher;
mod corruption_detector;
mod frame_correlator;
mod loader;
mod models;
mod trainer;

fn main() {
    // env_logger::builder()
    //     .filter_level(log::LevelFilter::Warn)
    //     .init();

    type Backend = burn::backend::Cuda;
    type AutodiffBackend = Autodiff<Backend>;
    let device = Default::default();
    let mut reader = FileReader::<AutodiffBackend>::new();

    match reader.read_capture_file(&device, "./data/gaze.bin", false, true, 0, 0) {
        Ok(_) => {
            info!("Finished processing capture file");
        }
        Err(e) => {
            error!("Error processing capture file: {e}");
        }
    }

    let aligned_frames = align_frames(reader);

    train::<AutodiffBackend>("./artifacts", 
        TrainingConfig::new(MicroChadConfig::new(3), AdamConfig::new()), device, aligned_frames);
}
