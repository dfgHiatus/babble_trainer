#![recursion_limit = "256"]

use crate::{
    frame_correlator::align_frames,
    loader::FileReader,
    models::{MicroChadConfig, MultiInputMergedMicroChadConfig},
    trainer::{TrainingConfig, train},
};
use burn::{backend::Autodiff, optim::AdamConfig};
use image::{ImageBuffer, Luma};
use log::{error, info};
// use burn_import::pytorch::PyTorchFileRecorder;

mod batcher;
mod corruption_detector;
mod frame_correlator;
mod loader;
mod models;
mod trainer;

pub type ImageData = ImageBuffer<Luma<u8>, Vec<u8>>;

fn main() {
    // env_logger::builder()
    //     .filter_level(log::LevelFilter::Info)
    //     .init();

    type Backend = burn::backend::Cuda;
    type AutodiffBackend = Autodiff<Backend>;
    let device = Default::default();
    let mut reader = FileReader::new();

    match reader.read_capture_file("./data/user_cal.bin", false, true, 0, 0) {
        Ok(_) => {
            info!("Finished processing capture file");
        }
        Err(e) => {
            error!("Error processing capture file: {e}");
        }
    }

    let aligned_frames = align_frames(reader);

    train::<AutodiffBackend>(
        "./artifacts",
        TrainingConfig::new(MultiInputMergedMicroChadConfig::new(3), AdamConfig::new()),
        device,
        aligned_frames,
    );
}
