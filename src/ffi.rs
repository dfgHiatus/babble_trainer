use std::sync::{Mutex, OnceLock};

use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    record::{CompactRecorder, Recorder},
};
use circular_buffer::CircularBuffer;
use histogram_equalization::hist_equal_hsv_rgb;
use image::{DynamicImage, GenericImageView, RgbImage};

use crate::{
    ImageData, ImageLabel,
    batcher::{DatasetInfo, EyeDataBatcher, WindowedFrame},
    models::MultiInputMergedMicroChad,
    trainer::TrainingConfig,
};

#[derive(Debug, Clone)]
#[repr(C)]
pub struct ModelOutput {
    pub pitch_l: f32,
    pub yaw_l: f32,
    pub blink_l: f32,
    pub pitch_r: f32,
    pub yaw_r: f32,
    pub blink_r: f32,
}

type B = burn::backend::Cuda;

#[derive(Debug)]
struct InferenceState {
    model: MultiInputMergedMicroChad<B>,
    batcher: EyeDataBatcher,
}

static STATE: OnceLock<Mutex<InferenceState>> = OnceLock::new();

static LEFT_EYE_FRAMES: OnceLock<Mutex<CircularBuffer<4, ImageData>>> = OnceLock::new();
static RIGHT_EYE_FRAMES: OnceLock<Mutex<CircularBuffer<4, ImageData>>> = OnceLock::new();

fn setup_inference() -> Mutex<InferenceState> {
    let artifact_dir = "E:\\github\\VRC Area\\babble_trainer\\artifacts";

    let device = Default::default();

    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let model = config.model.init::<B>(&device).load_record(record);

    let batcher = EyeDataBatcher {
        training: false,
        dataset_info: DatasetInfo::default(),
    };

    Mutex::new(InferenceState { model, batcher })
}

fn get_inference_state() -> std::sync::MutexGuard<'static, InferenceState> {
    STATE.get_or_init(|| setup_inference()).lock().unwrap()
}

fn get_left_eye_frames() -> std::sync::MutexGuard<'static, CircularBuffer<4, ImageData>> {
    LEFT_EYE_FRAMES
        .get_or_init(|| Mutex::new(CircularBuffer::new()))
        .lock()
        .unwrap()
}

fn get_right_eye_frames() -> std::sync::MutexGuard<'static, CircularBuffer<4, ImageData>> {
    RIGHT_EYE_FRAMES
        .get_or_init(|| Mutex::new(CircularBuffer::new()))
        .lock()
        .unwrap()
}

fn equalize_histogram(data: &[u8; 128 * 128]) -> ImageData {
    let img = DynamicImage::ImageLuma8(
        ImageData::from_raw(128, 128, data.to_vec()).expect("Input image should be valid"),
    );
    let dimensions = img.dimensions();
    let channels = 3;
    let stride = dimensions.0 as usize * channels;
    let mut dst_bytes: Vec<u8> = vec![0; stride * dimensions.1 as usize];
    let src_bytes = img.into_rgb8().into_raw();
    hist_equal_hsv_rgb(
        &src_bytes,
        stride as u32,
        &mut dst_bytes,
        stride as u32,
        dimensions.0,
        dimensions.1,
        128,
    );

    DynamicImage::ImageRgb8(
        RgbImage::from_raw(dimensions.0, dimensions.1, dst_bytes)
            .expect("histogram equalized image was invalid"),
    )
    .into_luma8()
}

#[unsafe(no_mangle)]
pub extern "C" fn infer(left: &[u8; 128 * 128], right: &[u8; 128 * 128]) -> ModelOutput {
    let start_time = std::time::Instant::now();
    let device = Default::default();

    let state = get_inference_state();
    let mut left_frames = get_left_eye_frames();
    let mut right_frames = get_right_eye_frames();

    left_frames.push_back(equalize_histogram(left));
    right_frames.push_back(equalize_histogram(right));

    let frame = WindowedFrame {
        label: ImageLabel::default(),
        left_eye: [
            left_frames
                .get(0)
                .cloned()
                .unwrap_or_else(|| ImageData::new(128, 128)),
            left_frames
                .get(1)
                .cloned()
                .unwrap_or_else(|| ImageData::new(128, 128)),
            left_frames
                .get(2)
                .cloned()
                .unwrap_or_else(|| ImageData::new(128, 128)),
            left_frames
                .get(3)
                .cloned()
                .unwrap_or_else(|| ImageData::new(128, 128)),
        ],
        right_eye: [
            right_frames
                .get(0)
                .cloned()
                .unwrap_or_else(|| ImageData::new(128, 128)),
            right_frames
                .get(1)
                .cloned()
                .unwrap_or_else(|| ImageData::new(128, 128)),
            right_frames
                .get(2)
                .cloned()
                .unwrap_or_else(|| ImageData::new(128, 128)),
            right_frames
                .get(3)
                .cloned()
                .unwrap_or_else(|| ImageData::new(128, 128)),
        ],
        timestamp: 0,
    };

    let batch = state.batcher.batch(vec![frame], &device);
    let output = state.model.forward(batch.images);

    let output = output.to_data().to_vec().unwrap();

    // println!(
    //     "Inference took {:.2?} ({} fps)",
    //     std::time::Instant::now() - start_time,
    //     1.0 / (std::time::Instant::now() - start_time).as_secs_f32()
    // );

    println!("Model output: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]", output[0], output[1], output[2], output[3], output[4], output[5]);

    ModelOutput {
        pitch_l: output[0],
        yaw_l: output[1],
        blink_l: output[2],
        pitch_r: output[3],
        yaw_r: output[4],
        blink_r: output[5],
    }
}
