use std::{
    ffi::{CStr, CString, c_char},
    path::Path,
    sync::{LazyLock, Mutex},
};

use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    record::{Recorder, SensitiveCompactRecorder},
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

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct ModelOutput {
    pub pitch_l: f32,
    pub yaw_l: f32,
    pub blink_l: f32,
    pub eyebrow_l: f32,
    pub eyewide_l: f32,
    pub pitch_r: f32,
    pub yaw_r: f32,
    pub blink_r: f32,
    pub eyebrow_r: f32,
    pub eyewide_r: f32,
}

type B = burn::backend::Cuda;

#[derive(Debug)]
struct InferenceState {
    model: MultiInputMergedMicroChad<B>,
    batcher: EyeDataBatcher,
}

static STATE: LazyLock<Mutex<Result<InferenceState, String>>> =
    LazyLock::new(|| Mutex::new(Err("Model is not yet initialised".into())));

static LEFT_EYE_FRAMES: LazyLock<Mutex<CircularBuffer<4, ImageData>>> =
    LazyLock::new(|| Mutex::new(CircularBuffer::new()));
static RIGHT_EYE_FRAMES: LazyLock<Mutex<CircularBuffer<4, ImageData>>> =
    LazyLock::new(|| Mutex::new(CircularBuffer::new()));

fn setup_inference(model_path: &str) -> Result<InferenceState, String> {
    let device = Default::default();

    let config = match TrainingConfig::load(format!("{model_path}_config.json")) {
        Ok(config) => config,
        Err(err) => {
            return Err(format!("Failed to load training config for model: {err}"));
        }
    };
    let record = match SensitiveCompactRecorder::new().load(model_path.into(), &device) {
        Ok(record) => record,
        Err(err) => {
            return Err(format!("Failed to load trained model: {err}"));
        }
    };

    let model = config.model.init::<B>(&device).load_record(record);

    let batcher = EyeDataBatcher {
        training: false,
        dataset_info: DatasetInfo::default(),
    };

    Ok(InferenceState { model, batcher })
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


#[repr(C)]
pub struct ModelOutputResult {
    is_error: bool,
    value: ModelOutputValue,
}

#[repr(C)]
pub union ModelOutputValue {
    model_output: ModelOutput,
    error_message: *mut c_char,
}

impl ModelOutputResult {
    pub fn from_error(message: String) -> Self {
        Self {
            is_error: true,
            value: ModelOutputValue {
                error_message: CString::new(message).unwrap().into_raw(),
            },
        }
    }

    pub fn from_output(output: ModelOutput) -> Self {
        Self {
            is_error: false,
            value: ModelOutputValue {
                model_output: output,
            },
        }
    }

    pub fn is_error(&self) -> bool {
        self.is_error
    }

    pub fn get_error_message(&self) -> Option<String> {
        if self.is_error() {
            unsafe {
                let c_str = CStr::from_ptr(self.value.error_message);
                Some(c_str.to_string_lossy().into_owned())
            }
        } else {
            None
        }
    }

    pub fn get_model_output(&self) -> Option<ModelOutput> {
        if !self.is_error() {
            unsafe { Some(self.value.model_output) }
        } else {
            None
        }
    }
}

pub fn load_model(model_path: &str) -> Result<(), String> {
    match setup_inference(model_path) {
        Ok(inference_state) => {
            println!("Model loaded successfully from path: {model_path}");
            *STATE.lock().unwrap() = Ok(inference_state);
            return Ok(());
        }
        Err(err) => {
            println!("Failed to load model from path: {model_path}, error: {err}");
            *STATE.lock().unwrap() = Err(err.clone());
            return Err(err);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn loadModel(path_ptr: *const c_char) -> ModelOutputResult {
    let c_str = unsafe { CStr::from_ptr(path_ptr) };

    let c_str_str = c_str.to_str().unwrap_or_default();
    let decoded = urlencoding::decode(c_str_str).unwrap().to_string();
    let file_path = Path::new(&decoded);

    // Remove file extension for model loading
    let file_stem = file_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .split('.')
        .next()
        .unwrap_or_default();
    let model_path = file_path.with_file_name(file_stem);

    println!("Loading model from path: {:?} ({:?})", c_str, model_path);

    match load_model(model_path.to_str().unwrap_or_default()) {
        Ok(()) => ModelOutputResult::from_output(ModelOutput::default()),
        Err(err) => ModelOutputResult::from_error(err),
    }
}

pub fn run_inference(
    left: &[u8; 128 * 128],
    right: &[u8; 128 * 128],
) -> Result<ModelOutput, String> {
    // let start_time = std::time::Instant::now();
    let device = Default::default();

    let state = STATE.lock().unwrap();
    if let Err(err) = state.as_ref() {
        return Err(err.clone());
    }

    let state = state.as_ref().unwrap();
    let mut left_frames = LEFT_EYE_FRAMES.lock().unwrap();
    let mut right_frames = RIGHT_EYE_FRAMES.lock().unwrap();

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

    // println!(
    //     "Model output: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
    //     output[0],
    //     output[1],
    //     output[2],
    //     output[3],
    //     output[4],
    //     output[5],
    //     output[6],
    //     output[7],
    //     output[8],
    //     output[9]
    // );

    Ok(ModelOutput {
        pitch_l: output[0],
        yaw_l: output[1],
        blink_l: output[2],
        eyebrow_l: output[3],
        eyewide_l: output[4],
        pitch_r: output[5],
        yaw_r: output[6],
        blink_r: output[7],
        eyebrow_r: output[8],
        eyewide_r: output[9],
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn infer(left: &[u8; 128 * 128], right: &[u8; 128 * 128]) -> ModelOutputResult {
    match run_inference(left, right) {
        Ok(output) => ModelOutputResult::from_output(output),
        Err(err) => ModelOutputResult::from_error(err),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn freeModelOutputResult(result: ModelOutputResult) {
    if result.is_error() {
        unsafe {
            let _ = CString::from_raw(result.value.error_message);
        }
    }
}
