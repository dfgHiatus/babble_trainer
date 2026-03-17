use anyhow::{Result, anyhow};
use burn::{Tensor, prelude::Backend};
use histogram_equalization::hist_equal_hsv_rgb;
use image::{DynamicImage, GenericImageView, RgbImage};
use log::debug;

use crate::ImageData;

pub fn decode_jpeg(data: &[u8], equalize_histogram: bool) -> Result<ImageData> {
    let img = image::load_from_memory(data)?;
    debug!(
        "Decoded JPEG image: dimensions={:?}, color_type={:?}",
        img.dimensions(),
        img.color()
    );

    Ok(if equalize_histogram {
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
                .ok_or(anyhow!("histogram equalized image was invalid"))?,
        )
        .into_luma8()
    } else {
        img.into_luma8()
    })
}

pub fn image_to_tensor<B: Backend>(device: &B::Device, img: ImageData) -> Result<Tensor<B, 2>> {
    let (width, height) = img.dimensions();
    let raw_pixels = img.into_raw();

    // debug!("Converted image to grayscale: data length={}", data.len());
    // let tensor: Tensor<B, 1> = Tensor::from_data(&data[..], device).div_scalar(255.0);

    // Convert the raw pixel data into a tensor
    let tensor = Tensor::<B, 1>::from_data(&raw_pixels[..], device)
        .div_scalar(255.0)
        .reshape([height as usize, width as usize]);
    Ok(tensor)
}
