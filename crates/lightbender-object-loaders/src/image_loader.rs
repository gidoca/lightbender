use std::path::Path;

use anyhow::{Context, Result};

use lightbender_scene::{SamplerDesc, TextureData, TextureFormat};

/// Load an image file (PNG, JPEG, etc.) and return RGBA8 texture data.
pub fn load_image_rgba8(path: &Path) -> Result<TextureData> {
    let img = image::open(path)
        .with_context(|| format!("open image: {}", path.display()))?;
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    let pixels = rgba.into_raw();

    Ok(TextureData {
        width:   w,
        height:  h,
        format:  TextureFormat::Rgba8,
        pixels,
        sampler: SamplerDesc::default(),
    })
}

/// Load an HDR / EXR image and return RGBA32F texture data.
///
/// The pixel data is stored as `Vec<u8>` containing the raw bytes of `f32`
/// values (4 channels × 4 bytes per channel per pixel).
pub fn load_image_hdr(path: &Path) -> Result<TextureData> {
    let img = image::open(path)
        .with_context(|| format!("open HDR image: {}", path.display()))?;
    let rgba = img.to_rgba32f();
    let (w, h) = (rgba.width(), rgba.height());
    let float_pixels: Vec<f32> = rgba.into_raw();
    // Reinterpret f32 slice as bytes
    let pixels: Vec<u8> = float_pixels
        .iter()
        .flat_map(|f| f.to_ne_bytes())
        .collect();

    Ok(TextureData {
        width:   w,
        height:  h,
        format:  TextureFormat::Rgba32F,
        pixels,
        sampler: SamplerDesc::default(),
    })
}
