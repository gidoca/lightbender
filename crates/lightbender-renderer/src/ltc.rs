#![allow(unsafe_op_in_unsafe_fn)]

//! Linearly Transformed Cosines (LTC) lookup-table resources.
//!
//! Two static 64×64 RGBA32F textures are loaded once at startup from the
//! embedded `ltc_lut.bin` blob and bound as descriptor set 4 to the PBR
//! pipeline:
//!
//!  * binding 0 (`ltcMat`) — GGX-fitted inverse-matrix components, sampled by
//!    `(roughness, sqrt(1 - NdotV))`.
//!  * binding 1 (`ltcMag`) — magnitude/scale, same parameterisation.
//!
//! See [`tools/gen_ltc_lut.py`](../../../../../tools/gen_ltc_lut.py) for the
//! exact byte layout.

use anyhow::{Context, Result};
use ash::vk;

use crate::image::GpuImage;

/// Embedded Heitz LTC LUT (two 64×64 RGBA32F textures, little-endian f32).
const LTC_LUT_BYTES: &[u8] = include_bytes!("../../../scenes/assets/ltc_lut.bin");

const LUT_SIZE: u32 = 64;
const LUT_TEXEL_BYTES: usize = 4 * 4; // RGBA32F
const LUT_SIZE_BYTES: usize = (LUT_SIZE as usize) * (LUT_SIZE as usize) * LUT_TEXEL_BYTES;

/// GPU resources for LTC area-light evaluation. Created once and never updated.
pub(crate) struct LtcResources {
    pub set_layout:      vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set:  vk::DescriptorSet,
    pub sampler:         vk::Sampler,
    pub mat_image:       GpuImage,
    pub mag_image:       GpuImage,
}

impl LtcResources {
    pub unsafe fn create(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<Self> {
        assert_eq!(
            LTC_LUT_BYTES.len(),
            2 * LUT_SIZE_BYTES,
            "ltc_lut.bin must contain exactly two 64x64 RGBA32F LUTs"
        );

        // `include_bytes!` produces a u8 array with alignment 1, so we can't
        // bytemuck::cast_slice it directly to &[f32]; copy into aligned Vecs.
        let (mat_bytes, mag_bytes) = LTC_LUT_BYTES.split_at(LUT_SIZE_BYTES);
        let to_floats = |bytes: &[u8]| -> Vec<f32> {
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let mat_pixels = to_floats(mat_bytes);
        let mag_pixels = to_floats(mag_bytes);

        let mat_image = GpuImage::upload_rgba32f(
            device, instance, physical_device, command_pool, queue,
            LUT_SIZE, LUT_SIZE, &mat_pixels,
        ).context("ltc Minv LUT")?;
        let mag_image = GpuImage::upload_rgba32f(
            device, instance, physical_device, command_pool, queue,
            LUT_SIZE, LUT_SIZE, &mag_pixels,
        ).context("ltc magnitude LUT")?;

        // Linear filtering, clamped at edges; the LUT bias is applied in the
        // shader so we don't need any wrap behaviour here.
        let sampler = device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
            None,
        ).context("ltc sampler")?;

        // --- Descriptor set layout (set 4) ---
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..2u32)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            })
            .collect();
        let set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
            .context("ltc descriptor set layout")?;

        // The LUTs are static, so a single descriptor set suffices for all frames.
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 2,
        }];
        let descriptor_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(1),
                None,
            )
            .context("ltc descriptor pool")?;

        let layouts = [set_layout];
        let descriptor_set = device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            )
            .context("ltc descriptor set")?[0];

        let mat_info = [vk::DescriptorImageInfo {
            sampler,
            image_view: mat_image.view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }];
        let mag_info = [vk::DescriptorImageInfo {
            sampler,
            image_view: mag_image.view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&mat_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&mag_info),
        ];
        device.update_descriptor_sets(&writes, &[]);

        Ok(Self {
            set_layout,
            descriptor_pool,
            descriptor_set,
            sampler,
            mat_image,
            mag_image,
        })
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        device.destroy_descriptor_pool(self.descriptor_pool, None);
        device.destroy_descriptor_set_layout(self.set_layout, None);
        device.destroy_sampler(self.sampler, None);
        self.mat_image.destroy(device);
        self.mag_image.destroy(device);
    }
}
