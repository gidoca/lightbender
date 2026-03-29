use std::path::Path;

use anyhow::{Context, Result};
use ash::vk;

use super::buffer::{begin_one_shot, end_one_shot, find_memory_type, GpuBuffer};

pub struct GpuImage {
    pub image:  vk::Image,
    pub memory: vk::DeviceMemory,
    pub view:   vk::ImageView,
}

impl GpuImage {
    /// Upload RGBA8 pixel data to a DEVICE_LOCAL 2D image.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn upload_rgba8(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) -> Result<Self> {
        assert_eq!(pixels.len(), (width * height * 4) as usize);
        let size = pixels.len() as vk::DeviceSize;

        // Staging buffer
        let staging = GpuBuffer::new(
            device,
            instance,
            physical_device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        staging.upload_slice(device, pixels)?;

        // Device-local image
        let image = device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_SRGB)
                    .extent(vk::Extent3D { width, height, depth: 1 })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(
                        vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::SAMPLED,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .context("create texture image")?;

        let req = device.get_image_memory_requirements(image);
        let mem_type = find_memory_type(
            instance,
            physical_device,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let memory = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mem_type),
                None,
            )
            .context("texture image memory")?;
        device.bind_image_memory(image, memory, 0)?;

        // Transition UNDEFINED → TRANSFER_DST, copy, then TRANSFER_DST → SHADER_READ
        let cmd = begin_one_shot(device, command_pool)?;

        transition_layout(
            device,
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
        );

        device.cmd_copy_buffer_to_image(
            cmd,
            staging.buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::BufferImageCopy {
                buffer_offset:       0,
                buffer_row_length:   0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask:      vk::ImageAspectFlags::COLOR,
                    mip_level:        0,
                    base_array_layer: 0,
                    layer_count:      1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D { width, height, depth: 1 },
            }],
        );

        transition_layout(
            device,
            cmd,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );

        end_one_shot(device, command_pool, queue, cmd)?;
        staging.destroy(device);

        let view = device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_SRGB)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask:      vk::ImageAspectFlags::COLOR,
                        base_mip_level:   0,
                        level_count:      1,
                        base_array_layer: 0,
                        layer_count:      1,
                    }),
                None,
            )
            .context("texture image view")?;

        Ok(Self { image, memory, view })
    }

    /// Upload RGBA float32 pixel data to a DEVICE_LOCAL 2D image (R32G32B32A32_SFLOAT).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn upload_rgba32f(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        width: u32,
        height: u32,
        pixels: &[f32],
    ) -> Result<Self> {
        assert_eq!(pixels.len(), (width * height * 4) as usize);
        let byte_data: &[u8] = bytemuck::cast_slice(pixels);
        let size = byte_data.len() as vk::DeviceSize;

        // Staging buffer
        let staging = GpuBuffer::new(
            device,
            instance,
            physical_device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        staging.upload_slice(device, byte_data)?;

        let format = vk::Format::R32G32B32A32_SFLOAT;

        // Device-local image
        let image = device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D { width, height, depth: 1 })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(
                        vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::SAMPLED,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .context("create float texture image")?;

        let req = device.get_image_memory_requirements(image);
        let mem_type = find_memory_type(
            instance,
            physical_device,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let memory = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mem_type),
                None,
            )
            .context("float texture image memory")?;
        device.bind_image_memory(image, memory, 0)?;

        // Transition UNDEFINED → TRANSFER_DST, copy, then TRANSFER_DST → SHADER_READ
        let cmd = begin_one_shot(device, command_pool)?;

        transition_layout(
            device, cmd, image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
        );

        device.cmd_copy_buffer_to_image(
            cmd, staging.buffer, image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::BufferImageCopy {
                buffer_offset:       0,
                buffer_row_length:   0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask:      vk::ImageAspectFlags::COLOR,
                    mip_level:        0,
                    base_array_layer: 0,
                    layer_count:      1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D { width, height, depth: 1 },
            }],
        );

        transition_layout(
            device, cmd, image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );

        end_one_shot(device, command_pool, queue, cmd)?;
        staging.destroy(device);

        let view = device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask:      vk::ImageAspectFlags::COLOR,
                        base_mip_level:   0,
                        level_count:      1,
                        base_array_layer: 0,
                        layer_count:      1,
                    }),
                None,
            )
            .context("float texture image view")?;

        Ok(Self { image, memory, view })
    }

    pub unsafe fn destroy(&self, device: &ash::Device) {
        device.destroy_image_view(self.view, None);
        device.destroy_image(self.image, None);
        device.free_memory(self.memory, None);
    }
}

/// Load an HDR/EXR image file and return RGBA f32 pixel data.
pub fn load_hdr_to_rgba32f(path: &Path) -> Result<(u32, u32, Vec<f32>)> {
    let img = image::open(path)
        .with_context(|| format!("open HDR image: {}", path.display()))?;
    let rgba = img.to_rgba32f();
    let (w, h) = (rgba.width(), rgba.height());
    let pixels: Vec<f32> = rgba.into_raw();
    Ok((w, h, pixels))
}

#[allow(clippy::too_many_arguments)]
fn transition_layout(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask:      vk::ImageAspectFlags::COLOR,
            base_mip_level:   0,
            level_count:      1,
            base_array_layer: 0,
            layer_count:      1,
        })
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}
