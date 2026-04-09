#![allow(unsafe_op_in_unsafe_fn)]

//! Screen-Space Ambient Occlusion (SSAO) resources.
//!
//! Provides three passes that slot between shadow mapping and the main PBR
//! pass:
//!
//!  1. **G-buffer pre-pass** — renders all opaque geometry into a view-space
//!     normal texture + depth.
//!  2. **SSAO pass** — fullscreen triangle that reads the G-buffer and writes
//!     a half-resolution R8 occlusion image.
//!  3. **SSAO blur pass** — 4×4 box blur to remove the tiled noise pattern.
//!
//! The blurred AO image is then bound as descriptor set 5 in the main PBR
//! pass, where `pbr.frag` multiplies it into the ambient term.

use anyhow::{Context, Result};
use ash::vk;

use crate::buffer::{find_memory_type, GpuBuffer};

const NOISE_SIZE: u32 = 4;
const KERNEL_SIZE: usize = 64;

// ── SSAO UBO (matches ssao.frag SsaoParams) ────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoUbo {
    samples: [[f32; 4]; KERNEL_SIZE],
    projection: [[f32; 4]; 4],
    inverse_projection: [[f32; 4]; 4],
    radius: f32,
    bias: f32,
    power: f32,
    _pad: f32,
    noise_scale: [f32; 2],
    _pad2: [f32; 2],
}

// ── Helper: create a device-local image with given format + usage ───────────

pub(crate) struct ImageAlloc {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
}

#[allow(clippy::too_many_arguments)]
unsafe fn create_attachment(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    width: u32,
    height: u32,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
) -> Result<ImageAlloc> {
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
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
            None,
        )
        .context("ssao create_attachment image")?;

    let req = device.get_image_memory_requirements(image);
    let mem_idx = find_memory_type(
        instance, physical_device, req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let memory = device
        .allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mem_idx),
            None,
        )
        .context("ssao create_attachment memory")?;
    device.bind_image_memory(image, memory, 0)?;

    let view = device
        .create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: aspect,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
            None,
        )
        .context("ssao create_attachment view")?;

    Ok(ImageAlloc { image, memory, view })
}

unsafe fn destroy_attachment(device: &ash::Device, a: &ImageAlloc) {
    device.destroy_image_view(a.view, None);
    device.destroy_image(a.image, None);
    device.free_memory(a.memory, None);
}

// ── G-buffer resources ─────────────────────────────────────────────────────

pub(crate) struct GBufferResources {
    pub render_pass: vk::RenderPass,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub framebuffer: vk::Framebuffer,
    pub normal_image: ImageAlloc,
    pub depth_image: ImageAlloc,
    extent: vk::Extent2D,
}

impl GBufferResources {
    pub unsafe fn create(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        extent: vk::Extent2D,
        frame_set_layout: vk::DescriptorSetLayout,
        pipeline_cache: vk::PipelineCache,
    ) -> Result<Self> {
        // Render pass: color attachment 0 (normal) + depth
        let attachments = [
            // Normal (A2B10G10R10 UNORM)
            vk::AttachmentDescription::default()
                .format(vk::Format::A2B10G10R10_UNORM_PACK32)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            // Depth
            vk::AttachmentDescription::default()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        ];
        let color_ref = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let depth_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_ref)
            .depth_stencil_attachment(&depth_ref);
        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::BOTTOM_OF_PIPE)
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );

        let render_pass = device
            .create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&attachments)
                    .subpasses(std::slice::from_ref(&subpass))
                    .dependencies(std::slice::from_ref(&dependency)),
                None,
            )
            .context("gbuffer render pass")?;

        // Pipeline layout: set 0 (frame UBO) + push constants (model mat4)
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(64);
        let pipeline_layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&frame_set_layout))
                    .push_constant_ranges(std::slice::from_ref(&push_range)),
                None,
            )
            .context("gbuffer pipeline layout")?;

        let pipeline = create_gbuffer_pipeline(device, render_pass, pipeline_layout, pipeline_cache)?;

        let normal_image = create_attachment(
            device, instance, physical_device,
            extent.width, extent.height,
            vk::Format::A2B10G10R10_UNORM_PACK32,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
        )?;
        let depth_image = create_attachment(
            device, instance, physical_device,
            extent.width, extent.height,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::DEPTH,
        )?;

        let fb_views = [normal_image.view, depth_image.view];
        let framebuffer = device
            .create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&fb_views)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1),
                None,
            )
            .context("gbuffer framebuffer")?;

        Ok(Self {
            render_pass,
            pipeline_layout,
            pipeline,
            framebuffer,
            normal_image,
            depth_image,
            extent,
        })
    }

    pub unsafe fn resize(
        &mut self,
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        extent: vk::Extent2D,
    ) -> Result<()> {
        device.destroy_framebuffer(self.framebuffer, None);
        destroy_attachment(device, &self.normal_image);
        destroy_attachment(device, &self.depth_image);

        self.normal_image = create_attachment(
            device, instance, physical_device,
            extent.width, extent.height,
            vk::Format::A2B10G10R10_UNORM_PACK32,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
        )?;
        self.depth_image = create_attachment(
            device, instance, physical_device,
            extent.width, extent.height,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::DEPTH,
        )?;

        let fb_views = [self.normal_image.view, self.depth_image.view];
        self.framebuffer = device
            .create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(self.render_pass)
                    .attachments(&fb_views)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1),
                None,
            )
            .context("gbuffer framebuffer resize")?;
        self.extent = extent;
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        device.destroy_framebuffer(self.framebuffer, None);
        destroy_attachment(device, &self.normal_image);
        destroy_attachment(device, &self.depth_image);
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_render_pass(self.render_pass, None);
    }
}

unsafe fn create_gbuffer_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    pipeline_cache: vk::PipelineCache,
) -> Result<vk::Pipeline> {
    let vert_spv = crate::renderer::gbuffer_vert_spv();
    let frag_spv = crate::renderer::gbuffer_frag_spv();
    let vert_mod = device
        .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vert_spv), None)
        .context("gbuffer vert module")?;
    let frag_mod = device
        .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&frag_spv), None)
        .context("gbuffer frag module")?;

    let entry = c"main";
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_mod)
            .name(entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_mod)
            .name(entry),
    ];

    let vertex_binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(std::mem::size_of::<lightbender_scene::Vertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let vertex_attrs = [
        vk::VertexInputAttributeDescription { location: 0, binding: 0, format: vk::Format::R32G32B32_SFLOAT, offset: 0 },
        vk::VertexInputAttributeDescription { location: 1, binding: 0, format: vk::Format::R32G32B32_SFLOAT, offset: 12 },
        vk::VertexInputAttributeDescription { location: 2, binding: 0, format: vk::Format::R32G32_SFLOAT, offset: 24 },
        vk::VertexInputAttributeDescription { location: 3, binding: 0, format: vk::Format::R32G32B32A32_SFLOAT, offset: 32 },
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(std::slice::from_ref(&vertex_binding))
        .vertex_attribute_descriptions(&vertex_attrs);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS);
    let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    let blend_attachments = [blend_attachment];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
        .dynamic_states(&dynamic_states);

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipeline = device
        .create_graphics_pipelines(pipeline_cache, &[pipeline_info], None)
        .map_err(|(_, e)| e)
        .context("gbuffer pipeline")?[0];

    device.destroy_shader_module(vert_mod, None);
    device.destroy_shader_module(frag_mod, None);

    Ok(pipeline)
}

// ── SSAO resources ─────────────────────────────────────────────────────────

pub(crate) struct SsaoResources {
    // Render passes
    pub ssao_render_pass: vk::RenderPass,
    pub blur_render_pass: vk::RenderPass,

    // Pipelines
    pub ssao_pipeline_layout: vk::PipelineLayout,
    pub ssao_pipeline: vk::Pipeline,
    pub blur_pipeline_layout: vk::PipelineLayout,
    pub blur_pipeline: vk::Pipeline,

    // Screen-sized images
    pub ao_image: ImageAlloc,
    pub ao_blur_image: ImageAlloc,
    pub ssao_framebuffer: vk::Framebuffer,
    pub blur_framebuffer: vk::Framebuffer,

    // Static resources
    pub noise_image: ImageAlloc,
    pub kernel_ubo: GpuBuffer,
    pub sampler: vk::Sampler,
    pub nearest_sampler: vk::Sampler,

    // Descriptor sets
    pub ssao_set_layout: vk::DescriptorSetLayout,
    pub blur_set_layout: vk::DescriptorSetLayout,
    /// Set 5: the blurred AO image, consumed by PBR pass
    pub output_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub ssao_descriptor_set: vk::DescriptorSet,
    pub blur_descriptor_set: vk::DescriptorSet,
    pub output_descriptor_set: vk::DescriptorSet,

    ao_extent: vk::Extent2D,
}

impl SsaoResources {
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn create(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        pipeline_cache: vk::PipelineCache,
        full_extent: vk::Extent2D,
        gbuffer: &GBufferResources,
    ) -> Result<Self> {
        let ao_extent = vk::Extent2D {
            width: (full_extent.width / 2).max(1),
            height: (full_extent.height / 2).max(1),
        };

        // --- Noise texture (4×4, RG16F, random unit vectors in tangent-space) ---
        let noise_pixels = generate_noise();
        let noise_image = upload_rg16f(
            device, instance, physical_device, command_pool, queue,
            NOISE_SIZE, NOISE_SIZE, &noise_pixels,
        )?;

        // --- Kernel UBO ---
        let kernel = generate_kernel();
        let ubo_size = std::mem::size_of::<SsaoUbo>() as vk::DeviceSize;
        let kernel_ubo = GpuBuffer::new(
            device, instance, physical_device, ubo_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        // Write kernel samples — the matrices are filled per-frame in update_ssao_ubo
        let mut ubo_data = SsaoUbo {
            samples: kernel,
            projection: [[0.0; 4]; 4],
            inverse_projection: [[0.0; 4]; 4],
            radius: 0.5,
            bias: 0.025,
            power: 1.5,
            _pad: 0.0,
            noise_scale: [
                ao_extent.width as f32 / NOISE_SIZE as f32,
                ao_extent.height as f32 / NOISE_SIZE as f32,
            ],
            _pad2: [0.0; 2],
        };
        // Zero-init the matrices; they'll be updated per-frame
        ubo_data.projection = glam::Mat4::IDENTITY.to_cols_array_2d();
        ubo_data.inverse_projection = glam::Mat4::IDENTITY.to_cols_array_2d();
        kernel_ubo.upload_slice(device, std::slice::from_ref(&ubo_data))?;

        // --- Samplers ---
        let sampler = device
            .create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )
            .context("ssao linear sampler")?;
        let nearest_sampler = device
            .create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT)
                    .address_mode_w(vk::SamplerAddressMode::REPEAT),
                None,
            )
            .context("ssao noise sampler")?;

        // --- Render passes ---
        let ssao_render_pass = create_r8_render_pass(device, "ssao")?;
        let blur_render_pass = create_r8_render_pass(device, "ssao blur")?;

        // --- Descriptor set layouts ---
        // SSAO input set: depth, normal, noise, UBO
        let ssao_bindings = [
            // 0: depth
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // 1: normal
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // 2: noise
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // 3: UBO (kernel + params)
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let ssao_set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&ssao_bindings),
                None,
            )
            .context("ssao set layout")?;

        // Blur input set: one sampler (the raw AO)
        let blur_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let blur_set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&blur_bindings),
                None,
            )
            .context("ssao blur set layout")?;

        // Output set (set 5 in PBR pipeline): blurred AO
        let output_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let output_set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&output_bindings),
                None,
            )
            .context("ssao output set layout")?;

        // --- Pipelines ---
        let ssao_pipeline_layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&ssao_set_layout)),
                None,
            )
            .context("ssao pipeline layout")?;
        let ssao_pipeline = create_fullscreen_pipeline(
            device, ssao_render_pass, ssao_pipeline_layout, pipeline_cache,
            &crate::renderer::fullscreen_vert_spv(), &crate::renderer::ssao_frag_spv(),
            "ssao",
        )?;

        let blur_pipeline_layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&blur_set_layout)),
                None,
            )
            .context("ssao blur pipeline layout")?;
        let blur_pipeline = create_fullscreen_pipeline(
            device, blur_render_pass, blur_pipeline_layout, pipeline_cache,
            &crate::renderer::fullscreen_vert_spv(), &crate::renderer::ssao_blur_frag_spv(),
            "ssao blur",
        )?;

        // --- Screen-sized images ---
        let ao_image = create_attachment(
            device, instance, physical_device,
            ao_extent.width, ao_extent.height,
            vk::Format::R8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
        )?;
        let ao_blur_image = create_attachment(
            device, instance, physical_device,
            ao_extent.width, ao_extent.height,
            vk::Format::R8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
        )?;

        let ssao_framebuffer = device
            .create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(ssao_render_pass)
                    .attachments(std::slice::from_ref(&ao_image.view))
                    .width(ao_extent.width)
                    .height(ao_extent.height)
                    .layers(1),
                None,
            )
            .context("ssao framebuffer")?;
        let blur_framebuffer = device
            .create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(blur_render_pass)
                    .attachments(std::slice::from_ref(&ao_blur_image.view))
                    .width(ao_extent.width)
                    .height(ao_extent.height)
                    .layers(1),
                None,
            )
            .context("ssao blur framebuffer")?;

        // --- Descriptor pool + sets ---
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                // ssao: 3 (depth, normal, noise) + blur: 1 + output: 1 = 5
                descriptor_count: 5,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
            },
        ];
        let descriptor_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(3),
                None,
            )
            .context("ssao descriptor pool")?;

        let layouts = [ssao_set_layout, blur_set_layout, output_set_layout];
        let sets = device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            )
            .context("ssao descriptor sets")?;
        let ssao_descriptor_set = sets[0];
        let blur_descriptor_set = sets[1];
        let output_descriptor_set = sets[2];

        let res = Self {
            ssao_render_pass,
            blur_render_pass,
            ssao_pipeline_layout,
            ssao_pipeline,
            blur_pipeline_layout,
            blur_pipeline,
            ao_image,
            ao_blur_image,
            ssao_framebuffer,
            blur_framebuffer,
            noise_image,
            kernel_ubo,
            sampler,
            nearest_sampler,
            ssao_set_layout,
            blur_set_layout,
            output_set_layout,
            descriptor_pool,
            ssao_descriptor_set,
            blur_descriptor_set,
            output_descriptor_set,
            ao_extent,
        };

        res.write_descriptor_sets(device, gbuffer);
        Ok(res)
    }

    unsafe fn write_descriptor_sets(&self, device: &ash::Device, gbuffer: &GBufferResources) {
        // SSAO input set
        let depth_info = [vk::DescriptorImageInfo {
            sampler: self.sampler,
            image_view: gbuffer.depth_image.view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }];
        let normal_info = [vk::DescriptorImageInfo {
            sampler: self.sampler,
            image_view: gbuffer.normal_image.view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }];
        let noise_info = [vk::DescriptorImageInfo {
            sampler: self.nearest_sampler,
            image_view: self.noise_image.view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }];
        let ubo_info = [vk::DescriptorBufferInfo {
            buffer: self.kernel_ubo.buffer,
            offset: 0,
            range: std::mem::size_of::<SsaoUbo>() as vk::DeviceSize,
        }];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.ssao_descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&depth_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.ssao_descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&normal_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.ssao_descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&noise_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.ssao_descriptor_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&ubo_info),
        ];
        device.update_descriptor_sets(&writes, &[]);

        // Blur input set: raw AO
        let ao_info = [vk::DescriptorImageInfo {
            sampler: self.sampler,
            image_view: self.ao_image.view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }];
        let blur_writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.blur_descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&ao_info),
        ];
        device.update_descriptor_sets(&blur_writes, &[]);

        // Output set: blurred AO (consumed by PBR)
        let blur_ao_info = [vk::DescriptorImageInfo {
            sampler: self.sampler,
            image_view: self.ao_blur_image.view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }];
        let output_writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.output_descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&blur_ao_info),
        ];
        device.update_descriptor_sets(&output_writes, &[]);
    }

    pub unsafe fn update_ubo(
        &self,
        device: &ash::Device,
        projection: glam::Mat4,
        inverse_projection: glam::Mat4,
        full_extent: vk::Extent2D,
    ) -> Result<()> {
        let kernel = generate_kernel();
        let ao_extent = vk::Extent2D {
            width: (full_extent.width / 2).max(1),
            height: (full_extent.height / 2).max(1),
        };
        let ubo = SsaoUbo {
            samples: kernel,
            projection: projection.to_cols_array_2d(),
            inverse_projection: inverse_projection.to_cols_array_2d(),
            radius: 0.5,
            bias: 0.025,
            power: 1.5,
            _pad: 0.0,
            noise_scale: [
                ao_extent.width as f32 / NOISE_SIZE as f32,
                ao_extent.height as f32 / NOISE_SIZE as f32,
            ],
            _pad2: [0.0; 2],
        };
        self.kernel_ubo.upload_slice(device, std::slice::from_ref(&ubo))?;
        Ok(())
    }

    pub unsafe fn resize(
        &mut self,
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        full_extent: vk::Extent2D,
        gbuffer: &GBufferResources,
    ) -> Result<()> {
        let ao_extent = vk::Extent2D {
            width: (full_extent.width / 2).max(1),
            height: (full_extent.height / 2).max(1),
        };

        device.destroy_framebuffer(self.ssao_framebuffer, None);
        device.destroy_framebuffer(self.blur_framebuffer, None);
        destroy_attachment(device, &self.ao_image);
        destroy_attachment(device, &self.ao_blur_image);

        self.ao_image = create_attachment(
            device, instance, physical_device,
            ao_extent.width, ao_extent.height,
            vk::Format::R8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
        )?;
        self.ao_blur_image = create_attachment(
            device, instance, physical_device,
            ao_extent.width, ao_extent.height,
            vk::Format::R8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
        )?;

        self.ssao_framebuffer = device
            .create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(self.ssao_render_pass)
                    .attachments(std::slice::from_ref(&self.ao_image.view))
                    .width(ao_extent.width)
                    .height(ao_extent.height)
                    .layers(1),
                None,
            )
            .context("ssao framebuffer resize")?;
        self.blur_framebuffer = device
            .create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(self.blur_render_pass)
                    .attachments(std::slice::from_ref(&self.ao_blur_image.view))
                    .width(ao_extent.width)
                    .height(ao_extent.height)
                    .layers(1),
                None,
            )
            .context("ssao blur framebuffer resize")?;

        self.ao_extent = ao_extent;
        self.write_descriptor_sets(device, gbuffer);
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        device.destroy_framebuffer(self.ssao_framebuffer, None);
        device.destroy_framebuffer(self.blur_framebuffer, None);
        destroy_attachment(device, &self.ao_image);
        destroy_attachment(device, &self.ao_blur_image);
        destroy_attachment(device, &self.noise_image);
        self.kernel_ubo.destroy(device);
        device.destroy_sampler(self.sampler, None);
        device.destroy_sampler(self.nearest_sampler, None);
        device.destroy_pipeline(self.ssao_pipeline, None);
        device.destroy_pipeline_layout(self.ssao_pipeline_layout, None);
        device.destroy_pipeline(self.blur_pipeline, None);
        device.destroy_pipeline_layout(self.blur_pipeline_layout, None);
        device.destroy_descriptor_pool(self.descriptor_pool, None);
        device.destroy_descriptor_set_layout(self.ssao_set_layout, None);
        device.destroy_descriptor_set_layout(self.blur_set_layout, None);
        device.destroy_descriptor_set_layout(self.output_set_layout, None);
        device.destroy_render_pass(self.ssao_render_pass, None);
        device.destroy_render_pass(self.blur_render_pass, None);
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn create_r8_render_pass(device: &ash::Device, label: &str) -> Result<vk::RenderPass> {
    let attachment = vk::AttachmentDescription::default()
        .format(vk::Format::R8_UNORM)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let color_ref = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_ref);
    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::SHADER_READ)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

    unsafe {
        device
            .create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(std::slice::from_ref(&attachment))
                    .subpasses(std::slice::from_ref(&subpass))
                    .dependencies(std::slice::from_ref(&dependency)),
                None,
            )
            .context(format!("{label} render pass"))
    }
}

fn create_fullscreen_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    pipeline_cache: vk::PipelineCache,
    vert_spv: &[u32],
    frag_spv: &[u32],
    label: &str,
) -> Result<vk::Pipeline> {
    unsafe {
        let vert_mod = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(vert_spv), None)
            .context(format!("{label} vert module"))?;
        let frag_mod = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(frag_spv), None)
            .context(format!("{label} frag module"))?;

        let entry = c"main";
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_mod)
                .name(entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_mod)
                .name(entry),
        ];

        // No vertex input — fullscreen triangle generated in vert shader
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
        let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::R);
        let blend_attachments = [blend_attachment];
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&blend_attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipeline = device
            .create_graphics_pipelines(pipeline_cache, &[pipeline_info], None)
            .map_err(|(_, e)| e)
            .context(format!("{label} pipeline"))?[0];

        device.destroy_shader_module(vert_mod, None);
        device.destroy_shader_module(frag_mod, None);

        Ok(pipeline)
    }
}

/// Upload RG16F pixel data to a device-local 2D image.
#[allow(clippy::too_many_arguments)]
unsafe fn upload_rg16f(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    width: u32,
    height: u32,
    pixels: &[u16],
) -> Result<ImageAlloc> {
    let byte_data: &[u8] = bytemuck::cast_slice(pixels);
    let size = byte_data.len() as vk::DeviceSize;

    let staging = GpuBuffer::new(
        device, instance, physical_device, size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    staging.upload_slice(device, pixels)?;

    let image = device
        .create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R16G16_SFLOAT)
                .extent(vk::Extent3D { width, height, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
            None,
        )
        .context("noise image")?;

    let req = device.get_image_memory_requirements(image);
    let mem_idx = find_memory_type(
        instance, physical_device, req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let memory = device
        .allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mem_idx),
            None,
        )
        .context("noise image memory")?;
    device.bind_image_memory(image, memory, 0)?;

    let cmd = crate::buffer::begin_one_shot(device, command_pool)?;

    crate::image::transition_layout(
        device, cmd, image,
        vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER,
        vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE,
    );

    device.cmd_copy_buffer_to_image(
        cmd, staging.buffer, image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D { width, height, depth: 1 },
        }],
    );

    crate::image::transition_layout(
        device, cmd, image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ,
    );

    crate::buffer::end_one_shot(device, command_pool, queue, cmd)?;
    staging.destroy(device);

    let view = device
        .create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R16G16_SFLOAT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
            None,
        )
        .context("noise image view")?;

    Ok(ImageAlloc { image, memory, view })
}

/// Generate 64 hemisphere sample vectors, cosine-weighted and biased toward
/// the origin for better distribution near the surface.
fn generate_kernel() -> [[f32; 4]; KERNEL_SIZE] {
    use std::f32::consts::PI;
    let mut kernel = [[0.0f32; 4]; KERNEL_SIZE];

    // Simple deterministic pseudo-random using a linear congruential generator
    let mut seed: u32 = 0x12345678;
    let mut next_f32 = || -> f32 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (seed >> 16) as f32 / 65535.0
    };

    for (i, sample) in kernel.iter_mut().enumerate() {
        // Random point on hemisphere
        let xi1 = next_f32();
        let xi2 = next_f32();
        let r = (1.0 - xi1 * xi1).sqrt(); // cosine-weighted
        let phi = 2.0 * PI * xi2;

        let mut x = r * phi.cos();
        let mut y = r * phi.sin();
        let mut z = xi1; // hemisphere, z >= 0

        // Scale: accelerate toward origin
        let scale = {
            let t = i as f32 / KERNEL_SIZE as f32;
            0.1 + 0.9 * t * t // lerp(0.1, 1.0, t*t)
        };
        x *= scale;
        y *= scale;
        z *= scale;

        *sample = [x, y, z, 0.0];
    }
    kernel
}

/// Generate a 4×4 tiled noise texture (RG16F). Each texel is a random
/// unit-length tangent-space rotation vector (xy, z=0).
fn generate_noise() -> Vec<u16> {
    let count = (NOISE_SIZE * NOISE_SIZE) as usize;
    let mut pixels = Vec::with_capacity(count * 2);

    let mut seed: u32 = 0xDEADBEEF;
    let mut next_f32 = || -> f32 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (seed >> 16) as f32 / 65535.0 * 2.0 - 1.0 // [-1, 1]
    };

    for _ in 0..count {
        let x = next_f32();
        let y = next_f32();
        let len = (x * x + y * y).sqrt().max(0.001);
        let nx = x / len;
        let ny = y / len;
        pixels.push(f32_to_f16_bits(nx));
        pixels.push(f32_to_f16_bits(ny));
    }
    pixels
}

/// Convert an f32 to IEEE 754 half-precision (f16) bit pattern.
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exponent == 0xFF {
        // Inf / NaN
        return (sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 }) as u16;
    }
    let exp = exponent - 127 + 15;
    if exp >= 31 {
        return (sign | 0x7C00) as u16; // overflow → inf
    }
    if exp <= 0 {
        return sign as u16; // underflow → zero
    }
    (sign | ((exp as u32) << 10) | (mantissa >> 13)) as u16
}
