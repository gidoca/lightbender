use std::sync::Arc;

use anyhow::{Context, Result};
use ash::vk;
use winit::window::Window;

const FRAMES_IN_FLIGHT: usize = 2;

pub struct Renderer {
    // Keep entry + instance alive
    _entry: ash::Entry,
    instance: ash::Instance,
    #[cfg(debug_assertions)]
    debug_utils: ash::ext::debug_utils::Instance,
    #[cfg(debug_assertions)]
    debug_messenger: vk::DebugUtilsMessengerEXT,
    #[cfg(debug_assertions)]
    enable_validation: bool,

    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,

    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    graphics_family: u32,
    present_family: u32,

    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,

    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,

    framebuffers: Vec<vk::Framebuffer>,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight: Vec<vk::Fence>,
    current_frame: usize,

    window: Arc<Window>,
}

impl Renderer {
    pub fn new(window: Arc<Window>) -> Result<Self> {
        unsafe { Self::init(window) }
    }

    unsafe fn init(window: Arc<Window>) -> Result<Self> {
        use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

        let entry = ash::Entry::load().context("load Vulkan — is a Vulkan driver installed?")?;

        // --- Instance ---
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"lightbender")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"lightbender")
            .api_version(vk::API_VERSION_1_0);

        let display_handle = window.display_handle()?.as_raw();
        let mut required_exts =
            ash_window::enumerate_required_extensions(display_handle)?.to_vec();

        // Enable validation layer + debug messenger if available in debug builds
        let validation_layer = c"VK_LAYER_KHRONOS_validation";
        let available_layers = entry.enumerate_instance_layer_properties()?;
        let has_validation = available_layers.iter().any(|l| {
            l.layer_name_as_c_str()
                .map(|n| n == validation_layer)
                .unwrap_or(false)
        });

        #[cfg(debug_assertions)]
        let enable_validation = has_validation;
        #[cfg(not(debug_assertions))]
        let enable_validation = false;

        if cfg!(debug_assertions) && !has_validation {
            log::warn!("VK_LAYER_KHRONOS_validation not available — running without validation");
        }

        if enable_validation {
            required_exts.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        let layer_ptrs: Vec<*const i8> = if enable_validation {
            vec![validation_layer.as_ptr()]
        } else {
            vec![]
        };

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&required_exts)
            .enabled_layer_names(&layer_ptrs);

        let instance = entry
            .create_instance(&instance_info, None)
            .context("create instance")?;

        // --- Debug messenger (only when validation layer is active) ---
        #[cfg(debug_assertions)]
        let (debug_utils, debug_messenger) = if enable_validation {
            let du = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));
            let messenger = du
                .create_debug_utils_messenger(&info, None)
                .context("debug messenger")?;
            (du, messenger)
        } else {
            (
                ash::ext::debug_utils::Instance::new(&entry, &instance),
                vk::DebugUtilsMessengerEXT::null(),
            )
        };

        // --- Surface ---
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        let surface = ash_window::create_surface(
            &entry,
            &instance,
            display_handle,
            window.window_handle()?.as_raw(),
            None,
        )
        .context("create surface")?;

        // --- Physical device ---
        let (physical_device, graphics_family, present_family) =
            pick_physical_device(&instance, &surface_loader, surface)?;

        // --- Logical device ---
        let queue_priorities = [1.0f32];
        let mut queue_infos = vec![vk::DeviceQueueCreateInfo::default()
            .queue_family_index(graphics_family)
            .queue_priorities(&queue_priorities)];
        if present_family != graphics_family {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(present_family)
                    .queue_priorities(&queue_priorities),
            );
        }

        let device_exts = [ash::khr::swapchain::NAME.as_ptr()];
        let features = vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_exts)
            .enabled_features(&features);

        let device = instance
            .create_device(physical_device, &device_info, None)
            .context("create device")?;
        let graphics_queue = device.get_device_queue(graphics_family, 0);
        let present_queue = device.get_device_queue(present_family, 0);

        // --- Swapchain ---
        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
        let (swapchain, swapchain_images, swapchain_image_views, swapchain_format, swapchain_extent) =
            create_swapchain(
                &instance,
                &device,
                physical_device,
                &surface_loader,
                surface,
                graphics_family,
                present_family,
                &swapchain_loader,
                vk::SwapchainKHR::null(),
                &window,
            )?;

        // --- Render pass ---
        let render_pass =
            create_render_pass(&device, swapchain_format).context("render pass")?;

        // --- Pipeline ---
        let (pipeline_layout, pipeline) =
            create_pipeline(&device, render_pass, swapchain_extent)
                .context("pipeline")?;

        // --- Framebuffers ---
        let framebuffers = create_framebuffers(
            &device,
            render_pass,
            &swapchain_image_views,
            swapchain_extent,
        )?;

        // --- Command pool & buffers ---
        let command_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(graphics_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .context("command pool")?;

        let command_buffers = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(FRAMES_IN_FLIGHT as u32),
            )
            .context("command buffers")?;

        // --- Sync ---
        let mut image_available = Vec::with_capacity(FRAMES_IN_FLIGHT);
        let mut render_finished = Vec::with_capacity(FRAMES_IN_FLIGHT);
        let mut in_flight = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            image_available.push(
                device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .context("semaphore")?,
            );
            render_finished.push(
                device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .context("semaphore")?,
            );
            in_flight.push(
                device
                    .create_fence(
                        &vk::FenceCreateInfo::default()
                            .flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )
                    .context("fence")?,
            );
        }

        Ok(Self {
            _entry: entry,
            instance,
            #[cfg(debug_assertions)]
            debug_utils,
            #[cfg(debug_assertions)]
            debug_messenger,
            #[cfg(debug_assertions)]
            enable_validation,
            surface_loader,
            surface,
            physical_device,
            device,
            graphics_queue,
            present_queue,
            graphics_family,
            present_family,
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            swapchain_format,
            swapchain_extent,
            render_pass,
            pipeline_layout,
            pipeline,
            framebuffers,
            command_pool,
            command_buffers,
            image_available,
            render_finished,
            in_flight,
            current_frame: 0,
            window,
        })
    }

    pub fn draw_frame(&mut self) -> Result<()> {
        unsafe { self.draw_frame_inner() }
    }

    unsafe fn draw_frame_inner(&mut self) -> Result<()> {
        let frame = self.current_frame;

        self.device
            .wait_for_fences(&[self.in_flight[frame]], true, u64::MAX)?;

        let (image_index, suboptimal) = match self.swapchain_loader.acquire_next_image(
            self.swapchain,
            u64::MAX,
            self.image_available[frame],
            vk::Fence::null(),
        ) {
            Ok(r) => r,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                let size = self.window.inner_size();
                self.rebuild_swapchain(size.width, size.height)?;
                return Ok(());
            }
            Err(e) => return Err(e.into()),
        };

        self.device.reset_fences(&[self.in_flight[frame]])?;

        let cmd = self.command_buffers[frame];
        self.device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        self.record_commands(cmd, image_index as usize)?;

        let wait_semaphores = [self.image_available[frame]];
        let signal_semaphores = [self.render_finished[frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [cmd];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        self.device
            .queue_submit(self.graphics_queue, &[submit_info], self.in_flight[frame])?;

        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        match self
            .swapchain_loader
            .queue_present(self.present_queue, &present_info)
        {
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                let size = self.window.inner_size();
                self.rebuild_swapchain(size.width, size.height)?;
            }
            Ok(false) => {}
            Err(e) if suboptimal => {
                let _ = e;
                let size = self.window.inner_size();
                self.rebuild_swapchain(size.width, size.height)?;
            }
            Err(e) => return Err(e.into()),
        }

        self.current_frame = (self.current_frame + 1) % FRAMES_IN_FLIGHT;
        Ok(())
    }

    unsafe fn record_commands(&self, cmd: vk::CommandBuffer, image_index: usize) -> Result<()> {
        self.device.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.1, 0.1, 0.15, 1.0],
            },
        }];
        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffers[image_index])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain_extent,
            })
            .clear_values(&clear_values);

        self.device.cmd_begin_render_pass(
            cmd,
            &render_pass_begin,
            vk::SubpassContents::INLINE,
        );

        self.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.swapchain_extent.width as f32,
            height: self.swapchain_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        self.device.cmd_set_viewport(cmd, 0, &[viewport]);

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.swapchain_extent,
        };
        self.device.cmd_set_scissor(cmd, 0, &[scissor]);

        self.device.cmd_draw(cmd, 3, 1, 0, 0);

        self.device.cmd_end_render_pass(cmd);
        self.device.end_command_buffer(cmd)?;
        Ok(())
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        unsafe { self.rebuild_swapchain(width, height) }
    }

    unsafe fn rebuild_swapchain(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        self.device.device_wait_idle()?;

        // Destroy old framebuffers and image views
        for fb in self.framebuffers.drain(..) {
            self.device.destroy_framebuffer(fb, None);
        }
        for iv in self.swapchain_image_views.drain(..) {
            self.device.destroy_image_view(iv, None);
        }

        let old_swapchain = self.swapchain;
        let (swapchain, images, image_views, format, extent) = create_swapchain(
            &self.instance,
            &self.device,
            self.physical_device,
            &self.surface_loader,
            self.surface,
            self.graphics_family,
            self.present_family,
            &self.swapchain_loader,
            old_swapchain,
            &self.window,
        )?;
        self.swapchain_loader.destroy_swapchain(old_swapchain, None);

        self.swapchain = swapchain;
        self.swapchain_images = images;
        self.swapchain_image_views = image_views;
        self.swapchain_format = format;
        self.swapchain_extent = extent;

        self.framebuffers = create_framebuffers(
            &self.device,
            self.render_pass,
            &self.swapchain_image_views,
            self.swapchain_extent,
        )?;

        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();

            for i in 0..FRAMES_IN_FLIGHT {
                self.device.destroy_semaphore(self.image_available[i], None);
                self.device.destroy_semaphore(self.render_finished[i], None);
                self.device.destroy_fence(self.in_flight[i], None);
            }

            self.device.destroy_command_pool(self.command_pool, None);

            for fb in &self.framebuffers {
                self.device.destroy_framebuffer(*fb, None);
            }
            for iv in &self.swapchain_image_views {
                self.device.destroy_image_view(*iv, None);
            }

            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);

            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);

            #[cfg(debug_assertions)]
            if self.enable_validation && self.debug_messenger != vk::DebugUtilsMessengerEXT::null() {
                self.debug_utils
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }

            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

unsafe fn pick_physical_device(
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32, u32)> {
    let devices = instance.enumerate_physical_devices()?;
    let mut best: Option<(vk::PhysicalDevice, u32, u32, u32)> = None; // (dev, gfx, present, score)

    for dev in devices {
        let props = instance.get_physical_device_properties(dev);
        let queue_families = instance.get_physical_device_queue_family_properties(dev);

        let mut graphics_family = None;
        let mut present_family = None;

        for (i, qf) in queue_families.iter().enumerate() {
            if qf.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                graphics_family = Some(i as u32);
            }
            if surface_loader
                .get_physical_device_surface_support(dev, i as u32, surface)
                .unwrap_or(false)
            {
                present_family = Some(i as u32);
            }
        }

        // Check swapchain extension
        let ext_props = instance.enumerate_device_extension_properties(dev)?;
        let has_swapchain = ext_props.iter().any(|e| {
            e.extension_name_as_c_str()
                .map(|n| n == ash::khr::swapchain::NAME)
                .unwrap_or(false)
        });

        if let (Some(gfx), Some(prs)) = (graphics_family, present_family) {
            if has_swapchain {
                let score = match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
                    _ => 1,
                };
                if best.is_none() || score > best.unwrap().3 {
                    best = Some((dev, gfx, prs, score));
                }
            }
        }
    }

    let (dev, gfx, prs, _) = best.context("no suitable Vulkan device found")?;
    let props = instance.get_physical_device_properties(dev);
    log::info!(
        "Using GPU: {}",
        props.device_name_as_c_str().unwrap_or(c"unknown").to_str().unwrap_or("unknown")
    );
    Ok((dev, gfx, prs))
}

#[allow(clippy::too_many_arguments)]
unsafe fn create_swapchain(
    _instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    graphics_family: u32,
    present_family: u32,
    swapchain_loader: &ash::khr::swapchain::Device,
    old_swapchain: vk::SwapchainKHR,
    window: &Window,
) -> Result<(
    vk::SwapchainKHR,
    Vec<vk::Image>,
    Vec<vk::ImageView>,
    vk::Format,
    vk::Extent2D,
)> {
    let capabilities =
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?;
    let formats =
        surface_loader.get_physical_device_surface_formats(physical_device, surface)?;
    let present_modes =
        surface_loader.get_physical_device_surface_present_modes(physical_device, surface)?;

    // Pick format: prefer B8G8R8A8_SRGB / SRGB_NONLINEAR
    let format = formats
        .iter()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .or_else(|| formats.first())
        .copied()
        .context("no surface format")?;

    // Pick present mode: prefer MAILBOX, fallback FIFO
    let present_mode = present_modes
        .iter()
        .find(|&&m| m == vk::PresentModeKHR::MAILBOX)
        .copied()
        .unwrap_or(vk::PresentModeKHR::FIFO);

    // Extent
    let extent = if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        let size = window.inner_size();
        vk::Extent2D {
            width: size.width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: size.height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    };

    let image_count = {
        let desired = capabilities.min_image_count + 1;
        if capabilities.max_image_count > 0 {
            desired.min(capabilities.max_image_count)
        } else {
            desired
        }
    };

    let sharing_mode;
    let queue_families_buf;
    let queue_family_indices: &[u32] = if graphics_family != present_family {
        sharing_mode = vk::SharingMode::CONCURRENT;
        queue_families_buf = [graphics_family, present_family];
        &queue_families_buf
    } else {
        sharing_mode = vk::SharingMode::EXCLUSIVE;
        &[]
    };

    let create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(sharing_mode)
        .queue_family_indices(queue_family_indices)
        .pre_transform(capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(old_swapchain);

    let swapchain = swapchain_loader
        .create_swapchain(&create_info, None)
        .context("create swapchain")?;
    let images = swapchain_loader.get_swapchain_images(swapchain)?;

    let image_views: Vec<vk::ImageView> = images
        .iter()
        .map(|&img| {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(img)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format.format)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
        })
        .collect::<std::result::Result<_, _>>()
        .context("image views")?;

    Ok((swapchain, images, image_views, format.format, extent))
}

unsafe fn create_render_pass(
    device: &ash::Device,
    format: vk::Format,
) -> Result<vk::RenderPass> {
    let attachments = [vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];

    let color_refs = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs)];

    let dependencies = [vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

    let render_pass_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    Ok(device.create_render_pass(&render_pass_info, None)?)
}

unsafe fn create_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
    let vert_spv = load_spirv(std::path::Path::new("shaders/compiled/triangle.vert.spv"))
        .context("load vertex shader")?;
    let frag_spv = load_spirv(std::path::Path::new("shaders/compiled/triangle.frag.spv"))
        .context("load fragment shader")?;

    let vert_module = device
        .create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&vert_spv),
            None,
        )
        .context("vert shader module")?;
    let frag_module = device
        .create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&frag_spv),
            None,
        )
        .context("frag shader module")?;

    let entry_point = c"main";
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(entry_point),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(entry_point),
    ];

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: extent.width as f32,
        height: extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let scissor = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent,
    };
    let viewports = [viewport];
    let scissors = [scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);

    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    let blend_attachments = [blend_attachment];
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let layout = device
        .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)
        .context("pipeline layout")?;

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipelines = device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        .map_err(|(_, e)| e)
        .context("graphics pipeline")?;

    device.destroy_shader_module(vert_module, None);
    device.destroy_shader_module(frag_module, None);

    Ok((layout, pipelines[0]))
}

unsafe fn create_framebuffers(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    image_views: &[vk::ImageView],
    extent: vk::Extent2D,
) -> Result<Vec<vk::Framebuffer>> {
    image_views
        .iter()
        .map(|&iv| {
            let attachments = [iv];
            device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(render_pass)
                        .attachments(&attachments)
                        .width(extent.width)
                        .height(extent.height)
                        .layers(1),
                    None,
                )
                .map_err(Into::into)
        })
        .collect()
}

pub fn load_spirv(path: &std::path::Path) -> Result<Vec<u32>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("read SPIR-V: {}", path.display()))?;
    anyhow::ensure!(bytes.len() % 4 == 0, "SPIR-V not 4-byte aligned");
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    anyhow::ensure!(magic == 0x07230203, "invalid SPIR-V magic");
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

#[cfg(debug_assertions)]
unsafe extern "system" fn vulkan_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _type: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let msg = unsafe {
        std::ffi::CStr::from_ptr((*data).p_message)
            .to_str()
            .unwrap_or("<invalid utf8>")
    };
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        log::error!("[Vulkan] {msg}");
    } else {
        log::warn!("[Vulkan] {msg}");
    }
    vk::FALSE
}

