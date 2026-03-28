use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use ash::vk;
use glam::Mat4;
use winit::window::Window;

use crate::camera::Camera;
use crate::scene::{gltf_loader::LoadContext, Scene};
use crate::shader::ShaderPair;
use crate::types::{FrameUniforms, GpuVertex, MAX_LIGHTS};
use crate::vulkan::buffer::{upload_to_device_local, GpuBuffer};

const FRAMES_IN_FLIGHT: usize = 2;
const PIPELINE_CACHE_FILE: &str = "pipeline_cache.bin";

pub struct Renderer {
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

    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,

    render_pass: vk::RenderPass,
    /// Set 0: per-frame UBO
    descriptor_set_layout: vk::DescriptorSetLayout,
    /// Set 1: per-material textures (5× combined image samplers)
    pub material_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline_cache: vk::PipelineCache,
    /// "default" pipeline + any user-loaded named pipelines
    pipelines: HashMap<String, vk::Pipeline>,

    framebuffers: Vec<vk::Framebuffer>,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    // Per-frame UBOs
    ubo_buffers: Vec<GpuBuffer>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,

    // Mesh geometry
    vertex_buffer: GpuBuffer,
    index_buffer: GpuBuffer,
    index_count: u32,

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

        // --- Debug messenger ---
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

        // --- Command pool ---
        let command_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(graphics_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .context("command pool")?;

        // --- Swapchain ---
        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
        let (swapchain, swapchain_images, swapchain_image_views, swapchain_format, swapchain_extent) =
            create_swapchain(
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

        // --- Depth image ---
        let (depth_image, depth_image_memory, depth_image_view) = create_depth_image(
            &device,
            &instance,
            physical_device,
            swapchain_extent,
            command_pool,
            graphics_queue,
        )?;

        // --- Render pass ---
        let render_pass =
            create_render_pass(&device, swapchain_format).context("render pass")?;

        // --- Descriptor set layout (set 0: frame UBO) ---
        let descriptor_set_layout = create_descriptor_set_layout(&device)?;

        // --- Descriptor set layout (set 1: material textures) ---
        let material_set_layout = create_material_set_layout(&device)?;

        // --- Pipeline cache (load from disk if available) ---
        let cache_data = std::fs::read(PIPELINE_CACHE_FILE).unwrap_or_default();
        let pipeline_cache = device.create_pipeline_cache(
            &vk::PipelineCacheCreateInfo::default().initial_data(&cache_data),
            None,
        ).context("pipeline cache")?;
        if cache_data.is_empty() {
            log::debug!("Pipeline cache cold (no file on disk)");
        } else {
            log::debug!("Pipeline cache loaded from {PIPELINE_CACHE_FILE} ({} bytes)", cache_data.len());
        }

        // --- Pipeline ---
        let (pipeline_layout, default_pipeline) =
            create_pipeline(&device, render_pass, descriptor_set_layout, material_set_layout, pipeline_cache, None)
                .context("pipeline")?;
        let mut pipelines = HashMap::new();
        pipelines.insert("default".to_string(), default_pipeline);

        // --- Framebuffers ---
        let framebuffers = create_framebuffers(
            &device,
            render_pass,
            &swapchain_image_views,
            depth_image_view,
            swapchain_extent,
        )?;

        // --- Command buffers ---
        let command_buffers = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(FRAMES_IN_FLIGHT as u32),
            )
            .context("command buffers")?;

        // --- Per-frame UBO buffers ---
        let ubo_size = std::mem::size_of::<FrameUniforms>() as vk::DeviceSize;
        let mut ubo_buffers = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            ubo_buffers.push(GpuBuffer::new(
                &device,
                &instance,
                physical_device,
                ubo_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?);
        }

        // --- Descriptor pool + sets ---
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: FRAMES_IN_FLIGHT as u32,
        }];
        let descriptor_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(FRAMES_IN_FLIGHT as u32),
                None,
            )
            .context("descriptor pool")?;

        let layouts = vec![descriptor_set_layout; FRAMES_IN_FLIGHT];
        let descriptor_sets = device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            )
            .context("descriptor sets")?;

        for (i, &ds) in descriptor_sets.iter().enumerate() {
            let buf_info = [vk::DescriptorBufferInfo {
                buffer: ubo_buffers[i].buffer,
                offset: 0,
                range:  ubo_size,
            }];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buf_info);
            device.update_descriptor_sets(&[write], &[]);
        }

        // --- Hardcoded cube mesh ---
        let (vertices, indices) = cube_mesh();
        let vertex_buffer = upload_to_device_local(
            &device,
            &instance,
            physical_device,
            command_pool,
            graphics_queue,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &vertices,
        )?;
        let index_buffer = upload_to_device_local(
            &device,
            &instance,
            physical_device,
            command_pool,
            graphics_queue,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &indices,
        )?;
        let index_count = indices.len() as u32;

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
            depth_image,
            depth_image_memory,
            depth_image_view,
            render_pass,
            descriptor_set_layout,
            material_set_layout,
            pipeline_layout,
            pipeline_cache,
            pipelines,
            framebuffers,
            command_pool,
            command_buffers,
            ubo_buffers,
            descriptor_pool,
            descriptor_sets,
            vertex_buffer,
            index_buffer,
            index_count,
            image_available,
            render_finished,
            in_flight,
            current_frame: 0,
            window,
        })
    }

    pub fn draw_frame(&mut self, camera: &Camera, scene: Option<&Scene>) -> Result<()> {
        unsafe { self.draw_frame_inner(camera, scene) }
    }

    unsafe fn draw_frame_inner(&mut self, camera: &Camera, scene: Option<&Scene>) -> Result<()> {
        let frame = self.current_frame;

        self.device
            .wait_for_fences(&[self.in_flight[frame]], true, u64::MAX)?;

        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(()); // minimized — nothing to draw
        }

        let (image_index, suboptimal) = match self.swapchain_loader.acquire_next_image(
            self.swapchain,
            u64::MAX,
            self.image_available[frame],
            vk::Fence::null(),
        ) {
            Ok(r) => r,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.rebuild_swapchain(size.width, size.height)?;
                return Ok(());
            }
            Err(e) => return Err(e.into()),
        };
        if suboptimal {
            // Swapchain still usable; rebuild after presenting this frame
        }

        self.device.reset_fences(&[self.in_flight[frame]])?;

        // Update UBO
        self.update_ubo(frame, camera)?;

        let cmd = self.command_buffers[frame];
        self.device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        self.record_commands(cmd, image_index as usize, frame, scene)?;

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

        let present_result = self
            .swapchain_loader
            .queue_present(self.present_queue, &present_info);

        self.current_frame = (self.current_frame + 1) % FRAMES_IN_FLIGHT;

        match present_result {
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                let sz = self.window.inner_size();
                self.rebuild_swapchain(sz.width, sz.height)?;
            }
            Ok(false) if suboptimal => {
                let sz = self.window.inner_size();
                self.rebuild_swapchain(sz.width, sz.height)?;
            }
            Ok(_) => {}
            Err(e) => return Err(e.into()),
        }
        Ok(())
    }

    unsafe fn update_ubo(&self, frame: usize, camera: &Camera) -> Result<()> {
        let extent = self.swapchain_extent;
        let aspect = extent.width as f32 / extent.height as f32;

        let view = camera.view_matrix();
        let proj = camera.projection_matrix(aspect);
        let eye  = camera.position;

        let uniforms = FrameUniforms {
            view: view.to_cols_array_2d(),
            projection: proj.to_cols_array_2d(),
            camera_position: [eye.x, eye.y, eye.z, 1.0],
            lights: [Default::default(); MAX_LIGHTS],
            light_count: 0,
            _pad: [0; 3],
        };
        self.ubo_buffers[frame].upload_slice(&self.device, std::slice::from_ref(&uniforms))
    }

    unsafe fn record_commands(
        &self,
        cmd: vk::CommandBuffer,
        image_index: usize,
        frame: usize,
        scene: Option<&Scene>,
    ) -> Result<()> {
        self.device.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.1, 0.1, 0.15, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];
        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffers[image_index])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain_extent,
            })
            .clear_values(&clear_values);

        self.device
            .cmd_begin_render_pass(cmd, &render_pass_begin, vk::SubpassContents::INLINE);

        let default_pipeline = *self.pipelines.get("default").expect("default pipeline missing");
        let mut bound_pipeline = vk::Pipeline::null();

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

        // Bind frame descriptor set (set 0)
        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline_layout,
            0,
            &[self.descriptor_sets[frame]],
            &[],
        );

        if let Some(scene) = scene {
            for (world, prim) in scene.draw_primitives() {
                let mat = &scene.materials[prim.material];

                // Switch pipeline if this material requests a different one
                let pipeline = mat.pipeline_name.as_deref()
                    .and_then(|n| self.pipelines.get(n).copied())
                    .unwrap_or(default_pipeline);
                if pipeline != bound_pipeline {
                    self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                    bound_pipeline = pipeline;
                }

                // Bind material descriptor set (set 1)
                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    1,
                    &[mat.descriptor_set],
                    &[],
                );

                let model_arr = world.to_cols_array_2d();
                let model_bytes = bytemuck::bytes_of(&model_arr);
                self.device.cmd_push_constants(
                    cmd,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    model_bytes,
                );

                self.device.cmd_bind_vertex_buffers(cmd, 0, &[prim.vertex_buffer.buffer], &[0]);
                self.device.cmd_bind_index_buffer(cmd, prim.index_buffer.buffer, 0, vk::IndexType::UINT32);
                self.device.cmd_draw_indexed(cmd, prim.index_count, 1, 0, 0, 0);
            }
        } else {
            // Fallback: draw the hardcoded cube with the default pipeline
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, default_pipeline);
            let model_arr = Mat4::IDENTITY.to_cols_array_2d();
            let model_bytes = bytemuck::bytes_of(&model_arr);
            self.device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                model_bytes,
            );
            self.device.cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buffer.buffer], &[0]);
            self.device.cmd_bind_index_buffer(cmd, self.index_buffer.buffer, 0, vk::IndexType::UINT32);
            self.device.cmd_draw_indexed(cmd, self.index_count, 1, 0, 0, 0);
        }

        self.device.cmd_end_render_pass(cmd);
        self.device.end_command_buffer(cmd)?;
        Ok(())
    }

    /// Build and register a named pipeline from an already-loaded shader pair.
    pub fn add_pipeline(&mut self, name: &str, pair: &ShaderPair) -> Result<()> {
        unsafe {
            let (_, pipeline) = create_pipeline(
                &self.device,
                self.render_pass,
                self.descriptor_set_layout,
                self.material_set_layout,
                self.pipeline_cache,
                Some(pair),
            )?;
            if let Some(old) = self.pipelines.insert(name.to_string(), pipeline) {
                self.device.destroy_pipeline(old, None);
            }
            Ok(())
        }
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    pub fn device_wait_idle(&self) -> Result<()> {
        unsafe { Ok(self.device.device_wait_idle()?) }
    }

    /// Return a LoadContext for the glTF loader to use.
    pub fn load_context(&self) -> LoadContext<'_> {
        LoadContext {
            device:              &self.device,
            instance:            &self.instance,
            physical_device:     self.physical_device,
            command_pool:        self.command_pool,
            queue:               self.graphics_queue,
            material_set_layout: self.material_set_layout,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        unsafe { self.rebuild_swapchain(width, height) }
    }

    unsafe fn rebuild_swapchain(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        self.device.device_wait_idle()?;

        for fb in self.framebuffers.drain(..) {
            self.device.destroy_framebuffer(fb, None);
        }
        for iv in self.swapchain_image_views.drain(..) {
            self.device.destroy_image_view(iv, None);
        }
        self.device.destroy_image_view(self.depth_image_view, None);
        self.device.destroy_image(self.depth_image, None);
        self.device.free_memory(self.depth_image_memory, None);

        let old_swapchain = self.swapchain;
        let (swapchain, images, image_views, format, extent) = create_swapchain(
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

        let (di, dim, div) = create_depth_image(
            &self.device,
            &self.instance,
            self.physical_device,
            extent,
            self.command_pool,
            self.graphics_queue,
        )?;
        self.depth_image = di;
        self.depth_image_memory = dim;
        self.depth_image_view = div;

        self.framebuffers = create_framebuffers(
            &self.device,
            self.render_pass,
            &self.swapchain_image_views,
            self.depth_image_view,
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

            self.vertex_buffer.destroy(&self.device);
            self.index_buffer.destroy(&self.device);

            for buf in &self.ubo_buffers {
                buf.destroy(&self.device);
            }
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_descriptor_set_layout(self.material_set_layout, None);

            self.device.destroy_command_pool(self.command_pool, None);

            for fb in &self.framebuffers {
                self.device.destroy_framebuffer(*fb, None);
            }
            self.device.destroy_image_view(self.depth_image_view, None);
            self.device.destroy_image(self.depth_image, None);
            self.device.free_memory(self.depth_image_memory, None);

            for iv in &self.swapchain_image_views {
                self.device.destroy_image_view(*iv, None);
            }

            for pipeline in self.pipelines.values() {
                self.device.destroy_pipeline(*pipeline, None);
            }
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);

            // Save pipeline cache to disk for faster startup next time
            match self.device.get_pipeline_cache_data(self.pipeline_cache) {
                Ok(data) => {
                    if let Err(e) = std::fs::write(PIPELINE_CACHE_FILE, &data) {
                        log::warn!("Failed to save pipeline cache: {e}");
                    } else {
                        log::debug!("Pipeline cache saved ({} bytes)", data.len());
                    }
                }
                Err(e) => log::warn!("Failed to get pipeline cache data: {e}"),
            }
            self.device.destroy_pipeline_cache(self.pipeline_cache, None);

            self.device.destroy_render_pass(self.render_pass, None);

            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
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
    let mut best: Option<(vk::PhysicalDevice, u32, u32, u32)> = None;

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

    let format = formats
        .iter()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .or_else(|| formats.first())
        .copied()
        .context("no surface format")?;

    let present_mode = present_modes
        .iter()
        .find(|&&m| m == vk::PresentModeKHR::MAILBOX)
        .copied()
        .unwrap_or(vk::PresentModeKHR::FIFO);

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

unsafe fn create_depth_image(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    extent: vk::Extent2D,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    let format = vk::Format::D32_SFLOAT;
    let image = device
        .create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
            None,
        )
        .context("create depth image")?;

    let req = device.get_image_memory_requirements(image);
    let mem_type = crate::vulkan::buffer::find_memory_type(
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
        .context("depth image memory")?;
    device.bind_image_memory(image, memory, 0)?;

    // Transition to depth attachment layout
    let cmd = crate::vulkan::buffer::begin_one_shot(device, command_pool)?;
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[barrier],
    );
    crate::vulkan::buffer::end_one_shot(device, command_pool, queue, cmd)?;

    let view = device
        .create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
            None,
        )
        .context("depth image view")?;

    Ok((image, memory, view))
}

unsafe fn create_render_pass(
    device: &ash::Device,
    color_format: vk::Format,
) -> Result<vk::RenderPass> {
    let attachments = [
        // Color
        vk::AttachmentDescription::default()
            .format(color_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
        // Depth
        vk::AttachmentDescription::default()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
    ];

    let color_refs = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let depth_ref = vk::AttachmentReference {
        attachment: 1,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs)
        .depth_stencil_attachment(&depth_ref)];

    let dependencies = [vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )];

    Ok(device.create_render_pass(
        &vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies),
        None,
    )?)
}

unsafe fn create_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)];

    Ok(device.create_descriptor_set_layout(
        &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
        None,
    )?)
}

unsafe fn create_material_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    // 5 combined image samplers: base_color, normal, metallic_roughness, occlusion, emissive
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..5u32)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        })
        .collect();
    Ok(device.create_descriptor_set_layout(
        &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
        None,
    )?)
}

unsafe fn create_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
    material_set_layout: vk::DescriptorSetLayout,
    pipeline_cache: vk::PipelineCache,
    // Pre-loaded shader modules to use. If None, loads the built-in mesh shaders.
    shader_pair: Option<&ShaderPair>,
) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
    // Either use provided modules or load the built-in mesh shaders
    let (vert_module, frag_module, owned) = if let Some(pair) = shader_pair {
        (pair.vert, pair.frag, false)
    } else {
        let vert_spv = load_spirv(std::path::Path::new("shaders/compiled/mesh.vert.spv"))
            .context("load vertex shader")?;
        let frag_spv = load_spirv(std::path::Path::new("shaders/compiled/mesh.frag.spv"))
            .context("load fragment shader")?;
        let v = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vert_spv), None)
            .context("vert shader module")?;
        let f = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&frag_spv), None)
            .context("frag shader module")?;
        (v, f, true)
    };

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

    let vertex_binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(std::mem::size_of::<GpuVertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let vertex_attrs = [
        vk::VertexInputAttributeDescription {
            location: 0, binding: 0,
            format: vk::Format::R32G32B32_SFLOAT, offset: 0,
        },
        vk::VertexInputAttributeDescription {
            location: 1, binding: 0,
            format: vk::Format::R32G32B32_SFLOAT, offset: 12,
        },
        vk::VertexInputAttributeDescription {
            location: 2, binding: 0,
            format: vk::Format::R32G32_SFLOAT, offset: 24,
        },
        vk::VertexInputAttributeDescription {
            location: 3, binding: 0,
            format: vk::Format::R32G32B32A32_SFLOAT, offset: 32,
        },
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
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false);

    let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    let blend_attachments = [blend_attachment];
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let set_layouts = [descriptor_set_layout, material_set_layout];
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(64); // mat4
    let layout = device
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(std::slice::from_ref(&push_constant_range)),
            None,
        )
        .context("pipeline layout")?;

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

    let pipelines = device
        .create_graphics_pipelines(pipeline_cache, &[pipeline_info], None)
        .map_err(|(_, e)| e)
        .context("graphics pipeline")?;

    // Only destroy modules we created ourselves (not caller-owned ones)
    if owned {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }

    Ok((layout, pipelines[0]))
}

unsafe fn create_framebuffers(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    image_views: &[vk::ImageView],
    depth_view: vk::ImageView,
    extent: vk::Extent2D,
) -> Result<Vec<vk::Framebuffer>> {
    image_views
        .iter()
        .map(|&iv| {
            let attachments = [iv, depth_view];
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

/// Hardcoded unit cube with per-face normals.
fn cube_mesh() -> (Vec<GpuVertex>, Vec<u32>) {
    // 6 faces × 4 vertices = 24 vertices
    let faces: [([f32; 3], [f32; 3]); 6] = [
        // (normal, +axis offset)
        ([0.0, 0.0,  1.0], [0.0, 0.0,  1.0]), // front  +Z
        ([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]), // back   -Z
        ([1.0, 0.0,  0.0], [1.0, 0.0,  0.0]), // right  +X
        ([-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]), // left   -X
        ([0.0,  1.0, 0.0], [0.0,  1.0, 0.0]), // top    +Y
        ([0.0, -1.0, 0.0], [0.0, -1.0, 0.0]), // bottom -Y
    ];

    // Local 2D corners (tangent / bitangent)
    let corners: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
    let uvs: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);

    for (face_idx, (normal, center)) in faces.iter().enumerate() {
        let n = glam::Vec3::from(*normal);
        // Choose tangent: for Y-faces use X, otherwise use Y
        let up = if n.y.abs() > 0.9 { glam::Vec3::Z } else { glam::Vec3::Y };
        let tangent = n.cross(up).normalize();
        let bitangent = n.cross(tangent).normalize();

        let base = face_idx as u32 * 4;
        for (i, corner) in corners.iter().enumerate() {
            let pos = glam::Vec3::from(*center)
                + tangent * corner[0]
                + bitangent * corner[1];
            vertices.push(GpuVertex {
                position: pos.to_array(),
                normal: *normal,
                uv: uvs[i],
                tangent: [tangent.x, tangent.y, tangent.z, 1.0],
            });
        }
        // Two CCW triangles per face
        indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    }

    (vertices, indices)
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
