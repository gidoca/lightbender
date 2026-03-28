use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use glam::Vec3;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

use crate::camera::OrbitalCamera;
use crate::input::InputState;
use crate::renderer::Renderer;
use crate::scene::{gltf_loader, loader, GpuTexture, Scene};
use crate::shader::ShaderLibrary;
use crate::vulkan::image::{load_hdr_to_rgba32f, GpuImage};

pub enum InputPath {
    Model(PathBuf),
    Scene(PathBuf),
}

pub struct App {
    pub input_path: Option<InputPath>,
    state: Option<AppState>,
}

struct AppState {
    window:   Arc<Window>,
    // scene must be dropped before renderer (GPU resources freed before device)
    scene:    Option<Scene>,
    renderer: Renderer,
    camera:   OrbitalCamera,
    input:    InputState,
}

impl Drop for AppState {
    fn drop(&mut self) {
        if let Some(scene) = self.scene.take() {
            unsafe {
                let _ = self.renderer.device_wait_idle();
                scene.destroy(self.renderer.device());
            }
        }
    }
}

impl App {
    pub fn new(input_path: Option<InputPath>) -> Self {
        Self { input_path, state: None }
    }

    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let window = Arc::new(
            event_loop.create_window(
                winit::window::Window::default_attributes()
                    .with_title("lightbender")
                    .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
            )?,
        );
        let mut renderer = Renderer::new(window.clone())?;

        let mut camera = OrbitalCamera::new(Vec3::ZERO, 3.5, 30.0, 20.0);

        let scene = match &self.input_path {
            Some(InputPath::Model(path)) => {
                let ctx = renderer.load_context();
                match gltf_loader::load(&ctx, path) {
                    Ok(s) => { log::info!("Loaded: {}", path.display()); Some(s) }
                    Err(e) => { log::error!("Failed to load model: {e:#}"); None }
                }
            }
            Some(InputPath::Scene(path)) => {
                match loader::load_scene(&renderer, path) {
                    Ok(loaded) => {
                        // Apply camera from scene description
                        let cd = &loaded.description.camera;
                        camera = OrbitalCamera::new(
                            Vec3::from(cd.target),
                            cd.distance,
                            cd.yaw,
                            cd.pitch,
                        );
                        camera.camera.fov_y = f32::to_radians(cd.fov_y);
                        camera.camera.near  = cd.near;
                        camera.camera.far   = cd.far;

                        // Load named shader pipelines from scene description
                        let base = path.parent().unwrap_or(std::path::Path::new("."));
                        let mut lib = ShaderLibrary::new();
                        for (name, sd) in &loaded.description.shaders {
                            let vert_path = if std::path::Path::new(&sd.vert).is_absolute() {
                                std::path::PathBuf::from(&sd.vert)
                            } else {
                                base.join(&sd.vert)
                            };
                            let frag_path = if std::path::Path::new(&sd.frag).is_absolute() {
                                std::path::PathBuf::from(&sd.frag)
                            } else {
                                base.join(&sd.frag)
                            };
                            match unsafe { lib.load(renderer.device(), name, &vert_path, &frag_path) } {
                                Ok(()) => {
                                    let pair = lib.pairs.get(name).unwrap();
                                    if let Err(e) = renderer.add_pipeline(name, pair) {
                                        log::error!("Failed to build pipeline '{name}': {e:#}");
                                    } else {
                                        log::info!("Registered shader pipeline: {name}");
                                    }
                                }
                                Err(e) => log::error!("Failed to load shader '{name}': {e:#}"),
                            }
                        }
                        // Shader modules can be destroyed after pipeline creation
                        unsafe { lib.destroy(renderer.device()); }

                        // Load environment map if specified
                        if let Some(map_path) = &loaded.description.environment.map {
                            let resolved = if std::path::Path::new(map_path).is_absolute() {
                                std::path::PathBuf::from(map_path)
                            } else {
                                base.join(map_path)
                            };
                            match load_and_upload_env_map(&renderer, &resolved) {
                                Ok(tex) => {
                                    let intensity = loaded.description.environment.map_intensity;
                                    renderer.set_environment_map(tex, intensity);
                                    log::info!("Loaded environment map: {} (intensity={})", resolved.display(), intensity);
                                }
                                Err(e) => log::error!("Failed to load environment map: {e:#}"),
                            }
                        }

                        log::info!("Loaded scene: {}", path.display());
                        Some(loaded.scene)
                    }
                    Err(e) => { log::error!("Failed to load scene: {e:#}"); None }
                }
            }
            None => None,
        };

        let input = InputState::default();
        self.state = Some(AppState { window, scene, renderer, camera, input });
        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            if let Err(e) = self.init(event_loop) {
                log::error!("Failed to initialize: {e:#}");
                event_loop.exit();
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        let Some(state) = self.state.as_mut() else { return };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            state.input.accumulate_mouse_delta(dx, dy);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.physical_key
                    == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape)
                {
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    if let Err(e) = state.renderer.resize(size.width, size.height) {
                        log::error!("Resize failed: {e:#}");
                    }
                }
            }
            WindowEvent::MouseInput { button, state: btn_state, .. } => {
                let pressed = btn_state == winit::event::ElementState::Pressed;
                match button {
                    MouseButton::Left   => state.input.left_button   = pressed,
                    MouseButton::Right  => state.input.right_button  = pressed,
                    MouseButton::Middle => state.input.middle_button = pressed,
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y)  => y,
                    MouseScrollDelta::PixelDelta(pos)  => pos.y as f32 / 50.0,
                };
                state.input.accumulate_scroll(scroll);
            }
            WindowEvent::RedrawRequested => {
                state.camera.update(&state.input);
                state.input.flush();

                match state.renderer.draw_frame(&state.camera.camera, state.scene.as_ref()) {
                    Ok(_) => {}
                    Err(e) => {
                        log::error!("draw_frame failed: {e:#}");
                        event_loop.exit();
                    }
                }
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }
}

fn load_and_upload_env_map(
    renderer: &Renderer,
    path: &std::path::Path,
) -> anyhow::Result<GpuTexture> {
    use anyhow::Context;
    use ash::vk;

    let (width, height, pixels) = load_hdr_to_rgba32f(path)?;
    let ctx = renderer.load_context();

    let image = unsafe {
        GpuImage::upload_rgba32f(
            ctx.device, ctx.instance, ctx.physical_device,
            ctx.command_pool, ctx.queue,
            width, height, &pixels,
        ).context("upload environment map")?
    };

    let sampler = unsafe {
        ctx.device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
            None,
        ).context("create env map sampler")?
    };

    Ok(GpuTexture { image, sampler })
}
