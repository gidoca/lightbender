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
use crate::scene::{gltf_loader, Scene};

pub struct App {
    /// Optional path to a GLB/glTF file to load at startup.
    pub model_path: Option<PathBuf>,
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
        // Wait for GPU to be idle before destroying scene resources
        if let Some(scene) = self.scene.take() {
            unsafe {
                let _ = self.renderer.device_wait_idle();
                scene.destroy(self.renderer.device());
            }
        }
    }
}

impl App {
    pub fn new(model_path: Option<PathBuf>) -> Self {
        Self { model_path, state: None }
    }

    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let window = Arc::new(
            event_loop.create_window(
                winit::window::Window::default_attributes()
                    .with_title("lightbender")
                    .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
            )?,
        );
        let renderer = Renderer::new(window.clone())?;

        let scene = if let Some(path) = &self.model_path {
            let ctx = renderer.load_context();
            match gltf_loader::load(&ctx, path) {
                Ok(s) => {
                    log::info!("Loaded model: {}", path.display());
                    Some(s)
                }
                Err(e) => {
                    log::error!("Failed to load model: {e:#}");
                    None
                }
            }
        } else {
            None
        };

        let camera = OrbitalCamera::new(Vec3::ZERO, 3.5, 30.0, 20.0);
        let input  = InputState::default();
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
                    MouseScrollDelta::LineDelta(_, y)   => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 50.0,
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
