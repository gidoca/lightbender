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

pub struct App {
    state: Option<AppState>,
}

struct AppState {
    window:   Arc<Window>,
    renderer: Renderer,
    camera:   OrbitalCamera,
    input:    InputState,
}

impl App {
    pub fn new() -> Self {
        Self { state: None }
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
        let camera   = OrbitalCamera::new(Vec3::ZERO, 3.5, 30.0, 20.0);
        let input    = InputState::default();
        self.state = Some(AppState { window, renderer, camera, input });
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

                match state.renderer.draw_frame(&state.camera.camera) {
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
