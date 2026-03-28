use std::sync::Arc;

use anyhow::Result;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

use crate::renderer::Renderer;

pub struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    renderer: Renderer,
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
        self.state = Some(AppState { window, renderer });
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

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
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
            WindowEvent::RedrawRequested => {
                match state.renderer.draw_frame() {
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
