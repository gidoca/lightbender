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

use lightbender_interaction::{InputState, OrbitalCamera};
use lightbender_object_loaders::{load_gltf, load_image_hdr};
use lightbender_renderer::{load_spirv, Renderer};
use lightbender_scene::Light;
use lightbender_scene_loaders::{load_json_scene, load_mitsuba, LightDesc};

pub enum InputPath {
    Model(PathBuf),
    Scene(PathBuf),
    MitsubaScene(PathBuf),
}

pub struct App {
    pub input_path: Option<InputPath>,
    pub output_path: Option<PathBuf>,
    state: Option<AppState>,
}

struct AppState {
    window:   Arc<Window>,
    renderer: Renderer,
    camera:   OrbitalCamera,
    input:    InputState,
}

impl App {
    pub fn new(input_path: Option<InputPath>, output_path: Option<PathBuf>) -> Self {
        Self { input_path, output_path, state: None }
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
        let camera = load_scene_into_renderer(&mut renderer, self.input_path.as_ref())?;

        let input = InputState::default();
        self.state = Some(AppState { window, renderer, camera, input });
        Ok(())
    }
}

/// Load a scene/model from the given input path, setting up camera, shaders,
/// and environment maps on the renderer. Used by both windowed and headless paths.
pub fn load_scene_into_renderer(
    renderer: &mut Renderer,
    input_path: Option<&InputPath>,
) -> Result<OrbitalCamera> {
    let mut camera = OrbitalCamera::new(Vec3::ZERO, 3.5, 30.0, 20.0);

    match input_path {
        Some(InputPath::Model(path)) => {
            match load_gltf(path) {
                Ok(scene) => {
                    renderer.upload_scene(&scene)?;
                    log::info!("Loaded: {}", path.display());
                }
                Err(e) => log::error!("Failed to load model: {e:#}"),
            }
        }
        Some(InputPath::Scene(path)) => {
            match load_json_scene(path) {
                Ok(loaded) => {
                    // Upload the scene
                    renderer.upload_scene(&loaded.scene)?;

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
                        match (load_spirv(&vert_path), load_spirv(&frag_path)) {
                            (Ok(vert_spv), Ok(frag_spv)) => {
                                if let Err(e) = renderer.add_pipeline_spirv(name, &vert_spv, &frag_spv) {
                                    log::error!("Failed to build pipeline '{name}': {e:#}");
                                } else {
                                    log::info!("Registered shader pipeline: {name}");
                                }
                            }
                            (Err(e), _) | (_, Err(e)) => {
                                log::error!("Failed to load shader '{name}': {e:#}");
                            }
                        }
                    }

                    // Load environment map if specified
                    if let Some(map_path) = &loaded.description.environment.map {
                        let resolved = if std::path::Path::new(map_path).is_absolute() {
                            std::path::PathBuf::from(map_path)
                        } else {
                            base.join(map_path)
                        };
                        match load_image_hdr(&resolved) {
                            Ok(tex_data) => {
                                let intensity = loaded.description.environment.map_intensity;
                                if let Err(e) = renderer.set_environment_map(&tex_data, intensity) {
                                    log::error!("Failed to set env map: {e:#}");
                                } else {
                                    log::info!("Loaded environment map: {} (intensity={})", resolved.display(), intensity);
                                }
                            }
                            Err(e) => log::error!("Failed to load environment map: {e:#}"),
                        }
                    }

                    // Convert JSON lights to scene lights
                    let lights: Vec<Light> = loaded.description.lights.iter().map(|l| {
                        match l {
                            LightDesc::Directional { direction, color, intensity, .. } => {
                                Light {
                                    position_or_direction: [direction[0], direction[1], direction[2], 0.0],
                                    color: *color,
                                    intensity: *intensity,
                                    range: 0.0,
                                    spot_angles: [0.0, 0.0],
                                }
                            }
                            LightDesc::Point { position, color, intensity, range, .. } => {
                                Light {
                                    position_or_direction: [position[0], position[1], position[2], 1.0],
                                    color: *color,
                                    intensity: *intensity,
                                    range: *range,
                                    spot_angles: [0.0, 0.0],
                                }
                            }
                            LightDesc::Spot { position, color, intensity, range, inner_cone_angle, outer_cone_angle, .. } => {
                                Light {
                                    position_or_direction: [position[0], position[1], position[2], 2.0],
                                    color: *color,
                                    intensity: *intensity,
                                    range: *range,
                                    spot_angles: [inner_cone_angle.to_radians().cos(), outer_cone_angle.to_radians().cos()],
                                }
                            }
                        }
                    }).collect();
                    if !lights.is_empty() {
                        log::info!("Setting {} scene lights", lights.len());
                        renderer.set_lights(lights);
                    }

                    log::info!("Loaded scene: {}", path.display());
                }
                Err(e) => log::error!("Failed to load scene: {e:#}"),
            }
        }
        Some(InputPath::MitsubaScene(path)) => {
            match load_mitsuba(path) {
                Ok(loaded) => {
                    // Upload the scene
                    renderer.upload_scene(&loaded.scene)?;

                    // Apply camera
                    let cp = &loaded.camera;
                    camera = OrbitalCamera::new(
                        cp.target, cp.distance, cp.yaw_deg, cp.pitch_deg,
                    );
                    camera.camera.fov_y = cp.fov_y;
                    camera.camera.near  = cp.near;
                    camera.camera.far   = cp.far;

                    // Load environment map if present
                    if let Some(ref map_path) = loaded.env_map {
                        let resolved = std::path::PathBuf::from(map_path);
                        match load_image_hdr(&resolved) {
                            Ok(tex_data) => {
                                if let Err(e) = renderer.set_environment_map(&tex_data, loaded.env_scale) {
                                    log::error!("Failed to set env map: {e:#}");
                                } else {
                                    log::info!("Loaded environment map: {} (intensity={})", resolved.display(), loaded.env_scale);
                                }
                            }
                            Err(e) => log::error!("Failed to load environment map: {e:#}"),
                        }
                    }

                    // Set scene lights
                    if !loaded.lights.is_empty() {
                        log::info!("Setting {} scene lights", loaded.lights.len());
                        renderer.set_lights(loaded.lights);
                    }
                    if !loaded.area_lights.is_empty() {
                        log::info!("Setting {} area lights", loaded.area_lights.len());
                        renderer.set_area_lights(loaded.area_lights);
                    }

                    log::info!("Loaded Mitsuba scene: {}", path.display());
                }
                Err(e) => log::error!("Failed to load Mitsuba scene: {e:#}"),
            }
        }
        None => {}
    }

    Ok(camera)
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(clippy::collapsible_if)]
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
                use winit::event::ElementState;
                use winit::keyboard::{KeyCode, PhysicalKey};
                let pressed = event.state == ElementState::Pressed;
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                    PhysicalKey::Code(KeyCode::KeyW) => state.input.key_w = pressed,
                    PhysicalKey::Code(KeyCode::KeyA) => state.input.key_a = pressed,
                    PhysicalKey::Code(KeyCode::KeyS) => state.input.key_s = pressed,
                    PhysicalKey::Code(KeyCode::KeyD) => state.input.key_d = pressed,
                    _ => {}
                }
            }
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    #[allow(clippy::collapsible_if)]
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

                match state.renderer.draw_frame(&state.camera.camera) {
                    Ok(_) => {}
                    Err(e) => {
                        log::error!("draw_frame failed: {e:#}");
                        event_loop.exit();
                        return;
                    }
                }

                if let Some(path) = &self.output_path {
                    match state.renderer.capture_frame_to_file(path) {
                        Ok(()) => log::info!("Saved output to {}", path.display()),
                        Err(e) => log::error!("Failed to save output: {e:#}"),
                    }
                    event_loop.exit();
                    return;
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
