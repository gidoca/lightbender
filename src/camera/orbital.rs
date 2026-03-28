use std::f32::consts::FRAC_PI_2;

use glam::Vec3;

use super::Camera;
use crate::input::InputState;

pub struct OrbitalCamera {
    pub camera: Camera,
    pub target:   Vec3,
    pub yaw:      f32, // radians, horizontal
    pub pitch:    f32, // radians, vertical (clamped away from poles)
    pub distance: f32,

    orbit_sensitivity: f32,
    pan_sensitivity:   f32,
    zoom_sensitivity:  f32,
}

impl OrbitalCamera {
    pub fn new(target: Vec3, distance: f32, yaw_deg: f32, pitch_deg: f32) -> Self {
        let mut cam = Self {
            camera: Camera {
                position: Vec3::ZERO,
                target,
                fov_y: f32::to_radians(60.0),
                near: 0.01,
                far: 1000.0,
            },
            target,
            yaw:      f32::to_radians(yaw_deg),
            pitch:    f32::to_radians(pitch_deg),
            distance,
            orbit_sensitivity: 0.005,
            pan_sensitivity:   0.001,
            zoom_sensitivity:  0.1,
        };
        cam.sync();
        cam
    }

    pub fn update(&mut self, input: &InputState) {
        let (dx, dy) = (input.mouse_delta.0 as f32, input.mouse_delta.1 as f32);

        if input.left_button {
            self.orbit(dx, dy);
        } else if input.middle_button || input.right_button {
            self.pan(dx, dy);
        }

        if input.scroll_delta != 0.0 {
            self.zoom(input.scroll_delta);
        }
    }

    fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * self.orbit_sensitivity;
        self.pitch = (self.pitch + dy * self.orbit_sensitivity)
            .clamp(-FRAC_PI_2 + 0.01, FRAC_PI_2 - 0.01);
        self.sync();
    }

    fn pan(&mut self, dx: f32, dy: f32) {
        let forward = (self.target - self.camera.position).normalize();
        let right   = forward.cross(Vec3::Y).normalize();
        let up      = right.cross(forward).normalize();
        let scale   = self.distance * self.pan_sensitivity;
        self.target -= right * dx * scale;
        self.target += up    * dy * scale;
        self.camera.target = self.target;
        self.sync();
    }

    fn zoom(&mut self, scroll: f32) {
        self.distance = (self.distance * (1.0 - scroll * self.zoom_sensitivity)).max(0.001);
        self.sync();
    }

    /// Recompute camera position from spherical coordinates.
    fn sync(&mut self) {
        let x = self.distance * self.pitch.cos() * self.yaw.sin();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.pitch.cos() * self.yaw.cos();
        self.camera.position = self.target + Vec3::new(x, y, z);
        self.camera.target   = self.target;
    }
}
