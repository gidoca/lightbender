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

#[cfg(test)]
mod tests {
    use std::f32::consts::FRAC_PI_2;

    use super::*;

    #[test]
    fn new_positions_camera_at_positive_z() {
        // yaw=0, pitch=0, distance=5 → camera at (0, 0, 5) relative to target
        let cam = OrbitalCamera::new(Vec3::ZERO, 5.0, 0.0, 0.0);
        let pos = cam.camera.position;
        assert!((pos.x).abs() < 1e-5, "x={}", pos.x);
        assert!((pos.y).abs() < 1e-5, "y={}", pos.y);
        assert!((pos.z - 5.0).abs() < 1e-5, "z={}", pos.z);
    }

    #[test]
    fn new_with_nonzero_target() {
        let target = Vec3::new(1.0, 2.0, 3.0);
        let cam = OrbitalCamera::new(target, 5.0, 0.0, 0.0);
        let expected = target + Vec3::new(0.0, 0.0, 5.0);
        assert!((cam.camera.position - expected).length() < 1e-5);
    }

    #[test]
    fn orbit_updates_yaw_and_pitch() {
        let mut cam = OrbitalCamera::new(Vec3::ZERO, 5.0, 0.0, 0.0);
        let initial_yaw = cam.yaw;
        let initial_pitch = cam.pitch;
        cam.orbit(100.0, 50.0);
        assert!((cam.yaw - initial_yaw).abs() > 0.01);
        assert!((cam.pitch - initial_pitch).abs() > 0.01);
    }

    #[test]
    fn orbit_clamps_pitch() {
        let mut cam = OrbitalCamera::new(Vec3::ZERO, 5.0, 0.0, 0.0);
        // Large positive dy drives pitch toward +π/2
        cam.orbit(0.0, 1_000_000.0);
        assert!(cam.pitch < FRAC_PI_2);
        assert!(cam.pitch > FRAC_PI_2 - 0.1);

        // Large negative dy drives pitch toward -π/2
        cam.orbit(0.0, -1_000_000.0);
        assert!(cam.pitch > -FRAC_PI_2);
        assert!(cam.pitch < -FRAC_PI_2 + 0.1);
    }

    #[test]
    fn zoom_reduces_distance() {
        let mut cam = OrbitalCamera::new(Vec3::ZERO, 10.0, 0.0, 0.0);
        cam.zoom(1.0); // positive scroll = zoom in
        assert!(cam.distance < 10.0);
    }

    #[test]
    fn zoom_minimum_distance_enforced() {
        let mut cam = OrbitalCamera::new(Vec3::ZERO, 1.0, 0.0, 0.0);
        // Extreme zoom in should not go below 0.001
        for _ in 0..1000 {
            cam.zoom(100.0);
        }
        assert!(cam.distance >= 0.001);
    }

    #[test]
    fn sync_spherical_to_cartesian() {
        let mut cam = OrbitalCamera::new(Vec3::ZERO, 3.0, 0.0, 0.0);
        // yaw=0, pitch=0, distance=3 → position=(0, 0, 3)
        cam.yaw = 0.0;
        cam.pitch = 0.0;
        cam.distance = 3.0;
        cam.sync();
        assert!((cam.camera.position.x).abs() < 1e-5);
        assert!((cam.camera.position.y).abs() < 1e-5);
        assert!((cam.camera.position.z - 3.0).abs() < 1e-5);
    }
}
