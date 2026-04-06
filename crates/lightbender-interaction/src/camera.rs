use glam::{Mat4, Vec3};

pub struct Camera {
    pub position: Vec3,
    pub target:   Vec3,
    pub fov_y:    f32, // radians
    pub near:     f32,
    pub far:      f32,
}

impl Camera {
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, Vec3::Y)
    }

    /// Returns a Vulkan-compatible projection matrix (Y flipped).
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        let proj = Mat4::perspective_rh(self.fov_y, aspect, self.near, self.far);
        // Flip Y column so NDC Y points down (Vulkan convention)
        Mat4::from_cols(proj.col(0), -proj.col(1), proj.col(2), proj.col(3))
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::FRAC_PI_2;

    use glam::Vec4;

    use super::*;

    fn camera_at_z(z: f32) -> Camera {
        Camera {
            position: Vec3::new(0.0, 0.0, z),
            target: Vec3::ZERO,
            fov_y: FRAC_PI_2,
            near: 0.1,
            far: 100.0,
        }
    }

    #[test]
    fn view_matrix_camera_on_z_axis() {
        let cam = camera_at_z(5.0);
        let view = cam.view_matrix();
        // World origin should map to (0, 0, -5) in view space (behind camera)
        let origin_view = view * Vec4::new(0.0, 0.0, 0.0, 1.0);
        assert!((origin_view.z + 5.0).abs() < 1e-5, "z={}", origin_view.z);
        assert!(origin_view.x.abs() < 1e-5);
        assert!(origin_view.y.abs() < 1e-5);
    }

    #[test]
    fn projection_matrix_y_is_flipped() {
        let cam = camera_at_z(1.0);
        let standard = Mat4::perspective_rh(cam.fov_y, 1.0, cam.near, cam.far);
        let vulkan = cam.projection_matrix(1.0);
        // Y column should be negated
        let diff = vulkan.col(1) + standard.col(1);
        assert!(diff.length() < 1e-5, "Y column not flipped: {:?}", diff);
        // X and Z columns unchanged
        assert!((vulkan.col(0) - standard.col(0)).length() < 1e-5);
        assert!((vulkan.col(2) - standard.col(2)).length() < 1e-5);
    }

    #[test]
    fn projection_matrix_aspect_affects_x_scale() {
        let cam = camera_at_z(1.0);
        let wide = cam.projection_matrix(2.0);
        let square = cam.projection_matrix(1.0);
        // Wider aspect → smaller X scale (objects appear narrower in NDC)
        assert!(wide.col(0).x < square.col(0).x);
    }
}
