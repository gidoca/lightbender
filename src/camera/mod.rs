mod orbital;
pub use orbital::OrbitalCamera;

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
