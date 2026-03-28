use bytemuck::{Pod, Zeroable};

/// Per-vertex data uploaded to the GPU. Stride = 48 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuVertex {
    pub position: [f32; 3], // location 0, offset  0
    pub normal:   [f32; 3], // location 1, offset 12
    pub uv:       [f32; 2], // location 2, offset 24
    pub tangent:  [f32; 4], // location 3, offset 32  (w = bitangent sign)
}

/// A single light packed for the UBO.
/// Layout matches GLSL std140: struct alignment rounded to 16 → 64 bytes each.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct GpuLight {
    /// xyz = direction (w=0) or position (w=1 point, w=2 spot)
    pub position_or_direction: [f32; 4], // offset  0
    pub color:       [f32; 3],           // offset 16
    pub intensity:   f32,                // offset 28
    pub range:       f32,                // offset 32
    pub _pad0:       f32,                // offset 36  (align vec2 to 8)
    pub spot_angles: [f32; 2],           // offset 40
    pub _pad1:       [f32; 4],           // offset 48  (pad to 64)
}

pub const MAX_LIGHTS: usize = 8;

/// Per-frame uniform buffer (set 0, binding 0).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FrameUniforms {
    pub view:            [[f32; 4]; 4],
    pub projection:      [[f32; 4]; 4],
    pub camera_position: [f32; 4], // w unused
    pub lights:          [GpuLight; MAX_LIGHTS],
    pub light_count:     u32,
    pub env_intensity:   f32,
    pub _pad:            [u32; 2],
}
