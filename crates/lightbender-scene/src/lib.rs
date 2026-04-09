mod tangent;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};

pub use tangent::compute_tangents;

// ── Vertex ──────────────────────────────────────────────────────────────────

/// Per-vertex data. Stride = 48 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3], // location 0, offset  0
    pub normal:   [f32; 3], // location 1, offset 12
    pub uv:       [f32; 2], // location 2, offset 24
    pub tangent:  [f32; 4], // location 3, offset 32  (w = bitangent sign)
}

// ── GPU uniform types (repr(C) / Pod for upload) ────────────────────────────

/// A single light packed for the UBO.
/// Layout matches GLSL std140: struct alignment rounded to 16 → 64 bytes each.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuLight {
    /// xyz = direction (w=0) or position (w=1 point, w=2 spot)
    pub position_or_direction: [f32; 4], // offset  0
    pub color:       [f32; 3],           // offset 16
    pub intensity:   f32,                // offset 28
    pub range:       f32,                // offset 32
    pub _pad0:       f32,                // offset 36  (align vec2 to 8)
    pub spot_angles: [f32; 2],           // offset 40
    /// Index into FrameUniforms::shadow_vp (-1 = no shadow).
    pub shadow_vp_index: i32,            // offset 48
    pub shadow_bias:     f32,            // offset 52
    /// World→UV scale of the light footprint at the shadow map's near plane,
    /// used by PCSS to drive penumbra width. Set to 0 to disable PCSS for this light.
    pub light_size_uv:   f32,            // offset 56
    pub _pad1:           f32,            // offset 60  (pad to 64)
}

impl Default for GpuLight {
    fn default() -> Self {
        Self {
            shadow_vp_index: -1,
            ..bytemuck::Zeroable::zeroed()
        }
    }
}

pub const MAX_LIGHTS: usize = 8;
pub const MAX_SHADOW_CASTERS: usize = 4;
pub const MAX_AREA_LIGHTS: usize = 4;

/// A rectangular area light, packed for the UBO.
/// std140 layout: 5×vec4 + vec4 = 96 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuAreaLight {
    /// Four world-space corners (xyz; w unused). Counter-clockwise when
    /// viewed from the lit side, so the rectangle normal is
    /// `normalize(cross(p1-p0, p3-p0))`.
    pub p0: [f32; 4],         // offset  0
    pub p1: [f32; 4],         // offset 16
    pub p2: [f32; 4],         // offset 32
    pub p3: [f32; 4],         // offset 48
    pub color:           [f32; 3], // offset 64
    pub intensity:       f32,      // offset 76
    pub shadow_vp_index: i32,      // offset 80 (-1 = no shadow)
    pub light_size_uv:   f32,      // offset 84 (PCSS kernel scale)
    pub _pad:            [f32; 2], // offset 88 (pad to 96)
}

impl Default for GpuAreaLight {
    fn default() -> Self {
        Self {
            shadow_vp_index: -1,
            ..bytemuck::Zeroable::zeroed()
        }
    }
}

/// Per-frame uniform buffer (set 0, binding 0).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FrameUniforms {
    pub view:             [[f32; 4]; 4],
    pub projection:       [[f32; 4]; 4],
    pub camera_position:  [f32; 4], // w unused
    pub lights:           [GpuLight; MAX_LIGHTS],
    pub light_count:      u32,
    pub env_intensity:    f32,
    pub shadow_count:     u32,
    pub area_light_count: u32,
    pub shadow_vp:        [[[f32; 4]; 4]; MAX_SHADOW_CASTERS],
    pub area_lights:      [GpuAreaLight; MAX_AREA_LIGHTS],
    pub inverse_projection: [[f32; 4]; 4],
    /// x = radius, y = bias, z = power, w = enable (> 0.5 = on)
    pub ssao_params:      [f32; 4],
    /// xy = screen width/height in pixels
    pub screen_size:      [f32; 4],
}

/// Material factors pushed alongside the model matrix (push constants).
/// Starts at offset 64 (after the mat4 model matrix).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MaterialPushConstants {
    pub base_color_factor: [f32; 4],  // offset 64
    pub emissive_factor:   [f32; 3],  // offset 80
    pub metallic_factor:   f32,       // offset 92
    pub roughness_factor:  f32,       // offset 96
    pub _pad:              [f32; 3],  // offset 100, pad to 112 total
}

// ── Texture data ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureFormat {
    Rgba8,
    Rgba32F,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterMode {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressMode {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
}

#[derive(Clone, Debug)]
pub struct SamplerDesc {
    pub mag_filter:     FilterMode,
    pub min_filter:     FilterMode,
    pub address_mode_u: AddressMode,
    pub address_mode_v: AddressMode,
    pub address_mode_w: AddressMode,
}

impl Default for SamplerDesc {
    fn default() -> Self {
        Self {
            mag_filter:     FilterMode::Linear,
            min_filter:     FilterMode::Linear,
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
        }
    }
}

/// CPU-side texture data: raw pixels plus a sampler description.
pub struct TextureData {
    pub width:   u32,
    pub height:  u32,
    pub format:  TextureFormat,
    pub pixels:  Vec<u8>,
    pub sampler: SamplerDesc,
}

// ── Material ────────────────────────────────────────────────────────────────

pub struct Material {
    pub base_color_factor:          [f32; 4],
    pub metallic_factor:            f32,
    pub roughness_factor:           f32,
    pub emissive_factor:            [f32; 3],
    pub base_color_texture:         Option<usize>,
    pub normal_texture:             Option<usize>,
    pub metallic_roughness_texture: Option<usize>,
    pub occlusion_texture:          Option<usize>,
    pub emissive_texture:           Option<usize>,
    pub double_sided:               bool,
    /// Named pipeline to use when rendering this material (None = "default").
    pub pipeline_name:              Option<String>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color_factor:          [1.0, 1.0, 1.0, 1.0],
            metallic_factor:            1.0,
            roughness_factor:           1.0,
            emissive_factor:            [0.0, 0.0, 0.0],
            base_color_texture:         None,
            normal_texture:             None,
            metallic_roughness_texture: None,
            occlusion_texture:          None,
            emissive_texture:           None,
            double_sided:               false,
            pipeline_name:              None,
        }
    }
}

// ── Mesh ────────────────────────────────────────────────────────────────────

pub struct Primitive {
    pub vertices: Vec<Vertex>,
    pub indices:  Vec<u32>,
    pub material: usize, // index into Scene::materials
}

pub struct Mesh {
    pub name:       String,
    pub primitives: Vec<Primitive>,
}

// ── Scene graph ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation:    Quat,
    pub scale:       Vec3,
}

impl Transform {
    pub fn to_mat4(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation:    Quat::IDENTITY,
            scale:       Vec3::ONE,
        }
    }
}

pub struct SceneNode {
    pub name:            String,
    pub local_transform: Transform,
    pub parent:          Option<usize>,
    pub children:        Vec<usize>,
    /// Index into `Scene::meshes`.
    pub mesh:            Option<usize>,
}

// ── Light (CPU-side description) ────────────────────────────────────────────

pub struct Light {
    /// xyz = direction (w=0) or position (w=1 point, w=2 spot)
    pub position_or_direction: [f32; 4],
    pub color:       [f32; 3],
    pub intensity:   f32,
    pub range:       f32,
    pub spot_angles: [f32; 2],
}

/// CPU-side description of a rectangular area light.
///
/// Corners are listed counter-clockwise as seen from the lit side, so the
/// rectangle's normal is `normalize(cross(corners[1]-corners[0], corners[3]-corners[0]))`.
#[derive(Clone, Debug)]
pub struct AreaLight {
    pub corners:   [Vec3; 4],
    pub color:     [f32; 3],
    pub intensity: f32,
}

impl AreaLight {
    /// Convert to the GPU-packed representation. The shadow VP index is
    /// initialised to -1; the renderer assigns slots during UBO upload.
    pub fn to_gpu(&self) -> GpuAreaLight {
        let pad = |v: Vec3| [v.x, v.y, v.z, 0.0];
        GpuAreaLight {
            p0: pad(self.corners[0]),
            p1: pad(self.corners[1]),
            p2: pad(self.corners[2]),
            p3: pad(self.corners[3]),
            color: self.color,
            intensity: self.intensity,
            shadow_vp_index: -1,
            light_size_uv: 0.0,
            _pad: [0.0; 2],
        }
    }
}

impl Light {
    /// Convert to the GPU-packed representation.
    pub fn to_gpu(&self) -> GpuLight {
        GpuLight {
            position_or_direction: self.position_or_direction,
            color:       self.color,
            intensity:   self.intensity,
            range:       self.range,
            _pad0:       0.0,
            spot_angles: self.spot_angles,
            shadow_vp_index: -1,
            shadow_bias:     0.005,
            light_size_uv:   0.0,
            _pad1:           0.0,
        }
    }
}

// ── Scene ───────────────────────────────────────────────────────────────────

pub struct Scene {
    pub nodes:     Vec<SceneNode>,
    pub meshes:    Vec<Mesh>,
    pub materials: Vec<Material>,
    pub textures:  Vec<TextureData>,

    /// Cached flat world transforms, parallel to `nodes`.
    pub world_transforms: Vec<Mat4>,
}

impl Scene {
    /// Create an empty scene.
    pub fn new() -> Self {
        Self {
            nodes:            Vec::new(),
            meshes:           Vec::new(),
            materials:        Vec::new(),
            textures:         Vec::new(),
            world_transforms: Vec::new(),
        }
    }

    /// Recompute world transforms for all nodes (parent-before-child order).
    pub fn update_world_transforms(&mut self) {
        self.world_transforms.resize(self.nodes.len(), Mat4::IDENTITY);
        for i in 0..self.nodes.len() {
            let local = self.nodes[i].local_transform.to_mat4();
            self.world_transforms[i] = match self.nodes[i].parent {
                Some(p) => self.world_transforms[p] * local,
                None    => local,
            };
        }
    }

    /// Iterator over (node_index, mesh_index) for all nodes that have a mesh.
    pub fn mesh_nodes(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.nodes.iter().enumerate().filter_map(|(i, node)| {
            node.mesh.map(|mesh_idx| (i, mesh_idx))
        })
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scene(nodes: Vec<SceneNode>) -> Scene {
        Scene {
            nodes,
            meshes: vec![],
            materials: vec![],
            textures: vec![],
            world_transforms: vec![],
        }
    }

    fn node(local: Transform, parent: Option<usize>) -> SceneNode {
        SceneNode {
            name: String::new(),
            local_transform: local,
            parent,
            children: vec![],
            mesh: None,
        }
    }

    fn translation(x: f32, y: f32, z: f32) -> Transform {
        Transform {
            translation: Vec3::new(x, y, z),
            ..Transform::default()
        }
    }

    // ── Transform::to_mat4 ──────────────────────────────────────────────────

    #[test]
    fn transform_identity_is_identity_matrix() {
        let m = Transform::default().to_mat4();
        assert!((m - Mat4::IDENTITY).abs_diff_eq(Mat4::ZERO, 1e-6));
    }

    #[test]
    fn transform_translation_only() {
        let t = translation(1.0, 2.0, 3.0);
        let m = t.to_mat4();
        let translated = m * glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
        assert!((translated.x - 1.0).abs() < 1e-6);
        assert!((translated.y - 2.0).abs() < 1e-6);
        assert!((translated.z - 3.0).abs() < 1e-6);
    }

    #[test]
    fn transform_scale_only() {
        let t = Transform {
            scale: Vec3::splat(2.0),
            ..Transform::default()
        };
        let m = t.to_mat4();
        let scaled = m * glam::Vec4::new(1.0, 1.0, 1.0, 1.0);
        assert!((scaled.x - 2.0).abs() < 1e-6);
        assert!((scaled.y - 2.0).abs() < 1e-6);
        assert!((scaled.z - 2.0).abs() < 1e-6);
    }

    #[test]
    fn transform_rotation_90_deg_y() {
        // 90° around Y: (1,0,0) → (0,0,-1)
        let t = Transform {
            rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            ..Transform::default()
        };
        let m = t.to_mat4();
        let rotated = m * glam::Vec4::new(1.0, 0.0, 0.0, 1.0);
        assert!(rotated.x.abs() < 1e-5, "x={}", rotated.x);
        assert!((rotated.z + 1.0).abs() < 1e-5, "z={}", rotated.z);
    }

    // ── Scene::update_world_transforms ──────────────────────────────────────

    #[test]
    fn update_world_transforms_single_root() {
        let mut scene = make_scene(vec![node(translation(1.0, 0.0, 0.0), None)]);
        scene.update_world_transforms();
        let wt = scene.world_transforms[0];
        let p = wt * glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
        assert!((p.x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn update_world_transforms_parent_child() {
        // parent at (1,0,0), child at (0,1,0) local → child world = (1,1,0)
        let mut scene = make_scene(vec![
            node(translation(1.0, 0.0, 0.0), None),
            node(translation(0.0, 1.0, 0.0), Some(0)),
        ]);
        scene.update_world_transforms();
        let child_world = scene.world_transforms[1];
        let p = child_world * glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
        assert!((p.x - 1.0).abs() < 1e-6, "x={}", p.x);
        assert!((p.y - 1.0).abs() < 1e-6, "y={}", p.y);
        assert!(p.z.abs() < 1e-6, "z={}", p.z);
    }

    #[test]
    fn update_world_transforms_deep_chain() {
        // grandparent (1,0,0) → parent (0,1,0) → child (0,0,1) → world (1,1,1)
        let mut scene = make_scene(vec![
            node(translation(1.0, 0.0, 0.0), None),
            node(translation(0.0, 1.0, 0.0), Some(0)),
            node(translation(0.0, 0.0, 1.0), Some(1)),
        ]);
        scene.update_world_transforms();
        let child_world = scene.world_transforms[2];
        let p = child_world * glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
        assert!((p.x - 1.0).abs() < 1e-6, "x={}", p.x);
        assert!((p.y - 1.0).abs() < 1e-6, "y={}", p.y);
        assert!((p.z - 1.0).abs() < 1e-6, "z={}", p.z);
    }
}
