pub mod gltf_loader;
pub mod loader;
pub mod mitsuba_loader;

use ash::vk;
use glam::{Mat4, Quat, Vec3};

use crate::vulkan::{buffer::GpuBuffer, image::GpuImage};

// ── Transform ────────────────────────────────────────────────────────────────

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

// ── Scene nodes ──────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub struct SceneNode {
    pub name:            String,
    pub local_transform: Transform,
    pub parent:          Option<usize>,
    pub children:        Vec<usize>,
    /// Index into `Scene::meshes`.
    pub mesh:            Option<usize>,
}

// ── GPU resources ────────────────────────────────────────────────────────────

pub struct GpuTexture {
    pub image:   GpuImage,
    pub sampler: vk::Sampler,
}

impl GpuTexture {
    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            self.image.destroy(device);
            device.destroy_sampler(self.sampler, None);
        }
    }
}

#[allow(dead_code)]
pub struct GpuMaterial {
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
    /// Descriptor set (set 1) for this material's textures.
    pub descriptor_set:             vk::DescriptorSet,
}

pub struct GpuPrimitive {
    pub vertex_buffer: GpuBuffer,
    pub index_buffer:  GpuBuffer,
    pub index_count:   u32,
    pub material:      usize, // index into Scene::materials
}

impl GpuPrimitive {
    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            self.vertex_buffer.destroy(device);
            self.index_buffer.destroy(device);
        }
    }
}

#[allow(dead_code)]
pub struct GpuMesh {
    pub name:       String,
    pub primitives: Vec<GpuPrimitive>,
}

impl GpuMesh {
    pub unsafe fn destroy(&self, device: &ash::Device) {
        for p in &self.primitives {
            unsafe { p.destroy(device); }
        }
    }
}

// ── Scene ────────────────────────────────────────────────────────────────────

pub struct Scene {
    pub nodes:     Vec<SceneNode>,
    pub meshes:    Vec<GpuMesh>,
    pub materials: Vec<GpuMaterial>,
    pub textures:  Vec<GpuTexture>,

    /// Cached flat world transforms, parallel to `nodes`.
    pub world_transforms: Vec<Mat4>,

    /// Descriptor pool owning all material descriptor sets.
    pub descriptor_pool: vk::DescriptorPool,
}

impl Scene {
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

    /// Iterator over (world_transform, primitive) for all mesh primitives in the scene.
    pub fn draw_primitives(&self) -> impl Iterator<Item = (Mat4, &GpuPrimitive)> {
        self.nodes.iter().enumerate().flat_map(|(i, node)| {
            node.mesh.map(|mesh_idx| {
                let world = self.world_transforms[i];
                self.meshes[mesh_idx].primitives.iter().map(move |prim| (world, prim))
            })
            .into_iter()
            .flatten()
        })
    }

    pub unsafe fn destroy(&self, device: &ash::Device) {
        for mesh in &self.meshes {
            unsafe { mesh.destroy(device); }
        }
        for tex in &self.textures {
            unsafe { tex.destroy(device); }
        }
        unsafe { device.destroy_descriptor_pool(self.descriptor_pool, None); }
    }
}

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3};

    use super::*;

    fn make_scene(nodes: Vec<SceneNode>) -> Scene {
        Scene {
            nodes,
            meshes: vec![],
            materials: vec![],
            textures: vec![],
            world_transforms: vec![],
            descriptor_pool: vk::DescriptorPool::null(),
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

    // ── Transform::to_mat4 ────────────────────────────────────────────────────

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

    // ── Scene::update_world_transforms ───────────────────────────────────────

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
