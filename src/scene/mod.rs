pub mod gltf_loader;
pub mod loader;

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
