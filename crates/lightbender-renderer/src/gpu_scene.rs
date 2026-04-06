#![allow(unsafe_op_in_unsafe_fn)]

use anyhow::{Context, Result};
use ash::vk;
use glam::Mat4;

use lightbender_scene::{
    AddressMode, FilterMode, Material, Scene, TextureData, TextureFormat, Vertex,
};

use crate::buffer::upload_to_device_local;
use crate::image::GpuImage;

// ── Internal GPU types ──────────────────────────────────────────────────────

pub(crate) struct GpuTexture {
    pub image:   GpuImage,
    pub sampler: vk::Sampler,
}

impl GpuTexture {
    pub unsafe fn destroy(&self, device: &ash::Device) {
        self.image.destroy(device);
        device.destroy_sampler(self.sampler, None);
    }
}

pub(crate) struct GpuMaterial {
    pub base_color_factor:          [f32; 4],
    pub metallic_factor:            f32,
    pub roughness_factor:           f32,
    pub emissive_factor:            [f32; 3],
    pub double_sided:               bool,
    pub pipeline_name:              Option<String>,
    pub descriptor_set:             vk::DescriptorSet,
}

pub(crate) struct GpuPrimitive {
    pub vertex_buffer: crate::buffer::GpuBuffer,
    pub index_buffer:  crate::buffer::GpuBuffer,
    pub index_count:   u32,
    pub material:      usize,
}

impl GpuPrimitive {
    pub unsafe fn destroy(&self, device: &ash::Device) {
        self.vertex_buffer.destroy(device);
        self.index_buffer.destroy(device);
    }
}

pub(crate) struct GpuMesh {
    pub primitives: Vec<GpuPrimitive>,
}

impl GpuMesh {
    pub unsafe fn destroy(&self, device: &ash::Device) {
        for p in &self.primitives {
            p.destroy(device);
        }
    }
}

// ── GpuScene ────────────────────────────────────────────────────────────────

pub(crate) struct GpuScene {
    pub meshes:           Vec<GpuMesh>,
    pub materials:        Vec<GpuMaterial>,
    pub textures:         Vec<GpuTexture>,
    pub world_transforms: Vec<Mat4>,
    /// (node_index, mesh_index) for nodes that have a mesh
    pub mesh_nodes:       Vec<(usize, usize)>,
    pub descriptor_pool:  vk::DescriptorPool,
    /// Placeholder textures owned by this scene (destroyed with it)
    placeholder_textures: Vec<GpuTexture>,
}

impl GpuScene {
    /// Upload a CPU-side `Scene` to the GPU.
    pub unsafe fn upload(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        material_set_layout: vk::DescriptorSetLayout,
        scene: &Scene,
    ) -> Result<Self> {
        // ── Textures ────────────────────────────────────────────────────────
        let mut textures: Vec<GpuTexture> = Vec::with_capacity(scene.textures.len());
        for tex_data in &scene.textures {
            let gpu_tex = upload_texture(
                device, instance, physical_device, command_pool, queue, tex_data,
            )?;
            textures.push(gpu_tex);
        }

        // ── Placeholder textures ────────────────────────────────────────────
        let make_sampler = || -> Result<vk::Sampler> {
            Ok(device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT)
                    .address_mode_w(vk::SamplerAddressMode::REPEAT),
                None,
            )?)
        };
        let make_srgb_placeholder = |pixels: [u8; 4]| -> Result<GpuTexture> {
            let img = GpuImage::upload_rgba8(
                device, instance, physical_device, command_pool, queue, 1, 1, &pixels,
            )?;
            Ok(GpuTexture { image: img, sampler: make_sampler()? })
        };
        let make_linear_placeholder = |pixels: [u8; 4]| -> Result<GpuTexture> {
            let img = GpuImage::upload_rgba8_unorm(
                device, instance, physical_device, command_pool, queue, 1, 1, &pixels,
            )?;
            Ok(GpuTexture { image: img, sampler: make_sampler()? })
        };
        // Base color and emissive are sRGB; normal, metallic-roughness, occlusion are linear
        let white_tex       = make_srgb_placeholder([255, 255, 255, 255])?;
        let flat_normal_tex = make_linear_placeholder([128, 128, 255, 255])?;
        let mr_placeholder  = make_linear_placeholder([0, 128, 0, 255])?;
        let black_tex       = make_srgb_placeholder([0, 0, 0, 255])?;

        // ── Descriptor pool for materials ───────────────────────────────────
        let mat_count = scene.materials.len().max(1) as u32;
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: mat_count * 5,
        }];
        let descriptor_pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(mat_count),
            None,
        ).context("scene descriptor pool")?;

        // ── Materials ───────────────────────────────────────────────────────
        let mut materials: Vec<GpuMaterial> = Vec::with_capacity(scene.materials.len());
        for mat in &scene.materials {
            let ds = allocate_material_set(device, descriptor_pool, material_set_layout)?;
            write_material_descriptors(
                device, ds, mat, &textures,
                &white_tex, &flat_normal_tex, &mr_placeholder, &black_tex,
            );
            materials.push(GpuMaterial {
                base_color_factor:  mat.base_color_factor,
                metallic_factor:    mat.metallic_factor,
                roughness_factor:   mat.roughness_factor,
                emissive_factor:    mat.emissive_factor,
                double_sided:       mat.double_sided,
                pipeline_name:      mat.pipeline_name.clone(),
                descriptor_set:     ds,
            });
        }

        // If no materials, create a default one
        if materials.is_empty() {
            let ds = allocate_material_set(device, descriptor_pool, material_set_layout)?;
            let default_mat = Material::default();
            write_material_descriptors(
                device, ds, &default_mat, &textures,
                &white_tex, &flat_normal_tex, &mr_placeholder, &black_tex,
            );
            materials.push(GpuMaterial {
                base_color_factor:  default_mat.base_color_factor,
                metallic_factor:    default_mat.metallic_factor,
                roughness_factor:   default_mat.roughness_factor,
                emissive_factor:    default_mat.emissive_factor,
                double_sided:       false,
                pipeline_name:      None,
                descriptor_set:     ds,
            });
        }

        // ── Meshes ──────────────────────────────────────────────────────────
        let default_mat_idx = 0;
        let mut meshes: Vec<GpuMesh> = Vec::with_capacity(scene.meshes.len());
        for mesh in &scene.meshes {
            let mut gpu_prims = Vec::with_capacity(mesh.primitives.len());
            for prim in &mesh.primitives {
                let vertex_buffer = upload_to_device_local(
                    device, instance, physical_device, command_pool, queue,
                    vk::BufferUsageFlags::VERTEX_BUFFER, &prim.vertices,
                )?;
                let index_buffer = upload_to_device_local(
                    device, instance, physical_device, command_pool, queue,
                    vk::BufferUsageFlags::INDEX_BUFFER, &prim.indices,
                )?;
                let mat_idx = if prim.material < materials.len() {
                    prim.material
                } else {
                    default_mat_idx
                };
                gpu_prims.push(GpuPrimitive {
                    vertex_buffer,
                    index_buffer,
                    index_count: prim.indices.len() as u32,
                    material: mat_idx,
                });
            }
            meshes.push(GpuMesh { primitives: gpu_prims });
        }

        // ── World transforms + mesh node list ───────────────────────────────
        let world_transforms = scene.world_transforms.clone();
        let mesh_nodes: Vec<(usize, usize)> = scene.mesh_nodes().collect();

        let placeholder_textures = vec![white_tex, flat_normal_tex, mr_placeholder, black_tex];

        Ok(Self {
            meshes,
            materials,
            textures,
            world_transforms,
            mesh_nodes,
            descriptor_pool,
            placeholder_textures,
        })
    }

    /// Iterator over (world_transform, &GpuPrimitive) for all primitives in draw order.
    pub fn draw_primitives(&self) -> impl Iterator<Item = (Mat4, &GpuPrimitive)> {
        self.mesh_nodes.iter().flat_map(|&(node_idx, mesh_idx)| {
            let world = self.world_transforms[node_idx];
            self.meshes[mesh_idx].primitives.iter().map(move |prim| (world, prim))
        })
    }

    pub unsafe fn destroy(&self, device: &ash::Device) {
        for mesh in &self.meshes {
            mesh.destroy(device);
        }
        for tex in &self.textures {
            tex.destroy(device);
        }
        for tex in &self.placeholder_textures {
            tex.destroy(device);
        }
        device.destroy_descriptor_pool(self.descriptor_pool, None);
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn map_filter(f: FilterMode) -> vk::Filter {
    match f {
        FilterMode::Nearest => vk::Filter::NEAREST,
        FilterMode::Linear  => vk::Filter::LINEAR,
    }
}

fn map_address(a: AddressMode) -> vk::SamplerAddressMode {
    match a {
        AddressMode::Repeat         => vk::SamplerAddressMode::REPEAT,
        AddressMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        AddressMode::ClampToEdge    => vk::SamplerAddressMode::CLAMP_TO_EDGE,
    }
}

unsafe fn upload_texture(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    tex_data: &TextureData,
) -> Result<GpuTexture> {
    let image = match tex_data.format {
        TextureFormat::Rgba8 => unsafe {
            GpuImage::upload_rgba8(
                device, instance, physical_device, command_pool, queue,
                tex_data.width, tex_data.height, &tex_data.pixels,
            )?
        },
        TextureFormat::Rgba32F => unsafe {
            let floats: &[f32] = bytemuck::cast_slice(&tex_data.pixels);
            GpuImage::upload_rgba32f(
                device, instance, physical_device, command_pool, queue,
                tex_data.width, tex_data.height, floats,
            )?
        },
    };

    let sampler = unsafe {
        device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(map_filter(tex_data.sampler.mag_filter))
                .min_filter(map_filter(tex_data.sampler.min_filter))
                .address_mode_u(map_address(tex_data.sampler.address_mode_u))
                .address_mode_v(map_address(tex_data.sampler.address_mode_v))
                .address_mode_w(map_address(tex_data.sampler.address_mode_w))
                .max_lod(vk::LOD_CLAMP_NONE)
                .max_anisotropy(1.0),
            None,
        ).context("create texture sampler")?
    };

    Ok(GpuTexture { image, sampler })
}

fn allocate_material_set(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
) -> Result<vk::DescriptorSet> {
    let layouts = [layout];
    let sets = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts),
        ).context("allocate material descriptor set")?
    };
    Ok(sets[0])
}

#[allow(clippy::too_many_arguments)]
fn write_material_descriptors(
    device: &ash::Device,
    ds: vk::DescriptorSet,
    mat: &Material,
    textures: &[GpuTexture],
    white: &GpuTexture,
    flat_normal: &GpuTexture,
    mr_placeholder: &GpuTexture,
    black: &GpuTexture,
) {
    let tex_or = |idx: Option<usize>, fallback: &GpuTexture| -> (vk::ImageView, vk::Sampler) {
        idx.and_then(|i| textures.get(i))
            .map(|t| (t.image.view, t.sampler))
            .unwrap_or((fallback.image.view, fallback.sampler))
    };

    let (bc_view, bc_samp) = tex_or(mat.base_color_texture, white);
    let (nm_view, nm_samp) = tex_or(mat.normal_texture, flat_normal);
    let (mr_view, mr_samp) = tex_or(mat.metallic_roughness_texture, mr_placeholder);
    let (oc_view, oc_samp) = tex_or(mat.occlusion_texture, white);
    let (em_view, em_samp) = tex_or(mat.emissive_texture, black);

    let image_infos = [
        vk::DescriptorImageInfo { image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, image_view: bc_view, sampler: bc_samp },
        vk::DescriptorImageInfo { image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, image_view: nm_view, sampler: nm_samp },
        vk::DescriptorImageInfo { image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, image_view: mr_view, sampler: mr_samp },
        vk::DescriptorImageInfo { image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, image_view: oc_view, sampler: oc_samp },
        vk::DescriptorImageInfo { image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, image_view: em_view, sampler: em_samp },
    ];

    let writes: Vec<vk::WriteDescriptorSet> = image_infos
        .iter()
        .enumerate()
        .map(|(i, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(info))
        })
        .collect();
    unsafe { device.update_descriptor_sets(&writes, &[]); }
}

// Vertex is used as a marker for the vertex buffer stride in pipeline creation.
// Assert it matches the expected GPU layout.
const _: () = assert!(std::mem::size_of::<Vertex>() == 48);
