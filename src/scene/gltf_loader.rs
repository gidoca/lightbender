#![allow(unsafe_op_in_unsafe_fn)]

use std::path::Path;

use anyhow::{Context, Result};
use ash::vk;
use glam::{Mat4, Quat, Vec3};

use crate::types::GpuVertex;
use crate::vulkan::{
    buffer::upload_to_device_local,
    image::GpuImage,
};

use super::{
    GpuMaterial, GpuMesh, GpuPrimitive, GpuTexture, Scene, SceneNode, Transform,
};

pub struct LoadContext<'a> {
    pub device:          &'a ash::Device,
    pub instance:        &'a ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub command_pool:    vk::CommandPool,
    pub queue:           vk::Queue,
    pub material_set_layout: vk::DescriptorSetLayout,
}

pub fn load(ctx: &LoadContext, path: &Path) -> Result<Scene> {
    let (document, buffers, images) =
        gltf::import(path).with_context(|| format!("load glTF: {}", path.display()))?;

    unsafe { load_inner(ctx, &document, &buffers, &images) }
}

unsafe fn load_inner(
    ctx: &LoadContext,
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
) -> Result<Scene> {
    // ── Textures ──────────────────────────────────────────────────────────────
    let mut textures: Vec<GpuTexture> = Vec::new();

    for texture in document.textures() {
        let img_data = &images[texture.source().index()];
        // Convert to RGBA8
        let rgba: Vec<u8> = match img_data.format {
            gltf::image::Format::R8G8B8A8 => img_data.pixels.clone(),
            gltf::image::Format::R8G8B8 => img_data
                .pixels
                .chunks_exact(3)
                .flat_map(|c| [c[0], c[1], c[2], 255])
                .collect(),
            gltf::image::Format::R8 => img_data
                .pixels
                .iter()
                .flat_map(|&v| [v, v, v, 255])
                .collect(),
            gltf::image::Format::R8G8 => img_data
                .pixels
                .chunks_exact(2)
                .flat_map(|c| [c[0], c[1], 0, 255])
                .collect(),
            _ => {
                // Fall back to image crate for other formats
                let dyn_img = image::load_from_memory(&img_data.pixels)
                    .context("decode texture")?;
                dyn_img.to_rgba8().into_raw()
            }
        };

        let gpu_image = GpuImage::upload_rgba8(
            ctx.device,
            ctx.instance,
            ctx.physical_device,
            ctx.command_pool,
            ctx.queue,
            img_data.width,
            img_data.height,
            &rgba,
        )?;

        let sampler_info = texture.sampler();
        let mag_filter = match sampler_info.mag_filter() {
            Some(gltf::texture::MagFilter::Nearest) => vk::Filter::NEAREST,
            _ => vk::Filter::LINEAR,
        };
        let (min_filter, mipmap_mode) = match sampler_info.min_filter() {
            Some(gltf::texture::MinFilter::Nearest)
            | Some(gltf::texture::MinFilter::NearestMipmapNearest) => {
                (vk::Filter::NEAREST, vk::SamplerMipmapMode::NEAREST)
            }
            _ => (vk::Filter::LINEAR, vk::SamplerMipmapMode::LINEAR),
        };
        let wrap = |w: gltf::texture::WrappingMode| match w {
            gltf::texture::WrappingMode::ClampToEdge   => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            gltf::texture::WrappingMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
            gltf::texture::WrappingMode::Repeat        => vk::SamplerAddressMode::REPEAT,
        };

        let sampler = ctx.device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(mag_filter)
                .min_filter(min_filter)
                .mipmap_mode(mipmap_mode)
                .address_mode_u(wrap(sampler_info.wrap_s()))
                .address_mode_v(wrap(sampler_info.wrap_t()))
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .max_lod(vk::LOD_CLAMP_NONE)
                .max_anisotropy(1.0),
            None,
        )?;

        textures.push(GpuTexture { image: gpu_image, sampler });
    }

    // ── 1×1 placeholder textures ──────────────────────────────────────────────
    let make_placeholder = |pixels: [u8; 4]| -> Result<GpuTexture> {
        let img = GpuImage::upload_rgba8(
            ctx.device, ctx.instance, ctx.physical_device,
            ctx.command_pool, ctx.queue, 1, 1, &pixels,
        )?;
        let sampler = ctx.device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT),
            None,
        )?;
        Ok(GpuTexture { image: img, sampler })
    };
    // White = base color (albedo=1), occlusion (fully lit)
    let white_tex = make_placeholder([255, 255, 255, 255])?;
    // Flat-normal placeholder: (0.5, 0.5, 1.0) ≈ tangent-space up
    let flat_normal_tex = make_placeholder([128, 128, 255, 255])?;
    // Metallic-roughness: G=roughness=0.5, B=metallic=0 (non-metallic, medium rough)
    let mr_placeholder = make_placeholder([0, 128, 0, 255])?;
    // Black = no emission
    let black_tex = make_placeholder([0, 0, 0, 255])?;

    // ── Descriptor pool for material sets ────────────────────────────────────
    let mat_count = document.materials().count().max(1) as u32;
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: mat_count * 5,
    }];
    let descriptor_pool = ctx.device.create_descriptor_pool(
        &vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(mat_count),
        None,
    )?;

    // ── Materials ─────────────────────────────────────────────────────────────
    let mut materials: Vec<GpuMaterial> = Vec::new();

    let alloc_set = |pool: vk::DescriptorPool| -> Result<vk::DescriptorSet> {
        let layouts = [ctx.material_set_layout];
        Ok(ctx.device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts),
        )?[0])
    };

    for material in document.materials() {
        let pbr = material.pbr_metallic_roughness();

        let ds = alloc_set(descriptor_pool)?;

        // Gather the five texture slots (view + sampler), using appropriate fallbacks
        let tex_or = |idx: Option<usize>, fallback: &GpuTexture| -> (vk::ImageView, vk::Sampler) {
            idx.map(|i| (textures[i].image.view, textures[i].sampler))
               .unwrap_or((fallback.image.view, fallback.sampler))
        };

        let (bc_view, bc_samp) = tex_or(pbr.base_color_texture().map(|t| t.texture().index()), &white_tex);
        let (nm_view, nm_samp) = tex_or(material.normal_texture().map(|t| t.texture().index()), &flat_normal_tex);
        let (mr_view, mr_samp) = tex_or(pbr.metallic_roughness_texture().map(|t| t.texture().index()), &mr_placeholder);
        let (oc_view, oc_samp) = tex_or(material.occlusion_texture().map(|t| t.texture().index()), &white_tex);
        let (em_view, em_samp) = tex_or(material.emissive_texture().map(|t| t.texture().index()), &black_tex);

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
        ctx.device.update_descriptor_sets(&writes, &[]);

        materials.push(GpuMaterial {
            base_color_factor:          pbr.base_color_factor(),
            metallic_factor:            pbr.metallic_factor(),
            roughness_factor:           pbr.roughness_factor(),
            emissive_factor:            material.emissive_factor(),
            base_color_texture:         pbr.base_color_texture().map(|t| t.texture().index()),
            normal_texture:             material.normal_texture().map(|t| t.texture().index()),
            metallic_roughness_texture: pbr.metallic_roughness_texture().map(|t| t.texture().index()),
            occlusion_texture:          material.occlusion_texture().map(|t| t.texture().index()),
            emissive_texture:           material.emissive_texture().map(|t| t.texture().index()),
            double_sided:               material.double_sided(),
            pipeline_name:              None,
            descriptor_set:             ds,
        });
    }

    // Add placeholder textures to list so they get cleaned up
    textures.push(white_tex);
    textures.push(flat_normal_tex);
    textures.push(mr_placeholder);
    textures.push(black_tex);

    // ── Meshes ────────────────────────────────────────────────────────────────
    // Default material index — fallback if a primitive has no material
    let default_mat_idx = if materials.is_empty() {
        // Create a minimal default material
        let ds = {
            // Need at least 1 set in the pool; bump mat_count above if this path is hit
            // For safety, allocate a fallback descriptor set from the pool
            let layouts = [ctx.material_set_layout];
            ctx.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            ).unwrap_or_default()
        };
        let ds = ds.into_iter().next().unwrap_or(vk::DescriptorSet::null());
        materials.push(GpuMaterial {
            base_color_factor: [1.0; 4],
            metallic_factor: 0.0,
            roughness_factor: 0.5,
            emissive_factor: [0.0; 3],
            base_color_texture: None,
            normal_texture: None,
            metallic_roughness_texture: None,
            occlusion_texture: None,
            emissive_texture: None,
            double_sided: false,
            pipeline_name: None,
            descriptor_set: ds,
        });
        0
    } else {
        0
    };

    let mut meshes: Vec<GpuMesh> = Vec::new();

    for mesh in document.meshes() {
        let mut primitives = Vec::new();
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buf| Some(&buffers[buf.index()]));

            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .context("mesh has no positions")?
                .collect();

            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|n| n.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);

            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|tc| tc.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

            let tangents: Vec<[f32; 4]> = reader
                .read_tangents()
                .map(|t| t.collect())
                .unwrap_or_else(|| vec![[1.0, 0.0, 0.0, 1.0]; positions.len()]);

            let indices: Vec<u32> = reader
                .read_indices()
                .map(|i| i.into_u32().collect())
                .unwrap_or_else(|| (0..positions.len() as u32).collect());

            let vertices: Vec<GpuVertex> = positions
                .iter()
                .zip(normals.iter())
                .zip(uvs.iter())
                .zip(tangents.iter())
                .map(|(((pos, norm), uv), tan)| GpuVertex {
                    position: *pos,
                    normal:   *norm,
                    uv:       *uv,
                    tangent:  *tan,
                })
                .collect();

            let vertex_buffer = upload_to_device_local(
                ctx.device,
                ctx.instance,
                ctx.physical_device,
                ctx.command_pool,
                ctx.queue,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                &vertices,
            )?;
            let index_buffer = upload_to_device_local(
                ctx.device,
                ctx.instance,
                ctx.physical_device,
                ctx.command_pool,
                ctx.queue,
                vk::BufferUsageFlags::INDEX_BUFFER,
                &indices,
            )?;

            let mat_idx = primitive
                .material()
                .index()
                .unwrap_or(default_mat_idx);

            primitives.push(GpuPrimitive {
                vertex_buffer,
                index_buffer,
                index_count: indices.len() as u32,
                material: mat_idx,
            });
        }
        meshes.push(GpuMesh {
            name: mesh.name().unwrap_or("").to_string(),
            primitives,
        });
    }

    // ── Scene graph ───────────────────────────────────────────────────────────
    let gltf_scene = document.default_scene().or_else(|| document.scenes().next());
    let mut nodes: Vec<SceneNode> = document
        .nodes()
        .map(|n| {
            let (translation, rotation, scale) = match n.transform() {
                gltf::scene::Transform::Matrix { matrix } => {
                    Mat4::from_cols_array_2d(&matrix).to_scale_rotation_translation()
                }
                gltf::scene::Transform::Decomposed {
                    translation,
                    rotation,
                    scale,
                } => (
                    Vec3::from(scale),
                    Quat::from_array(rotation),
                    Vec3::from(translation),
                ),
            };
            SceneNode {
                name:            n.name().unwrap_or("").to_string(),
                local_transform: Transform {
                    translation: scale,  // swap: decomposed gives (scale, rot, trans)
                    rotation,
                    scale:       translation,
                },
                parent:   None,
                children: n.children().map(|c| c.index()).collect(),
                mesh:     n.mesh().map(|m| m.index()),
            }
        })
        .collect();

    // Fix: Mat4::to_scale_rotation_translation returns (scale, rotation, translation)
    // but I accidentally swapped them above. Re-extract correctly.
    for (i, n) in document.nodes().enumerate() {
        let tf = match n.transform() {
            gltf::scene::Transform::Matrix { matrix } => {
                let m = Mat4::from_cols_array_2d(&matrix);
                let (s, r, t) = m.to_scale_rotation_translation();
                Transform { translation: t, rotation: r, scale: s }
            }
            gltf::scene::Transform::Decomposed { translation, rotation, scale } => Transform {
                translation: Vec3::from(translation),
                rotation:    Quat::from_array(rotation),
                scale:       Vec3::from(scale),
            },
        };
        nodes[i].local_transform = tf;
    }

    // Wire up parent indices
    for i in 0..nodes.len() {
        let children = nodes[i].children.clone();
        for child in children {
            nodes[child].parent = Some(i);
        }
    }

    // If a scene is specified, mark only scene-root nodes (parent == None and in scene)
    // All other nodes keep parent == None and will still appear in draw calls.
    let _ = gltf_scene;

    let node_count = nodes.len();
    let mut scene = Scene {
        nodes,
        meshes,
        materials,
        textures,
        world_transforms: vec![Mat4::IDENTITY; node_count],
        descriptor_pool,
    };
    scene.update_world_transforms();
    Ok(scene)
}
