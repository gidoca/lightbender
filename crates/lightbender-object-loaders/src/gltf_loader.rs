use std::path::Path;

use anyhow::{Context, Result};
use glam::{Mat4, Quat, Vec3};

use lightbender_scene::{
    AddressMode, FilterMode, Material, Mesh, Primitive, SamplerDesc, Scene, SceneNode,
    TextureData, TextureFormat, Transform, Vertex,
};

pub fn load_gltf(path: &Path) -> Result<Scene> {
    let (document, buffers, images) =
        gltf::import(path).with_context(|| format!("load glTF: {}", path.display()))?;

    load_inner(&document, &buffers, &images)
}

fn load_inner(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
) -> Result<Scene> {
    // ── Textures ─────────────────────────────────────────────────────────────
    let mut textures: Vec<TextureData> = Vec::new();

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

        let sampler_info = texture.sampler();
        let mag_filter = match sampler_info.mag_filter() {
            Some(gltf::texture::MagFilter::Nearest) => FilterMode::Nearest,
            _ => FilterMode::Linear,
        };
        let min_filter = match sampler_info.min_filter() {
            Some(gltf::texture::MinFilter::Nearest)
            | Some(gltf::texture::MinFilter::NearestMipmapNearest) => FilterMode::Nearest,
            _ => FilterMode::Linear,
        };
        let wrap = |w: gltf::texture::WrappingMode| match w {
            gltf::texture::WrappingMode::ClampToEdge    => AddressMode::ClampToEdge,
            gltf::texture::WrappingMode::MirroredRepeat => AddressMode::MirroredRepeat,
            gltf::texture::WrappingMode::Repeat         => AddressMode::Repeat,
        };

        textures.push(TextureData {
            width:  img_data.width,
            height: img_data.height,
            format: TextureFormat::Rgba8,
            pixels: rgba,
            sampler: SamplerDesc {
                mag_filter,
                min_filter,
                address_mode_u: wrap(sampler_info.wrap_s()),
                address_mode_v: wrap(sampler_info.wrap_t()),
                address_mode_w: AddressMode::Repeat,
            },
        });
    }

    // ── Materials ─────────────────────────────────────────────────────────────
    let mut materials: Vec<Material> = Vec::new();

    for material in document.materials() {
        let pbr = material.pbr_metallic_roughness();
        materials.push(Material {
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
        });
    }

    // ── Default material fallback ────────────────────────────────────────────
    if materials.is_empty() {
        materials.push(Material {
            base_color_factor: [1.0; 4],
            metallic_factor:   0.0,
            roughness_factor:  0.5,
            ..Material::default()
        });
    }

    // ── Meshes ───────────────────────────────────────────────────────────────
    let mut meshes: Vec<Mesh> = Vec::new();

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

            let vertices: Vec<Vertex> = positions
                .iter()
                .zip(normals.iter())
                .zip(uvs.iter())
                .zip(tangents.iter())
                .map(|(((pos, norm), uv), tan)| Vertex {
                    position: *pos,
                    normal:   *norm,
                    uv:       *uv,
                    tangent:  *tan,
                })
                .collect();

            let mat_idx = primitive.material().index().unwrap_or(0);

            primitives.push(Primitive { vertices, indices, material: mat_idx });
        }
        meshes.push(Mesh {
            name: mesh.name().unwrap_or("").to_string(),
            primitives,
        });
    }

    // ── Scene graph ──────────────────────────────────────────────────────────
    let mut nodes: Vec<SceneNode> = document
        .nodes()
        .map(|n| {
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
            SceneNode {
                name:            n.name().unwrap_or("").to_string(),
                local_transform: tf,
                parent:          None,
                children:        n.children().map(|c| c.index()).collect(),
                mesh:            n.mesh().map(|m| m.index()),
            }
        })
        .collect();

    // Wire up parent indices
    for i in 0..nodes.len() {
        let children = nodes[i].children.clone();
        for child in children {
            nodes[child].parent = Some(i);
        }
    }

    let mut scene = Scene {
        nodes,
        meshes,
        materials,
        textures,
        world_transforms: Vec::new(),
    };
    scene.update_world_transforms();
    Ok(scene)
}
