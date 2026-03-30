use std::path::Path;

use anyhow::{Context, Result};
use ash::vk;
use glam::{Mat4, Vec3};
use quick_xml::events::Event;
use quick_xml::Reader;

use crate::camera::OrbitalCamera;
use crate::renderer::Renderer;
use crate::types::GpuLight;
use crate::vulkan::buffer::upload_to_device_local;
use crate::vulkan::image::GpuImage;

use super::{
    gltf_loader::LoadContext,
    GpuMaterial, GpuMesh, GpuPrimitive, GpuTexture, Scene, SceneNode, Transform,
};

pub struct LoadedMitsubaScene {
    pub scene:       Scene,
    pub camera:      OrbitalCamera,
    pub lights:      Vec<GpuLight>,
    pub env_map:     Option<String>,  // path to environment map
    pub env_scale:   f32,
}

pub fn load_mitsuba(renderer: &Renderer, path: &Path) -> Result<LoadedMitsubaScene> {
    let xml = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("read Mitsuba XML: {}: {e}", path.display()))?;

    let base_dir = path.parent().unwrap_or(Path::new("."));

    // ── Phase 1: Parse XML ───────────────────────────────────────────────────
    let mut reader = Reader::from_str(&xml);
    let mut buf = Vec::new();
    let mut skip_buf = Vec::new();

    let mut camera_desc: Option<MitsubaCameraDesc> = None;
    let mut shapes: Vec<MitsubaShape> = Vec::new();
    let mut emitters: Vec<(MitsubaEmitter, Mat4)> = Vec::new();
    let mut named_bsdfs: HashMap<String, MitsubaMaterial> = HashMap::new();

    loop {
        match reader.read_event_into(&mut buf).context("parse scene")? {
            Event::Start(ref e) => {
                let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                match tag.as_str() {
                    "sensor" => {
                        camera_desc = Some(parse_sensor(&mut reader)?);
                    }
                    "shape" => {
                        let shape_type = attr_str(e, "type").unwrap_or_default();
                        match parse_shape(&mut reader, &shape_type, base_dir) {
                            Ok(shape) => shapes.push(shape),
                            Err(err) => log::error!("Failed to parse shape: {err:#}"),
                        }
                    }
                    "emitter" => {
                        let emitter_type = attr_str(e, "type").unwrap_or_default();
                        match parse_emitter(&mut reader, &emitter_type) {
                            Ok(em) => emitters.push((em, Mat4::IDENTITY)),
                            Err(err) => log::error!("Failed to parse emitter: {err:#}"),
                        }
                    }
                    "bsdf" => {
                        let bsdf_type = attr_str(e, "type").unwrap_or_default();
                        let bsdf_id = attr_str(e, "id");
                        match parse_bsdf(&mut reader, &bsdf_type) {
                            Ok(mat) => {
                                if let Some(id) = bsdf_id {
                                    named_bsdfs.insert(id, mat);
                                }
                            }
                            Err(err) => log::error!("Failed to parse BSDF: {err:#}"),
                        }
                    }
                    "scene" => {
                        // Continue into scene children
                    }
                    _ => {
                        // Skip integrator, sampler, etc.
                        let end = e.to_end().into_owned();
                        reader.read_to_end_into(end.name(), &mut skip_buf)?;
                        skip_buf.clear();
                    }
                }
            }
            Event::Eof => break,
            _ => {}
        }
        buf.clear();
    }

    log::info!(
        "Parsed Mitsuba scene: {} shapes, {} emitters, {} named BSDFs",
        shapes.len(), emitters.len(), named_bsdfs.len()
    );

    // ── Phase 2: Upload GPU resources ────────────────────────────────────────
    let ctx = renderer.load_context();

    // Create placeholder textures
    let make_placeholder = |pixels: [u8; 4]| -> Result<GpuTexture> {
        let img = unsafe {
            GpuImage::upload_rgba8(
                ctx.device, ctx.instance, ctx.physical_device,
                ctx.command_pool, ctx.queue, 1, 1, &pixels,
            )?
        };
        let sampler = unsafe {
            ctx.device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT)
                    .address_mode_w(vk::SamplerAddressMode::REPEAT),
                None,
            )?
        };
        Ok(GpuTexture { image: img, sampler })
    };

    let white_tex = make_placeholder([255, 255, 255, 255])?;
    let flat_normal_tex = make_placeholder([128, 128, 255, 255])?;
    let mr_placeholder = make_placeholder([0, 128, 0, 255])?;
    let black_tex = make_placeholder([0, 0, 0, 255])?;

    let mut textures: Vec<GpuTexture> = Vec::new();
    let mut texture_index_cache: HashMap<String, usize> = HashMap::new();

    // Helper: load a texture file and return its index
    let load_texture = |path: &Path, textures: &mut Vec<GpuTexture>, cache: &mut HashMap<String, usize>| -> Result<usize> {
        let key = path.to_string_lossy().to_string();
        if let Some(&idx) = cache.get(&key) {
            return Ok(idx);
        }

        let img = image::open(path)
            .with_context(|| format!("load texture: {}", path.display()))?;
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();

        let gpu_img = unsafe {
            GpuImage::upload_rgba8(
                ctx.device, ctx.instance, ctx.physical_device,
                ctx.command_pool, ctx.queue, w, h, &rgba,
            )?
        };

        let sampler = unsafe {
            ctx.device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT)
                    .address_mode_w(vk::SamplerAddressMode::REPEAT)
                    .max_lod(vk::LOD_CLAMP_NONE)
                    .max_anisotropy(1.0),
                None,
            )?
        };

        let idx = textures.len();
        textures.push(GpuTexture { image: gpu_img, sampler });
        cache.insert(key, idx);
        Ok(idx)
    };

    // Descriptor pool for materials
    let mat_count = shapes.len().max(1) as u32;
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: mat_count * 5,
    }];
    let descriptor_pool = unsafe {
        ctx.device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(mat_count),
            None,
        )?
    };

    let alloc_descriptor_set = || -> Result<vk::DescriptorSet> {
        let layouts = [ctx.material_set_layout];
        Ok(unsafe {
            ctx.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            )?
        }[0])
    };

    // ── Build scene from shapes ──────────────────────────────────────────────
    let mut materials: Vec<GpuMaterial> = Vec::new();
    let mut meshes: Vec<GpuMesh> = Vec::new();
    let mut nodes: Vec<SceneNode> = Vec::new();
    let mut gpu_lights: Vec<GpuLight> = Vec::new();

    for (i, shape) in shapes.iter().enumerate() {
        if shape.vertices.is_empty() || shape.indices.is_empty() {
            continue;
        }

        // Resolve material
        let mat_desc = if let Some(ref mat) = shape.material {
            mat.clone()
        } else if let Some(ref ref_id) = shape.material_ref {
            named_bsdfs.get(ref_id).cloned().unwrap_or_else(|| {
                log::warn!("BSDF ref '{ref_id}' not found, using default");
                MitsubaMaterial::default()
            })
        } else {
            MitsubaMaterial::default()
        };

        // Load base color texture if referenced
        let bc_tex_idx = if let Some(ref tex_path) = mat_desc.base_color_texture {
            if let Some(checker_spec) = tex_path.strip_prefix("__checkerboard:") {
                // Generate procedural checkerboard texture
                match generate_checkerboard_texture(checker_spec, &ctx) {
                    Ok(tex) => {
                        let idx = textures.len();
                        textures.push(tex);
                        Some(idx)
                    }
                    Err(e) => {
                        log::error!("Failed to generate checkerboard: {e:#}");
                        None
                    }
                }
            } else {
                let resolved = resolve_path(base_dir, tex_path);
                match load_texture(&resolved, &mut textures, &mut texture_index_cache) {
                    Ok(idx) => Some(idx),
                    Err(e) => {
                        log::error!("Failed to load texture {}: {e:#}", resolved.display());
                        None
                    }
                }
            }
        } else {
            None
        };

        // Handle area emitters -> emissive factor
        let mut emissive = mat_desc.emissive;
        if let Some(MitsubaEmitter::Area { radiance }) = &shape.area_emitter {
            emissive = *radiance;
        }

        // Create material
        let ds = alloc_descriptor_set()?;

        let tex_or = |idx: Option<usize>, fallback: &GpuTexture| -> (vk::ImageView, vk::Sampler) {
            idx.map(|i| (textures[i].image.view, textures[i].sampler))
               .unwrap_or((fallback.image.view, fallback.sampler))
        };

        let (bc_view, bc_samp) = tex_or(bc_tex_idx, &white_tex);
        let (nm_view, nm_samp) = (flat_normal_tex.image.view, flat_normal_tex.sampler);
        let (mr_view, mr_samp) = (mr_placeholder.image.view, mr_placeholder.sampler);
        let (oc_view, oc_samp) = (white_tex.image.view, white_tex.sampler);
        let (em_view, em_samp) = (black_tex.image.view, black_tex.sampler);

        let image_infos = [
            vk::DescriptorImageInfo { image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, image_view: bc_view, sampler: bc_samp },
            vk::DescriptorImageInfo { image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, image_view: nm_view, sampler: nm_samp },
            vk::DescriptorImageInfo { image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, image_view: mr_view, sampler: mr_samp },
            vk::DescriptorImageInfo { image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, image_view: oc_view, sampler: oc_samp },
            vk::DescriptorImageInfo { image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, image_view: em_view, sampler: em_samp },
        ];

        let writes: Vec<vk::WriteDescriptorSet> = image_infos.iter().enumerate().map(|(binding, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(binding as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(info))
        }).collect();
        unsafe { ctx.device.update_descriptor_sets(&writes, &[]) };

        let mat_idx = materials.len();
        materials.push(GpuMaterial {
            base_color_factor:          [mat_desc.base_color.x, mat_desc.base_color.y, mat_desc.base_color.z, mat_desc.alpha],
            metallic_factor:            mat_desc.metallic,
            roughness_factor:           mat_desc.roughness,
            emissive_factor:            [emissive.x, emissive.y, emissive.z],
            base_color_texture:         bc_tex_idx,
            normal_texture:             None,
            metallic_roughness_texture: None,
            occlusion_texture:          None,
            emissive_texture:           None,
            double_sided:               mat_desc.double_sided,
            pipeline_name:              None,
            descriptor_set:             ds,
        });

        // Upload vertex/index buffers
        let vertex_buffer = unsafe {
            upload_to_device_local(
                ctx.device, ctx.instance, ctx.physical_device,
                ctx.command_pool, ctx.queue,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                &shape.vertices,
            )?
        };
        let index_buffer = unsafe {
            upload_to_device_local(
                ctx.device, ctx.instance, ctx.physical_device,
                ctx.command_pool, ctx.queue,
                vk::BufferUsageFlags::INDEX_BUFFER,
                &shape.indices,
            )?
        };

        let mesh_idx = meshes.len();
        meshes.push(GpuMesh {
            name: format!("shape_{i}"),
            primitives: vec![GpuPrimitive {
                vertex_buffer,
                index_buffer,
                index_count: shape.indices.len() as u32,
                material: mat_idx,
            }],
        });

        // Decompose to_world into Transform for the scene node
        let (scale, rotation, translation) = shape.to_world.to_scale_rotation_translation();
        nodes.push(SceneNode {
            name: format!("shape_{i}"),
            local_transform: Transform { translation, rotation, scale },
            parent: None,
            children: Vec::new(),
            mesh: Some(mesh_idx),
        });

        // Area emitter -> point light proxy at centroid
        if let Some(MitsubaEmitter::Area { radiance }) = &shape.area_emitter {
            let centroid = shape.to_world.transform_point3(Vec3::ZERO);
            let max_r = radiance.x.max(radiance.y).max(radiance.z);
            if max_r > 0.0 {
                gpu_lights.push(GpuLight {
                    position_or_direction: [centroid.x, centroid.y, centroid.z, 1.0],
                    color: [radiance.x / max_r, radiance.y / max_r, radiance.z / max_r],
                    intensity: max_r,
                    range: 100.0,
                    ..Default::default()
                });
            }
        }
    }

    // Add placeholder textures to list for cleanup
    textures.push(white_tex);
    textures.push(flat_normal_tex);
    textures.push(mr_placeholder);
    textures.push(black_tex);

    // Convert emitters to GPU lights
    for (emitter, to_world) in &emitters {
        if let Some(light) = gpu_light_from_emitter(emitter, to_world) {
            gpu_lights.push(light);
        }
    }

    // Find environment map
    let mut env_map = None;
    let mut env_scale = 1.0f32;
    for (emitter, _) in &emitters {
        if let MitsubaEmitter::EnvMap { filename, scale } = emitter {
            env_map = Some(resolve_path(base_dir, filename).to_string_lossy().to_string());
            env_scale = *scale;
        }
    }

    // Camera
    let camera = camera_desc
        .map(|d| camera_from_mitsuba(&d))
        .unwrap_or_else(|| OrbitalCamera::new(Vec3::ZERO, 5.0, 30.0, 20.0));

    // Build scene
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

    Ok(LoadedMitsubaScene { scene, camera, lights: gpu_lights, env_map, env_scale })
}

// ── Sensor (camera) parsing ──────────────────────────────────────────────────

struct MitsubaCameraDesc {
    fov:      f32, // degrees, as specified in the scene
    fov_axis: String, // "x" (default), "y", "smaller", "larger"
    near:     f32,
    far:      f32,
    to_world: Mat4,
    film_width:  u32,
    film_height: u32,
}

impl Default for MitsubaCameraDesc {
    fn default() -> Self {
        Self {
            fov:      45.0,
            fov_axis: "x".to_string(),
            near:     0.01,
            far:      1000.0,
            to_world: Mat4::IDENTITY,
            film_width:  768,
            film_height: 576,
        }
    }
}

/// Parse a `<sensor>` element. Reader is positioned just after `<sensor ...>`.
fn parse_sensor(reader: &mut Reader<&[u8]>) -> Result<MitsubaCameraDesc> {
    let mut desc = MitsubaCameraDesc::default();
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf).context("parse_sensor")? {
            Event::Empty(ref e) => {
                let tag = e.name();
                if tag.as_ref() == b"float"
                    && let (Some(name), Some(val)) = (attr_str(e, "name"), attr_f32(e, "value"))
                {
                    match name.as_str() {
                        "fov" => desc.fov = val,
                        "near_clip" => desc.near = val,
                        "far_clip" => desc.far = val,
                        _ => {}
                    }
                } else if tag.as_ref() == b"string"
                    && let (Some(name), Some(val)) = (attr_str(e, "name"), attr_str(e, "value"))
                    && name == "fov_axis"
                {
                    desc.fov_axis = val;
                }
            }
            Event::Start(ref e) => {
                let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                match tag.as_str() {
                    "transform" => {
                        desc.to_world = parse_transform(reader)?;
                    }
                    "film" => {
                        parse_film_into(reader, &mut desc)?;
                    }
                    _ => {
                        let end = e.to_end().into_owned();
                        let mut skip_buf = Vec::new();
                        reader.read_to_end_into(end.name(), &mut skip_buf)?;
                    }
                }
            }
            Event::End(ref e) if e.name().as_ref() == b"sensor" => break,
            Event::Eof => anyhow::bail!("unexpected EOF inside <sensor>"),
            _ => {}
        }
        buf.clear();
    }

    Ok(desc)
}

/// Parse `<film>` to extract width/height.
fn parse_film_into(reader: &mut Reader<&[u8]>, desc: &mut MitsubaCameraDesc) -> Result<()> {
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf).context("parse_film")? {
            Event::Empty(ref e) => {
                if e.name().as_ref() == b"integer"
                    && let (Some(name), Some(val)) = (attr_str(e, "name"), attr_str(e, "value"))
                {
                    match name.as_str() {
                        "width"  => { desc.film_width  = val.parse().unwrap_or(desc.film_width); }
                        "height" => { desc.film_height = val.parse().unwrap_or(desc.film_height); }
                        _ => {}
                    }
                }
            }
            Event::Start(ref e) => {
                let end = e.to_end().into_owned();
                let mut skip_buf = Vec::new();
                reader.read_to_end_into(end.name(), &mut skip_buf)?;
            }
            Event::End(ref e) if e.name().as_ref() == b"film" => break,
            Event::Eof => anyhow::bail!("unexpected EOF inside <film>"),
            _ => {}
        }
        buf.clear();
    }
    Ok(())
}

/// Convert Mitsuba camera description to an OrbitalCamera.
fn camera_from_mitsuba(desc: &MitsubaCameraDesc) -> OrbitalCamera {
    // Extract camera position and forward direction from the to_world matrix.
    // In Mitsuba's convention, column 2 (+Z) is the forward (viewing) direction.
    let origin = desc.to_world.col(3).truncate();
    let forward = desc.to_world.col(2).truncate().normalize();

    // Place the orbital target along the forward direction.  Use the distance
    // from origin to the world origin as a heuristic, with a minimum of 1.0.
    let distance = origin.length().max(1.0);
    let target = origin + forward * distance;

    // Decompose into spherical coordinates matching OrbitalCamera::sync():
    //   x = dist * cos(pitch) * sin(yaw)
    //   y = dist * sin(pitch)
    //   z = dist * cos(pitch) * cos(yaw)
    let offset = origin - target;
    let dist = offset.length().max(0.001);
    let pitch = (offset.y / dist).asin();
    let yaw = offset.x.atan2(offset.z);

    let mut cam = OrbitalCamera::new(target, dist, yaw.to_degrees(), pitch.to_degrees());

    // Convert FOV to vertical radians.  Mitsuba's fov_axis defaults to "x" (horizontal).
    let aspect = desc.film_width as f32 / desc.film_height as f32;
    let fov_rad = desc.fov.to_radians();
    let hfov_to_vfov = |h: f32| 2.0 * ((h * 0.5).tan() / aspect).atan();
    let is_horizontal = match desc.fov_axis.as_str() {
        "y" => false,
        "x" | "" => true,
        "smaller" => aspect < 1.0,  // smaller dim is x when aspect < 1 → fov is horizontal
        "larger"  => aspect >= 1.0, // larger dim is x when aspect >= 1 → fov is horizontal
        _ => true,
    };
    cam.camera.fov_y = if is_horizontal { hfov_to_vfov(fov_rad) } else { fov_rad };
    cam.camera.near = desc.near;
    cam.camera.far = desc.far;
    cam
}

// ── Emitter (light / environment) parsing ────────────────────────────────────

/// Parsed Mitsuba emitter.
enum MitsubaEmitter {
    Point { position: Vec3, color: Vec3, intensity: f32 },
    Directional { direction: Vec3, color: Vec3, intensity: f32 },
    Spot { position: Vec3, direction: Vec3, color: Vec3, intensity: f32, cutoff_angle: f32 },
    EnvMap { filename: String, scale: f32 },
    Area { radiance: Vec3 },
}

/// Parse a `<emitter>` element. Reader is positioned just after `<emitter type="...">`.
fn parse_emitter(reader: &mut Reader<&[u8]>, emitter_type: &str) -> Result<MitsubaEmitter> {
    let props = parse_properties(reader, b"emitter")?;

    match emitter_type {
        "point" => {
            let position = props.get_point("position").unwrap_or(Vec3::ZERO);
            let (color, intensity) = props.get_color_intensity("intensity", Vec3::ONE, 1.0);
            Ok(MitsubaEmitter::Point { position, color, intensity })
        }
        "directional" | "distant" => {
            let direction = props.get_vec3("direction").unwrap_or(Vec3::new(0.0, -1.0, 0.0));
            let (color, intensity) = props.get_color_intensity("irradiance", Vec3::ONE, 1.0);
            Ok(MitsubaEmitter::Directional { direction: direction.normalize(), color, intensity })
        }
        "spot" => {
            let position = props.get_point("position").unwrap_or(Vec3::ZERO);
            let direction = props.get_vec3("direction").unwrap_or(Vec3::new(0.0, -1.0, 0.0));
            let (color, intensity) = props.get_color_intensity("intensity", Vec3::ONE, 1.0);
            let cutoff = props.get_float("cutoff_angle").unwrap_or(20.0);
            Ok(MitsubaEmitter::Spot {
                position,
                direction: direction.normalize(),
                color,
                intensity,
                cutoff_angle: cutoff,
            })
        }
        "envmap" => {
            let filename = props.get_string("filename")
                .context("<emitter type=\"envmap\"> missing 'filename'")?;
            let scale = props.get_float("scale").unwrap_or(1.0);
            Ok(MitsubaEmitter::EnvMap { filename, scale })
        }
        "area" => {
            let (radiance, _) = props.get_color_intensity("radiance", Vec3::ONE, 1.0);
            Ok(MitsubaEmitter::Area { radiance })
        }
        _ => {
            log::warn!("Unsupported emitter type: {emitter_type}");
            Ok(MitsubaEmitter::Point { position: Vec3::ZERO, color: Vec3::ONE, intensity: 0.0 })
        }
    }
}

/// Convert Mitsuba emitter to GpuLight.
fn gpu_light_from_emitter(emitter: &MitsubaEmitter, to_world: &Mat4) -> Option<GpuLight> {
    match *emitter {
        MitsubaEmitter::Point { position, color, intensity } => {
            let world_pos = to_world.transform_point3(position);
            Some(GpuLight {
                position_or_direction: [world_pos.x, world_pos.y, world_pos.z, 1.0],
                color: color.into(),
                intensity,
                range: 100.0,
                ..Default::default()
            })
        }
        MitsubaEmitter::Directional { direction, color, intensity } => {
            let world_dir = to_world.transform_vector3(direction).normalize();
            Some(GpuLight {
                position_or_direction: [world_dir.x, world_dir.y, world_dir.z, 0.0],
                color: color.into(),
                intensity,
                ..Default::default()
            })
        }
        MitsubaEmitter::Spot { position, direction, color, intensity, cutoff_angle } => {
            let world_pos = to_world.transform_point3(position);
            // TODO: store spot direction in GpuLight when light UBO is wired (step 12)
            let _world_dir = to_world.transform_vector3(direction).normalize();
            let inner = (cutoff_angle * 0.75).to_radians().cos();
            let outer = cutoff_angle.to_radians().cos();
            Some(GpuLight {
                position_or_direction: [world_pos.x, world_pos.y, world_pos.z, 2.0],
                color: color.into(),
                intensity,
                range: 100.0,
                spot_angles: [inner, outer],
                ..Default::default()
            })
        }
        MitsubaEmitter::EnvMap { .. } | MitsubaEmitter::Area { .. } => None,
    }
}

// ── Shape parsing ────────────────────────────────────────────────────────────

use crate::types::GpuVertex;

/// Parsed Mitsuba shape with geometry and material reference.
struct MitsubaShape {
    vertices:    Vec<GpuVertex>,
    indices:     Vec<u32>,
    to_world:    Mat4,
    material:    Option<MitsubaMaterial>,
    material_ref: Option<String>,  // id reference to top-level BSDF
    area_emitter: Option<MitsubaEmitter>,
}

/// Parse a `<shape>` element. Reader positioned just after `<shape type="...">`.
fn parse_shape(reader: &mut Reader<&[u8]>, shape_type: &str, base_dir: &Path) -> Result<MitsubaShape> {
    let mut to_world = Mat4::IDENTITY;
    let mut material: Option<MitsubaMaterial> = None;
    let mut material_ref: Option<String> = None;
    let mut area_emitter: Option<MitsubaEmitter> = None;
    let mut filename: Option<String> = None;
    let mut buf = Vec::new();
    let mut skip_buf = Vec::new();

    // shape-specific properties
    let mut center = Vec3::ZERO;
    let mut radius = 1.0f32;
    let mut flip_normals = false;

    loop {
        match reader.read_event_into(&mut buf).context("parse_shape")? {
            Event::Empty(ref e) => {
                let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                match tag.as_str() {
                    "string" => {
                        if attr_str(e, "name").as_deref() == Some("filename") {
                            filename = attr_str(e, "value");
                        }
                    }
                    "float" => {
                        if let (Some(name), Some(val)) = (attr_str(e, "name"), attr_f32(e, "value"))
                            && name == "radius"
                        {
                            radius = val;
                        }
                    }
                    "point" => {
                        if attr_str(e, "name").as_deref() == Some("center")
                            && let (Some(x), Some(y), Some(z)) = (attr_f32(e, "x"), attr_f32(e, "y"), attr_f32(e, "z"))
                        {
                            center = Vec3::new(x, y, z);
                        }
                    }
                    "boolean" => {
                        if attr_str(e, "name").as_deref() == Some("flip_normals") {
                            flip_normals = attr_str(e, "value").as_deref() == Some("true");
                        }
                    }
                    "ref" => {
                        material_ref = attr_str(e, "id");
                    }
                    _ => {}
                }
            }
            Event::Start(ref e) => {
                let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                match tag.as_str() {
                    "transform" => {
                        to_world = parse_transform(reader)?;
                    }
                    "bsdf" => {
                        let bsdf_type = attr_str(e, "type").unwrap_or_default();
                        material = Some(parse_bsdf(reader, &bsdf_type)?);
                    }
                    "emitter" => {
                        let emitter_type = attr_str(e, "type").unwrap_or_default();
                        area_emitter = Some(parse_emitter(reader, &emitter_type)?);
                    }
                    "ref" => {
                        material_ref = attr_str(e, "id");
                        let end = e.to_end().into_owned();
                        reader.read_to_end_into(end.name(), &mut skip_buf)?;
                        skip_buf.clear();
                    }
                    _ => {
                        let end = e.to_end().into_owned();
                        reader.read_to_end_into(end.name(), &mut skip_buf)?;
                        skip_buf.clear();
                    }
                }
            }
            Event::End(ref e) if e.name().as_ref() == b"shape" => break,
            Event::Eof => anyhow::bail!("unexpected EOF inside <shape>"),
            _ => {}
        }
        buf.clear();
    }

    // Build geometry based on shape type
    let (vertices, indices) = match shape_type {
        "obj" => {
            let path = resolve_path(base_dir, &filename.context("<shape type=\"obj\"> missing 'filename'")?);
            let meshes = super::obj_loader::load_obj(&path)?;
            merge_obj_meshes(meshes)
        }
        "ply" => {
            let path = resolve_path(base_dir, &filename.context("<shape type=\"ply\"> missing 'filename'")?);
            let mesh = super::ply_loader::load_ply(&path)?;
            (mesh.vertices, mesh.indices)
        }
        "sphere" => {
            generate_sphere(center, radius, 32, 16, flip_normals)
        }
        "rectangle" => generate_rectangle(flip_normals),
        "cube" => generate_cube(),
        "disk" => generate_disk(32, flip_normals),
        "cylinder" => generate_cylinder(32, flip_normals),
        _ => {
            log::warn!("Unsupported shape type: {shape_type}");
            (Vec::new(), Vec::new())
        }
    };

    Ok(MitsubaShape {
        vertices,
        indices,
        to_world,
        material,
        material_ref,
        area_emitter,
    })
}

/// Generate a checkerboard texture from a spec string "r0:g0:b0:r1:g1:b1".
fn generate_checkerboard_texture(spec: &str, ctx: &LoadContext) -> Result<GpuTexture> {
    let vals: Vec<f32> = spec.split(':').filter_map(|s| s.parse().ok()).collect();
    let (c0, c1) = if vals.len() >= 6 {
        ([vals[0], vals[1], vals[2]], [vals[3], vals[4], vals[5]])
    } else {
        ([0.4, 0.4, 0.4], [0.2, 0.2, 0.2])
    };

    let size = 128u32;
    let tiles = 8u32; // 8×8 check pattern
    let tile_size = size / tiles;
    let mut pixels = vec![0u8; (size * size * 4) as usize];

    for y in 0..size {
        for x in 0..size {
            let tx = x / tile_size;
            let ty = y / tile_size;
            let is_c0 = (tx + ty).is_multiple_of(2);
            let c = if is_c0 { c0 } else { c1 };
            let idx = ((y * size + x) * 4) as usize;
            pixels[idx]     = (c[0].clamp(0.0, 1.0) * 255.0) as u8;
            pixels[idx + 1] = (c[1].clamp(0.0, 1.0) * 255.0) as u8;
            pixels[idx + 2] = (c[2].clamp(0.0, 1.0) * 255.0) as u8;
            pixels[idx + 3] = 255;
        }
    }

    let image = unsafe {
        GpuImage::upload_rgba8(
            ctx.device, ctx.instance, ctx.physical_device,
            ctx.command_pool, ctx.queue, size, size, &pixels,
        )?
    };
    let sampler = unsafe {
        ctx.device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT),
            None,
        )?
    };
    Ok(GpuTexture { image, sampler })
}

fn resolve_path(base: &Path, relative: &str) -> std::path::PathBuf {
    let p = Path::new(relative);
    if p.is_absolute() { p.to_path_buf() } else { base.join(p) }
}

fn merge_obj_meshes(meshes: Vec<super::obj_loader::ObjMesh>) -> (Vec<GpuVertex>, Vec<u32>) {
    let mut all_verts = Vec::new();
    let mut all_indices = Vec::new();
    for mesh in meshes {
        let base = all_verts.len() as u32;
        all_verts.extend(mesh.vertices);
        all_indices.extend(mesh.indices.iter().map(|&i| i + base));
    }
    (all_verts, all_indices)
}

// ── Procedural geometry generators ───────────────────────────────────────────

fn generate_sphere(center: Vec3, radius: f32, slices: u32, stacks: u32, flip: bool) -> (Vec<GpuVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let sign = if flip { -1.0f32 } else { 1.0 };

    for j in 0..=stacks {
        let theta = std::f32::consts::PI * j as f32 / stacks as f32;
        let sin_t = theta.sin();
        let cos_t = theta.cos();

        for i in 0..=slices {
            let phi = 2.0 * std::f32::consts::PI * i as f32 / slices as f32;
            let nx = sin_t * phi.cos();
            let ny = cos_t;
            let nz = sin_t * phi.sin();

            vertices.push(GpuVertex {
                position: [center.x + radius * nx, center.y + radius * ny, center.z + radius * nz],
                normal:   [sign * nx, sign * ny, sign * nz],
                uv:       [i as f32 / slices as f32, j as f32 / stacks as f32],
                tangent:  [1.0, 0.0, 0.0, 1.0],
            });
        }
    }

    for j in 0..stacks {
        for i in 0..slices {
            let a = j * (slices + 1) + i;
            let b = a + slices + 1;
            if flip {
                indices.extend_from_slice(&[a, b, a + 1, b, b + 1, a + 1]);
            } else {
                indices.extend_from_slice(&[a, a + 1, b, b, a + 1, b + 1]);
            }
        }
    }

    super::obj_loader::compute_tangents(&mut vertices, &indices);
    (vertices, indices)
}

fn generate_rectangle(flip: bool) -> (Vec<GpuVertex>, Vec<u32>) {
    // Mitsuba rectangle: centered at origin, in XY plane, extends [-1,1] x [-1,1], normal +Z
    let sign = if flip { -1.0 } else { 1.0 };
    let vertices = vec![
        GpuVertex { position: [-1.0, -1.0, 0.0], normal: [0.0, 0.0, sign], uv: [0.0, 0.0], tangent: [1.0, 0.0, 0.0, 1.0] },
        GpuVertex { position: [ 1.0, -1.0, 0.0], normal: [0.0, 0.0, sign], uv: [1.0, 0.0], tangent: [1.0, 0.0, 0.0, 1.0] },
        GpuVertex { position: [ 1.0,  1.0, 0.0], normal: [0.0, 0.0, sign], uv: [1.0, 1.0], tangent: [1.0, 0.0, 0.0, 1.0] },
        GpuVertex { position: [-1.0,  1.0, 0.0], normal: [0.0, 0.0, sign], uv: [0.0, 1.0], tangent: [1.0, 0.0, 0.0, 1.0] },
    ];
    let indices = if flip {
        vec![0, 2, 1, 0, 3, 2]
    } else {
        vec![0, 1, 2, 0, 2, 3]
    };
    (vertices, indices)
}

fn generate_cube() -> (Vec<GpuVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        ([0.0, 0.0, 1.0],  [[-1.0,-1.0, 1.0], [ 1.0,-1.0, 1.0], [ 1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]]),  // +Z
        ([0.0, 0.0,-1.0],  [[ 1.0,-1.0,-1.0], [-1.0,-1.0,-1.0], [-1.0, 1.0,-1.0], [ 1.0, 1.0,-1.0]]),  // -Z
        ([1.0, 0.0, 0.0],  [[ 1.0,-1.0, 1.0], [ 1.0,-1.0,-1.0], [ 1.0, 1.0,-1.0], [ 1.0, 1.0, 1.0]]),  // +X
        ([-1.0,0.0, 0.0],  [[-1.0,-1.0,-1.0], [-1.0,-1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, 1.0,-1.0]]),  // -X
        ([0.0, 1.0, 0.0],  [[-1.0, 1.0, 1.0], [ 1.0, 1.0, 1.0], [ 1.0, 1.0,-1.0], [-1.0, 1.0,-1.0]]),  // +Y
        ([0.0,-1.0, 0.0],  [[-1.0,-1.0,-1.0], [ 1.0,-1.0,-1.0], [ 1.0,-1.0, 1.0], [-1.0,-1.0, 1.0]]),  // -Y
    ];
    let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

    for (normal, positions) in &faces {
        let base = vertices.len() as u32;
        for (i, pos) in positions.iter().enumerate() {
            vertices.push(GpuVertex {
                position: *pos,
                normal: *normal,
                uv: uvs[i],
                tangent: [1.0, 0.0, 0.0, 1.0],
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    super::obj_loader::compute_tangents(&mut vertices, &indices);
    (vertices, indices)
}

fn generate_disk(segments: u32, flip: bool) -> (Vec<GpuVertex>, Vec<u32>) {
    let sign = if flip { -1.0 } else { 1.0 };
    let mut vertices = vec![GpuVertex {
        position: [0.0, 0.0, 0.0],
        normal: [0.0, 0.0, sign],
        uv: [0.5, 0.5],
        tangent: [1.0, 0.0, 0.0, 1.0],
    }];
    let mut indices = Vec::new();

    for i in 0..=segments {
        let angle = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
        let x = angle.cos();
        let y = angle.sin();
        vertices.push(GpuVertex {
            position: [x, y, 0.0],
            normal: [0.0, 0.0, sign],
            uv: [0.5 + x * 0.5, 0.5 + y * 0.5],
            tangent: [1.0, 0.0, 0.0, 1.0],
        });
    }

    for i in 1..=segments {
        if flip {
            indices.extend_from_slice(&[0, i + 1, i]);
        } else {
            indices.extend_from_slice(&[0, i, i + 1]);
        }
    }

    (vertices, indices)
}

fn generate_cylinder(segments: u32, flip: bool) -> (Vec<GpuVertex>, Vec<u32>) {
    let sign = if flip { -1.0 } else { 1.0 };
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Side wall
    for i in 0..=segments {
        let angle = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
        let x = angle.cos();
        let y = angle.sin();
        let u = i as f32 / segments as f32;

        // Bottom vertex
        vertices.push(GpuVertex {
            position: [x, y, 0.0],
            normal: [sign * x, sign * y, 0.0],
            uv: [u, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
        });
        // Top vertex
        vertices.push(GpuVertex {
            position: [x, y, 1.0],
            normal: [sign * x, sign * y, 0.0],
            uv: [u, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
        });
    }

    for i in 0..segments {
        let b = i * 2;
        if flip {
            indices.extend_from_slice(&[b, b + 3, b + 1, b, b + 2, b + 3]);
        } else {
            indices.extend_from_slice(&[b, b + 1, b + 3, b, b + 3, b + 2]);
        }
    }

    super::obj_loader::compute_tangents(&mut vertices, &indices);
    (vertices, indices)
}

// ── BSDF (material) parsing ──────────────────────────────────────────────────

/// Parsed Mitsuba BSDF mapped to PBR parameters.
#[derive(Clone, Debug)]
struct MitsubaMaterial {
    base_color:    Vec3,
    alpha:         f32, // opacity (1.0 = opaque)
    metallic:      f32,
    roughness:     f32,
    emissive:      Vec3,
    double_sided:  bool,
    base_color_texture: Option<String>, // path to bitmap texture
    // Extended properties for improved BSDF mapping
    int_ior:              Option<f32>,
    ext_ior:              Option<f32>,
    specular_reflectance: Option<Vec3>,
    conductor_material:   Option<String>, // e.g. "Au", "Cu", "Ag"
    eta:                  Option<Vec3>,   // complex IOR (real part)
    k:                    Option<Vec3>,   // complex IOR (imaginary part)
    blend_weight:         Option<f32>,   // for blendbsdf
}

impl Default for MitsubaMaterial {
    fn default() -> Self {
        Self {
            base_color:    Vec3::new(0.5, 0.5, 0.5),
            alpha:         1.0,
            metallic:      0.0,
            roughness:     1.0,
            emissive:      Vec3::ZERO,
            double_sided:  false,
            base_color_texture: None,
            int_ior:              None,
            ext_ior:              None,
            specular_reflectance: None,
            conductor_material:   None,
            eta:                  None,
            k:                    None,
            blend_weight:         None,
        }
    }
}

/// Parse a `<bsdf>` element. Reader positioned just after `<bsdf type="...">`.
/// Returns the material and optionally its `id` for reference resolution.
fn parse_bsdf(reader: &mut Reader<&[u8]>, bsdf_type: &str) -> Result<MitsubaMaterial> {
    let mut mat = MitsubaMaterial::default();
    let mut buf = Vec::new();
    let mut inner_bsdfs: Vec<MitsubaMaterial> = Vec::new();
    let mut skip_buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf).context("parse_bsdf")? {
            Event::Empty(ref e) => {
                let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                handle_bsdf_property(e, &tag, &mut mat);
            }
            Event::Start(ref e) => {
                let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                match tag.as_str() {
                    "bsdf" => {
                        let inner_type = attr_str(e, "type").unwrap_or_default();
                        inner_bsdfs.push(parse_bsdf(reader, &inner_type)?);
                    }
                    "texture" => {
                        let tex_type = attr_str(e, "type").unwrap_or_default();
                        let tex_name = attr_str(e, "name").unwrap_or_default();
                        let is_color_slot = tex_name == "reflectance"
                            || tex_name == "diffuse_reflectance"
                            || tex_name == "base_color";
                        match tex_type.as_str() {
                            "bitmap" => {
                                let tex_props = parse_properties(reader, b"texture")?;
                                if let Some(filename) = tex_props.get_string("filename")
                                    && is_color_slot
                                {
                                    mat.base_color_texture = Some(filename);
                                }
                            }
                            "checkerboard" => {
                                let tex_props = parse_properties(reader, b"texture")?;
                                if is_color_slot {
                                    let color0 = tex_props.get_color("color0")
                                        .unwrap_or(Vec3::new(0.4, 0.4, 0.4));
                                    let color1 = tex_props.get_color("color1")
                                        .unwrap_or(Vec3::new(0.2, 0.2, 0.2));
                                    mat.base_color_texture = Some(
                                        format!("__checkerboard:{}:{}:{}:{}:{}:{}",
                                            color0.x, color0.y, color0.z,
                                            color1.x, color1.y, color1.z)
                                    );
                                }
                            }
                            _ => {
                                log::warn!("Unsupported texture type: {tex_type}");
                                let end = e.to_end().into_owned();
                                reader.read_to_end_into(end.name(), &mut skip_buf)?;
                                skip_buf.clear();
                            }
                        }
                    }
                    _ => {
                        let end = e.to_end().into_owned();
                        reader.read_to_end_into(end.name(), &mut skip_buf)?;
                        skip_buf.clear();
                    }
                }
            }
            Event::End(ref e) if e.name().as_ref() == b"bsdf" => break,
            Event::Eof => anyhow::bail!("unexpected EOF inside <bsdf>"),
            _ => {}
        }
        buf.clear();
    }

    // Apply BSDF-type-specific mapping
    apply_bsdf_type_mapping(&mut mat, bsdf_type);

    // Handle composite BSDF types that depend on inner BSDFs
    match bsdf_type {
        "twosided" => {
            if let Some(inner) = inner_bsdfs.into_iter().next() {
                mat.base_color = inner.base_color;
                mat.alpha = inner.alpha;
                mat.metallic = inner.metallic;
                mat.roughness = inner.roughness;
                mat.emissive = inner.emissive;
                mat.base_color_texture = inner.base_color_texture;
            }
            mat.double_sided = true;
        }
        "blendbsdf" => {
            let w = mat.blend_weight.unwrap_or(0.5);
            let mut iter = inner_bsdfs.into_iter();
            if let (Some(a), Some(b)) = (iter.next(), iter.next()) {
                mat.base_color = a.base_color * (1.0 - w) + b.base_color * w;
                mat.alpha = a.alpha * (1.0 - w) + b.alpha * w;
                mat.metallic = a.metallic * (1.0 - w) + b.metallic * w;
                mat.roughness = a.roughness * (1.0 - w) + b.roughness * w;
                mat.emissive = a.emissive * (1.0 - w) + b.emissive * w;
                mat.double_sided = a.double_sided || b.double_sided;
                mat.base_color_texture = a.base_color_texture.or(b.base_color_texture);
            }
        }
        "mask" => {
            // Opacity mask: inner BSDF provides surface properties, opacity from 'opacity' property
            if let Some(inner) = inner_bsdfs.into_iter().next() {
                mat.base_color = inner.base_color;
                mat.metallic = inner.metallic;
                mat.roughness = inner.roughness;
                mat.emissive = inner.emissive;
                mat.base_color_texture = inner.base_color_texture;
                mat.double_sided = inner.double_sided;
            }
            // mat.alpha is already set from 'opacity' float property (handled below)
        }
        "null" => {
            mat.alpha = 0.0;
        }
        _ => {}
    }

    Ok(mat)
}

/// Handle a single property element inside a BSDF.
fn handle_bsdf_property(e: &quick_xml::events::BytesStart, tag: &str, mat: &mut MitsubaMaterial) {
    let name = match attr_str(e, "name") {
        Some(n) => n,
        None => return,
    };

    match tag {
        "rgb" | "color" | "spectrum" => {
            if let Some(v) = attr_str(e, "value")
                && let Ok(c) = parse_rgb_value(&v)
            {
                match name.as_str() {
                    "reflectance" | "diffuse_reflectance" | "base_color" => mat.base_color = c,
                    "specular_reflectance" => mat.specular_reflectance = Some(c),
                    "eta" => mat.eta = Some(c),
                    "k" => mat.k = Some(c),
                    _ => {}
                }
            }
        }
        "float" => {
            if let Some(v) = attr_f32(e, "value") {
                match name.as_str() {
                    "alpha" => mat.roughness = v,
                    "int_ior" => mat.int_ior = Some(v),
                    "ext_ior" => mat.ext_ior = Some(v),
                    "opacity" => mat.alpha = v,
                    "weight" => mat.blend_weight = Some(v),
                    _ => {}
                }
            }
        }
        "string" => {
            if let Some(v) = attr_str(e, "value")
                && name == "material"
            {
                mat.conductor_material = Some(v);
            }
        }
        _ => {}
    }
}

/// Apply BSDF type-specific PBR parameter overrides.
fn apply_bsdf_type_mapping(mat: &mut MitsubaMaterial, bsdf_type: &str) {
    match bsdf_type {
        "diffuse" => {
            mat.metallic = 0.0;
            mat.roughness = 1.0;
        }
        "roughplastic" | "plastic" => {
            mat.metallic = 0.0;
            if bsdf_type == "plastic" {
                mat.roughness = 0.01;
            }
            // Compute F0 from IOR for dielectric fresnel
            let int_ior = mat.int_ior.unwrap_or(1.49); // default polypropylene
            let ext_ior = mat.ext_ior.unwrap_or(1.000277);
            let f0_scalar = ((int_ior - ext_ior) / (int_ior + ext_ior)).powi(2);
            // For plastics, the specular reflectance modulates the Fresnel
            if let Some(spec) = mat.specular_reflectance {
                // Scale base color to account for energy conservation
                mat.base_color *= 1.0 - spec.x.max(spec.y).max(spec.z) * f0_scalar;
            }
            let _ = f0_scalar; // F0 used implicitly via metallic=0 + dielectric fresnel in shader
        }
        "roughconductor" | "conductor" => {
            mat.metallic = 1.0;
            if bsdf_type == "conductor" {
                mat.roughness = 0.01;
            }
            // Derive base_color (F0) from conductor properties
            let f0 = conductor_f0(mat);
            mat.base_color = f0;
        }
        "dielectric" | "roughdielectric" | "thindielectric" => {
            mat.metallic = 0.0;
            if bsdf_type == "dielectric" || bsdf_type == "thindielectric" {
                mat.roughness = 0.0;
            }
            // Approximate glass: semi-transparent with specular highlights
            let int_ior = mat.int_ior.unwrap_or(1.5);
            let ext_ior = mat.ext_ior.unwrap_or(1.000277);
            let f0 = ((int_ior - ext_ior) / (int_ior + ext_ior)).powi(2);
            mat.base_color = Vec3::ONE;
            mat.alpha = f0.max(0.05); // at least slightly visible
            mat.double_sided = true;
        }
        "twosided" => {
            // Handled after inner BSDF is parsed
        }
        _ => {
            if bsdf_type != "null" && bsdf_type != "mask" && bsdf_type != "blendbsdf" {
                log::warn!("Unsupported BSDF type '{bsdf_type}', using default gray diffuse");
            }
        }
    }
}

/// Compute F0 (normal-incidence reflectance) for a conductor material.
fn conductor_f0(mat: &MitsubaMaterial) -> Vec3 {
    // If eta and k are specified directly, compute from complex IOR
    if let (Some(eta), Some(k)) = (mat.eta, mat.k) {
        return compute_conductor_f0(eta, k);
    }

    // If a named material is given, look up from table
    if let Some(ref name) = mat.conductor_material
        && let Some(f0) = conductor_f0_table(name)
    {
        return if let Some(spec) = mat.specular_reflectance {
            f0 * spec
        } else {
            f0
        };
    }

    // If specular_reflectance is given, use directly
    if let Some(spec) = mat.specular_reflectance {
        return spec;
    }

    // Fallback: generic shiny metal
    Vec3::new(0.9, 0.9, 0.9)
}

/// F0 from complex IOR: F0 = ((n-1)² + k²) / ((n+1)² + k²) per channel
fn compute_conductor_f0(eta: Vec3, k: Vec3) -> Vec3 {
    let f = |n: f32, kk: f32| -> f32 {
        ((n - 1.0).powi(2) + kk.powi(2)) / ((n + 1.0).powi(2) + kk.powi(2))
    };
    Vec3::new(f(eta.x, k.x), f(eta.y, k.y), f(eta.z, k.z))
}

/// Lookup table for common conductor materials → F0 at normal incidence.
/// Values are approximate sRGB F0 derived from spectral data.
fn conductor_f0_table(name: &str) -> Option<Vec3> {
    Some(match name {
        "Au" | "gold"           => Vec3::new(1.000, 0.710, 0.290),
        "Ag" | "silver"         => Vec3::new(0.950, 0.930, 0.880),
        "Cu" | "copper"         => Vec3::new(0.950, 0.640, 0.540),
        "Al" | "aluminum"       => Vec3::new(0.910, 0.920, 0.920),
        "Fe" | "iron"           => Vec3::new(0.560, 0.570, 0.580),
        "Cr" | "chromium"       => Vec3::new(0.550, 0.550, 0.550),
        "Ni" | "nickel"         => Vec3::new(0.660, 0.610, 0.530),
        "Ti" | "titanium"       => Vec3::new(0.540, 0.500, 0.430),
        "W"  | "tungsten"       => Vec3::new(0.500, 0.510, 0.540),
        "Pt" | "platinum"       => Vec3::new(0.670, 0.640, 0.590),
        "Pb" | "lead"           => Vec3::new(0.630, 0.630, 0.630),
        "Zn" | "zinc"           => Vec3::new(0.640, 0.620, 0.580),
        "V"  | "vanadium"       => Vec3::new(0.530, 0.500, 0.470),
        "Hg" | "mercury"        => Vec3::new(0.780, 0.780, 0.780),
        "none" | "default"      => Vec3::new(0.900, 0.900, 0.900),
        _ => return None,
    })
}

// ── Property bag (generic Mitsuba property parsing) ──────────────────────────

use std::collections::HashMap;

/// A bag of named Mitsuba properties extracted from an element's children.
struct Properties {
    strings: HashMap<String, String>,
    floats:  HashMap<String, f32>,
    colors:  HashMap<String, Vec3>,
    points:  HashMap<String, Vec3>,
    vectors: HashMap<String, Vec3>,
    transform: Option<Mat4>,
}

impl Properties {
    fn get_string(&self, name: &str) -> Option<String> {
        self.strings.get(name).cloned()
    }
    fn get_float(&self, name: &str) -> Option<f32> {
        self.floats.get(name).copied()
    }
    #[allow(dead_code)]
    fn get_color(&self, name: &str) -> Option<Vec3> {
        self.colors.get(name).copied()
    }
    fn get_point(&self, name: &str) -> Option<Vec3> {
        self.points.get(name).copied()
    }
    fn get_vec3(&self, name: &str) -> Option<Vec3> {
        self.vectors.get(name).copied()
    }
    #[allow(dead_code)]
    fn get_transform(&self) -> Option<Mat4> {
        self.transform
    }

    /// Get a color+intensity from a named RGB property.
    /// Mitsuba encodes emitter power as an RGB value where the magnitude is the intensity.
    fn get_color_intensity(&self, name: &str, default_color: Vec3, default_intensity: f32) -> (Vec3, f32) {
        if let Some(rgb) = self.colors.get(name) {
            let max = rgb.x.max(rgb.y).max(rgb.z);
            if max > 0.0 {
                (*rgb / max, max)
            } else {
                (default_color, 0.0)
            }
        } else {
            (default_color, default_intensity)
        }
    }
}

/// Parse all child property elements until the closing tag with `end_tag` name.
fn parse_properties(reader: &mut Reader<&[u8]>, end_tag: &[u8]) -> Result<Properties> {
    let mut props = Properties {
        strings: HashMap::new(),
        floats: HashMap::new(),
        colors: HashMap::new(),
        points: HashMap::new(),
        vectors: HashMap::new(),
        transform: None,
    };
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf).context("parse_properties")? {
            Event::Empty(ref e) => {
                parse_property_element(e, &mut props);
            }
            Event::Start(ref e) => {
                let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                match tag.as_str() {
                    "transform" => {
                        props.transform = Some(parse_transform(reader)?);
                    }
                    _ => {
                        // Skip unknown nested elements
                        let end = e.to_end().into_owned();
                        let mut skip_buf = Vec::new();
                        reader.read_to_end_into(end.name(), &mut skip_buf)?;
                    }
                }
            }
            Event::End(ref e) if e.name().as_ref() == end_tag => break,
            Event::Eof => anyhow::bail!("unexpected EOF inside <{}>", String::from_utf8_lossy(end_tag)),
            _ => {}
        }
        buf.clear();
    }

    Ok(props)
}

/// Parse a single empty property element like `<float name="x" value="1.0"/>`.
fn parse_property_element(e: &quick_xml::events::BytesStart, props: &mut Properties) {
    let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
    let name = match attr_str(e, "name") {
        Some(n) => n,
        None => return,
    };

    match tag.as_str() {
        "string" => {
            if let Some(v) = attr_str(e, "value") {
                props.strings.insert(name, v);
            }
        }
        "float" | "integer" => {
            if let Some(v) = attr_f32(e, "value") {
                props.floats.insert(name, v);
            }
        }
        "rgb" | "color" | "spectrum" => {
            if let Some(v) = attr_str(e, "value")
                && let Ok(c) = parse_rgb_value(&v)
            {
                props.colors.insert(name, c);
            }
        }
        "point" => {
            if let (Some(x), Some(y), Some(z)) = (attr_f32(e, "x"), attr_f32(e, "y"), attr_f32(e, "z")) {
                props.points.insert(name, Vec3::new(x, y, z));
            } else if let Some(v) = attr_str(e, "value")
                && let Ok(p) = parse_vec3(&v)
            {
                props.points.insert(name, p);
            }
        }
        "vector" => {
            if let (Some(x), Some(y), Some(z)) = (attr_f32(e, "x"), attr_f32(e, "y"), attr_f32(e, "z")) {
                props.vectors.insert(name, Vec3::new(x, y, z));
            } else if let Some(v) = attr_str(e, "value")
                && let Ok(p) = parse_vec3(&v)
            {
                props.vectors.insert(name, p);
            }
        }
        "boolean" => {
            if let Some(v) = attr_str(e, "value") {
                let b = v == "true" || v == "1";
                props.floats.insert(name, if b { 1.0 } else { 0.0 });
            }
        }
        _ => {}
    }
}

/// Parse an RGB color value string. Supports "r, g, b" or "r g b" or single value.
fn parse_rgb_value(s: &str) -> Result<Vec3> {
    let v = parse_floats(s);
    match v.len() {
        1 => Ok(Vec3::splat(v[0])),
        3 => Ok(Vec3::new(v[0], v[1], v[2])),
        n => anyhow::bail!("expected 1 or 3 components for RGB, got {n}: '{s}'"),
    }
}

// ── XML helpers ──────────────────────────────────────────────────────────────

/// Get an attribute value by name from a quick-xml BytesStart element.
fn attr_str<'a>(
    e: &'a quick_xml::events::BytesStart<'a>,
    name: &str,
) -> Option<String> {
    e.attributes()
        .filter_map(|a| a.ok())
        .find(|a| a.key.as_ref() == name.as_bytes())
        .map(|a| String::from_utf8_lossy(&a.value).into_owned())
}

fn attr_f32(e: &quick_xml::events::BytesStart, name: &str) -> Option<f32> {
    attr_str(e, name).and_then(|s| s.trim().parse().ok())
}

/// Parse a comma-or-whitespace separated list of f32 values.
fn parse_floats(s: &str) -> Vec<f32> {
    s.split(|c: char| c == ',' || c.is_whitespace())
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .filter_map(|t| t.parse().ok())
        .collect()
}

/// Parse a "x, y, z" or "x y z" string into Vec3.
fn parse_vec3(s: &str) -> Result<Vec3> {
    let v = parse_floats(s);
    if v.len() >= 3 {
        Ok(Vec3::new(v[0], v[1], v[2]))
    } else {
        anyhow::bail!("expected 3 components, got {}: '{s}'", v.len())
    }
}

// ── Transform parsing ────────────────────────────────────────────────────────

/// Parse a `<transform>` element and its children, returning the composed Mat4.
/// The reader must be positioned just after the opening `<transform>` tag.
/// On return the reader is positioned just after the closing `</transform>`.
fn parse_transform(reader: &mut Reader<&[u8]>) -> Result<Mat4> {
    let mut result = Mat4::IDENTITY;
    let mut buf = Vec::new();
    let mut skip_buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf).context("parse_transform")? {
            Event::Empty(ref e) => {
                let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                let mat = parse_transform_child(e, &tag)?;
                result = mat * result;
            }
            Event::Start(ref e) => {
                let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                let mat = parse_transform_child(e, &tag)?;
                result = mat * result;
                // Skip to the matching closing tag
                let end = e.to_end().into_owned();
                reader.read_to_end_into(end.name(), &mut skip_buf)?;
                skip_buf.clear();
            }
            Event::End(ref e) if e.name().as_ref() == b"transform" => break,
            Event::Eof => anyhow::bail!("unexpected EOF inside <transform>"),
            _ => {}
        }
        buf.clear();
    }

    Ok(result)
}

fn parse_transform_child(e: &quick_xml::events::BytesStart, tag: &str) -> Result<Mat4> {
    match tag {
        "matrix"    => parse_matrix_element(e),
        "translate" => parse_translate_element(e),
        "rotate"    => parse_rotate_element(e),
        "scale"     => parse_scale_element(e),
        "lookat"    => parse_lookat_element(e),
        _ => {
            log::warn!("Unknown transform child: <{tag}>");
            Ok(Mat4::IDENTITY)
        }
    }
}

fn parse_matrix_element(e: &quick_xml::events::BytesStart) -> Result<Mat4> {
    let value = attr_str(e, "value").context("<matrix> missing 'value'")?;
    let v = parse_floats(&value);
    if v.len() != 16 {
        anyhow::bail!("<matrix> expected 16 floats, got {}", v.len());
    }
    // Mitsuba uses row-major order
    Ok(Mat4::from_cols_array(&[
        v[0], v[4], v[8],  v[12],
        v[1], v[5], v[9],  v[13],
        v[2], v[6], v[10], v[14],
        v[3], v[7], v[11], v[15],
    ]))
}

fn parse_translate_element(e: &quick_xml::events::BytesStart) -> Result<Mat4> {
    // Can be <translate value="x y z"/> or <translate x="" y="" z=""/>
    if let Some(value) = attr_str(e, "value") {
        let v = parse_vec3(&value)?;
        Ok(Mat4::from_translation(v))
    } else {
        let x = attr_f32(e, "x").unwrap_or(0.0);
        let y = attr_f32(e, "y").unwrap_or(0.0);
        let z = attr_f32(e, "z").unwrap_or(0.0);
        Ok(Mat4::from_translation(Vec3::new(x, y, z)))
    }
}

fn parse_rotate_element(e: &quick_xml::events::BytesStart) -> Result<Mat4> {
    let x = attr_f32(e, "x").unwrap_or(0.0);
    let y = attr_f32(e, "y").unwrap_or(0.0);
    let z = attr_f32(e, "z").unwrap_or(0.0);
    let angle_deg = attr_f32(e, "angle").unwrap_or(0.0);
    let axis = Vec3::new(x, y, z);
    if axis.length_squared() < 1e-12 {
        return Ok(Mat4::IDENTITY);
    }
    Ok(Mat4::from_axis_angle(axis.normalize(), angle_deg.to_radians()))
}

fn parse_scale_element(e: &quick_xml::events::BytesStart) -> Result<Mat4> {
    if let Some(value) = attr_str(e, "value") {
        let v = parse_floats(&value);
        if v.len() == 1 {
            Ok(Mat4::from_scale(Vec3::splat(v[0])))
        } else if v.len() >= 3 {
            Ok(Mat4::from_scale(Vec3::new(v[0], v[1], v[2])))
        } else {
            anyhow::bail!("<scale value> expected 1 or 3 floats, got {}", v.len())
        }
    } else {
        let x = attr_f32(e, "x").unwrap_or(1.0);
        let y = attr_f32(e, "y").unwrap_or(1.0);
        let z = attr_f32(e, "z").unwrap_or(1.0);
        Ok(Mat4::from_scale(Vec3::new(x, y, z)))
    }
}

fn parse_lookat_element(e: &quick_xml::events::BytesStart) -> Result<Mat4> {
    let origin_str = attr_str(e, "origin").context("<lookat> missing 'origin'")?;
    let target_str = attr_str(e, "target").context("<lookat> missing 'target'")?;
    let up_str = attr_str(e, "up").unwrap_or_else(|| "0, 1, 0".to_string());

    let origin = parse_vec3(&origin_str)?;
    let target = parse_vec3(&target_str)?;
    let up = parse_vec3(&up_str)?;

    // Build the camera-to-world matrix in Mitsuba's convention:
    // column 2 (+Z) = forward direction (toward target).
    let dir = (target - origin).normalize();
    let left = up.cross(dir).normalize();
    let new_up = dir.cross(left);
    Ok(Mat4::from_cols(
        left.extend(0.0),
        new_up.extend(0.0),
        dir.extend(0.0),
        origin.extend(1.0),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_mat4(a: &Mat4, b: &Mat4, eps: f32) -> bool {
        for c in 0..4 {
            let ac = a.col(c);
            let bc = b.col(c);
            for r in 0..4 {
                if (ac[r] - bc[r]).abs() > eps {
                    return false;
                }
            }
        }
        true
    }

    fn parse_transform_str(xml: &str) -> Result<Mat4> {
        let full = format!("<transform>{xml}</transform>");
        let mut reader = Reader::from_str(&full);
        // Advance past the opening <transform> tag
        let mut buf = Vec::new();
        loop {
            match reader.read_event_into(&mut buf)? {
                Event::Start(ref e) if e.name().as_ref() == b"transform" => break,
                Event::Eof => anyhow::bail!("no <transform> found"),
                _ => {}
            }
            buf.clear();
        }
        parse_transform(&mut reader)
    }

    #[test]
    fn test_translate() {
        let m = parse_transform_str(r#"<translate x="1" y="2" z="3"/>"#).unwrap();
        let expected = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        assert!(approx_eq_mat4(&m, &expected, 1e-6));
    }

    #[test]
    fn test_translate_value() {
        let m = parse_transform_str(r#"<translate value="4, 5, 6"/>"#).unwrap();
        let expected = Mat4::from_translation(Vec3::new(4.0, 5.0, 6.0));
        assert!(approx_eq_mat4(&m, &expected, 1e-6));
    }

    #[test]
    fn test_scale_uniform() {
        let m = parse_transform_str(r#"<scale value="2"/>"#).unwrap();
        let expected = Mat4::from_scale(Vec3::splat(2.0));
        assert!(approx_eq_mat4(&m, &expected, 1e-6));
    }

    #[test]
    fn test_scale_nonuniform() {
        let m = parse_transform_str(r#"<scale x="1" y="2" z="3"/>"#).unwrap();
        let expected = Mat4::from_scale(Vec3::new(1.0, 2.0, 3.0));
        assert!(approx_eq_mat4(&m, &expected, 1e-6));
    }

    #[test]
    fn test_rotate() {
        let m = parse_transform_str(r#"<rotate x="0" y="1" z="0" angle="90"/>"#).unwrap();
        let expected = Mat4::from_axis_angle(Vec3::Y, 90f32.to_radians());
        assert!(approx_eq_mat4(&m, &expected, 1e-5));
    }

    #[test]
    fn test_matrix() {
        let m = parse_transform_str(
            r#"<matrix value="1 0 0 5  0 1 0 6  0 0 1 7  0 0 0 1"/>"#,
        ).unwrap();
        // Row-major input: translation at row entries [0][3], [1][3], [2][3]
        let expected = Mat4::from_translation(Vec3::new(5.0, 6.0, 7.0));
        assert!(approx_eq_mat4(&m, &expected, 1e-6));
    }

    #[test]
    fn test_lookat() {
        let m = parse_transform_str(
            r#"<lookat origin="0, 0, 5" target="0, 0, 0" up="0, 1, 0"/>"#,
        ).unwrap();
        // Camera at (0,0,5) looking at origin.
        // Column 3 = camera position
        let pos = m.col(3).truncate();
        assert!((pos - Vec3::new(0.0, 0.0, 5.0)).length() < 1e-5);
        // Column 2 = forward direction (+Z in Mitsuba convention) = toward target
        let fwd = m.col(2).truncate().normalize();
        assert!((fwd - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-5);
    }

    fn parse_sensor_str(xml: &str) -> Result<MitsubaCameraDesc> {
        let full = format!("<sensor type=\"perspective\">{xml}</sensor>");
        let mut reader = Reader::from_str(&full);
        let mut buf = Vec::new();
        loop {
            match reader.read_event_into(&mut buf)? {
                Event::Start(ref e) if e.name().as_ref() == b"sensor" => break,
                Event::Eof => anyhow::bail!("no <sensor> found"),
                _ => {}
            }
            buf.clear();
        }
        parse_sensor(&mut reader)
    }

    #[test]
    fn test_sensor_basic() {
        let desc = parse_sensor_str(r#"
            <float name="fov" value="60"/>
            <float name="near_clip" value="0.1"/>
            <float name="far_clip" value="500"/>
            <transform name="to_world">
                <lookat origin="0, 5, 10" target="0, 0, 0" up="0, 1, 0"/>
            </transform>
        "#).unwrap();

        assert!((desc.fov - 60.0).abs() < 1e-6);
        assert_eq!(desc.fov_axis, "x"); // default
        assert!((desc.near - 0.1).abs() < 1e-6);
        assert!((desc.far - 500.0).abs() < 1e-6);

        let cam = camera_from_mitsuba(&desc);
        // Camera should be at (0, 5, 10), looking at origin
        let pos = cam.camera.position;
        assert!((pos.x).abs() < 0.01, "x={}", pos.x);
        assert!((pos.y - 5.0).abs() < 0.01, "y={}", pos.y);
        assert!((pos.z - 10.0).abs() < 0.01, "z={}", pos.z);
        // Target should be at origin
        assert!(cam.target.length() < 0.01, "target={:?}", cam.target);
        // FOV should be converted from horizontal (60°) to vertical
        // With default 768x576 (aspect=4/3): vfov = 2*atan(tan(30°)/1.333) ≈ 46.8°
        let expected_vfov = 2.0 * ((60f32.to_radians() * 0.5).tan() / (768.0 / 576.0)).atan();
        assert!((cam.camera.fov_y - expected_vfov).abs() < 0.01,
            "fov_y={} expected={}", cam.camera.fov_y.to_degrees(), expected_vfov.to_degrees());
    }

    fn parse_bsdf_str(xml: &str, bsdf_type: &str) -> Result<MitsubaMaterial> {
        let full = format!("<bsdf type=\"{bsdf_type}\">{xml}</bsdf>");
        let mut reader = Reader::from_str(&full);
        let mut buf = Vec::new();
        loop {
            match reader.read_event_into(&mut buf)? {
                Event::Start(ref e) if e.name().as_ref() == b"bsdf" => break,
                Event::Eof => anyhow::bail!("no <bsdf> found"),
                _ => {}
            }
            buf.clear();
        }
        parse_bsdf(&mut reader, bsdf_type)
    }

    #[test]
    fn test_diffuse_bsdf() {
        let m = parse_bsdf_str(r#"
            <rgb name="reflectance" value="0.8, 0.2, 0.1"/>
        "#, "diffuse").unwrap();
        assert!((m.base_color.x - 0.8).abs() < 1e-6);
        assert!((m.base_color.y - 0.2).abs() < 1e-6);
        assert!((m.metallic).abs() < 1e-6);
        assert!((m.roughness - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_roughplastic_bsdf() {
        let m = parse_bsdf_str(r#"
            <rgb name="diffuse_reflectance" value="0.3, 0.5, 0.3"/>
            <float name="alpha" value="0.1"/>
        "#, "roughplastic").unwrap();
        assert!((m.base_color.y - 0.5).abs() < 1e-6);
        assert!((m.roughness - 0.1).abs() < 1e-6);
        assert!((m.metallic).abs() < 1e-6);
    }

    #[test]
    fn test_conductor_bsdf() {
        let m = parse_bsdf_str(r#""#, "conductor").unwrap();
        assert!((m.metallic - 1.0).abs() < 1e-6);
        assert!(m.roughness < 0.05);
    }

    #[test]
    fn test_twosided_bsdf() {
        let m = parse_bsdf_str(r#"
            <bsdf type="diffuse">
                <rgb name="reflectance" value="0.9, 0.1, 0.1"/>
            </bsdf>
        "#, "twosided").unwrap();
        assert!(m.double_sided);
        assert!((m.base_color.x - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_bsdf_with_texture() {
        let m = parse_bsdf_str(r#"
            <texture type="bitmap" name="reflectance">
                <string name="filename" value="textures/wood.jpg"/>
            </texture>
        "#, "diffuse").unwrap();
        assert_eq!(m.base_color_texture.as_deref(), Some("textures/wood.jpg"));
    }

    fn parse_emitter_str(xml: &str, emitter_type: &str) -> Result<MitsubaEmitter> {
        let full = format!("<emitter type=\"{emitter_type}\">{xml}</emitter>");
        let mut reader = Reader::from_str(&full);
        let mut buf = Vec::new();
        loop {
            match reader.read_event_into(&mut buf)? {
                Event::Start(ref e) if e.name().as_ref() == b"emitter" => break,
                Event::Eof => anyhow::bail!("no <emitter> found"),
                _ => {}
            }
            buf.clear();
        }
        parse_emitter(&mut reader, emitter_type)
    }

    #[test]
    fn test_point_light() {
        let e = parse_emitter_str(r#"
            <point name="position" x="1" y="2" z="3"/>
            <rgb name="intensity" value="100, 50, 50"/>
        "#, "point").unwrap();
        match e {
            MitsubaEmitter::Point { position, color, intensity } => {
                assert!((position.x - 1.0).abs() < 1e-6);
                assert!((position.y - 2.0).abs() < 1e-6);
                assert!((position.z - 3.0).abs() < 1e-6);
                assert!(intensity > 0.0);
                assert!(color.x >= color.y); // red dominant
            }
            _ => panic!("expected Point emitter"),
        }
    }

    #[test]
    fn test_directional_light() {
        let e = parse_emitter_str(r#"
            <vector name="direction" x="0" y="-1" z="0"/>
            <rgb name="irradiance" value="3, 3, 3"/>
        "#, "directional").unwrap();
        match e {
            MitsubaEmitter::Directional { direction, .. } => {
                assert!((direction.y - (-1.0)).abs() < 1e-6);
            }
            _ => panic!("expected Directional emitter"),
        }
    }

    #[test]
    fn test_envmap_emitter() {
        let e = parse_emitter_str(r#"
            <string name="filename" value="textures/env.exr"/>
            <float name="scale" value="0.5"/>
        "#, "envmap").unwrap();
        match e {
            MitsubaEmitter::EnvMap { filename, scale } => {
                assert_eq!(filename, "textures/env.exr");
                assert!((scale - 0.5).abs() < 1e-6);
            }
            _ => panic!("expected EnvMap emitter"),
        }
    }

    #[test]
    fn test_gpu_light_from_point() {
        let emitter = MitsubaEmitter::Point {
            position: Vec3::new(1.0, 2.0, 3.0),
            color: Vec3::ONE,
            intensity: 10.0,
        };
        let light = gpu_light_from_emitter(&emitter, &Mat4::IDENTITY).unwrap();
        assert!((light.position_or_direction[0] - 1.0).abs() < 1e-6);
        assert!((light.position_or_direction[3] - 1.0).abs() < 1e-6); // w=1 for point
        assert!((light.intensity - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_composed_transforms() {
        // translate then scale: first translate by (1,0,0), then scale by 2
        let m = parse_transform_str(
            r#"<translate x="1" y="0" z="0"/><scale value="2"/>"#,
        ).unwrap();
        // Composing left-to-right: T * S
        // A point (0,0,0) -> translate to (1,0,0) -> scale to (2,0,0)
        let p = m.transform_point3(Vec3::ZERO);
        assert!((p.x - 2.0).abs() < 1e-5);
        assert!((p.y - 0.0).abs() < 1e-5);
        assert!((p.z - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_generate_sphere() {
        let (verts, indices) = generate_sphere(Vec3::ZERO, 1.0, 16, 8, false);
        assert!(!verts.is_empty());
        assert!(!indices.is_empty());
        assert_eq!(indices.len() % 3, 0); // all triangles
    }

    #[test]
    fn test_generate_rectangle() {
        let (verts, indices) = generate_rectangle(false);
        assert_eq!(verts.len(), 4);
        assert_eq!(indices.len(), 6);
    }

    #[test]
    fn test_generate_cube() {
        let (verts, indices) = generate_cube();
        assert_eq!(verts.len(), 24); // 6 faces * 4 vertices
        assert_eq!(indices.len(), 36); // 6 faces * 2 triangles * 3 indices
    }
}
