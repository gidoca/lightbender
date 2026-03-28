use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use glam::{Quat, Vec3};
use serde::Deserialize;

use super::{gltf_loader, Scene, Transform};
use crate::renderer::Renderer;

// ── JSON schema ───────────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct SceneDescription {
    pub camera:      CameraDesc,
    #[serde(default)]
    pub lights:      Vec<LightDesc>,
    /// Named shader pairs: "name" -> { vert, frag } SPIR-V paths.
    #[serde(default)]
    pub shaders:     HashMap<String, ShaderDesc>,
    #[serde(default)]
    pub models:      Vec<ModelDesc>,
    #[serde(default)]
    pub environment: EnvironmentDesc,
}

#[derive(Deserialize)]
pub struct ShaderDesc {
    pub vert: String,
    pub frag: String,
}

#[derive(Deserialize)]
pub struct CameraDesc {
    #[serde(default)]
    pub target:   [f32; 3],
    #[serde(default = "default_distance")]
    pub distance: f32,
    #[serde(default)]
    pub yaw:      f32,
    #[serde(default = "default_pitch")]
    pub pitch:    f32,
    #[serde(default = "default_fov")]
    pub fov_y:    f32,
    #[serde(default = "default_near")]
    pub near:     f32,
    #[serde(default = "default_far")]
    pub far:      f32,
}

fn default_distance() -> f32 { 5.0 }
fn default_pitch()    -> f32 { 20.0 }
fn default_fov()      -> f32 { 60.0 }
fn default_near()     -> f32 { 0.01 }
fn default_far()      -> f32 { 1000.0 }

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(dead_code)]
pub enum LightDesc {
    Directional {
        #[serde(default)]
        name:      String,
        direction: [f32; 3],
        #[serde(default = "white")]
        color:     [f32; 3],
        #[serde(default = "one")]
        intensity: f32,
    },
    Point {
        #[serde(default)]
        name:      String,
        position:  [f32; 3],
        #[serde(default = "white")]
        color:     [f32; 3],
        #[serde(default = "one")]
        intensity: f32,
        #[serde(default = "default_range")]
        range:     f32,
    },
    Spot {
        #[serde(default)]
        name:            String,
        position:        [f32; 3],
        direction:       [f32; 3],
        #[serde(default = "white")]
        color:           [f32; 3],
        #[serde(default = "one")]
        intensity:       f32,
        #[serde(default = "default_range")]
        range:           f32,
        inner_cone_angle: f32,
        outer_cone_angle: f32,
    },
}

fn white() -> [f32; 3] { [1.0, 1.0, 1.0] }
fn one()   -> f32      { 1.0 }
fn default_range() -> f32 { 10.0 }

#[derive(Deserialize)]
pub struct TransformDesc {
    #[serde(default)]
    pub translation: [f32; 3],
    /// Quaternion [x, y, z, w]
    #[serde(default = "identity_quat")]
    pub rotation:    [f32; 4],
    #[serde(default = "one_scale")]
    pub scale:       [f32; 3],
}

fn identity_quat() -> [f32; 4] { [0.0, 0.0, 0.0, 1.0] }
fn one_scale()     -> [f32; 3] { [1.0, 1.0, 1.0] }

impl Default for TransformDesc {
    fn default() -> Self {
        Self {
            translation: [0.0; 3],
            rotation:    identity_quat(),
            scale:       one_scale(),
        }
    }
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct ModelDesc {
    #[serde(default)]
    pub name:      String,
    pub path:      String,
    #[serde(default)]
    pub transform: TransformDesc,
    /// Named shader to use for this model's primitives (must be in `shaders`).
    pub shader:    Option<String>,
}

#[derive(Deserialize, Default)]
#[allow(dead_code)]
pub struct EnvironmentDesc {
    #[serde(default)]
    pub ambient_color:     [f32; 3],
    #[serde(default = "default_ambient")]
    pub ambient_intensity: f32,
    #[serde(default)]
    pub background_color:  [f32; 3],
    /// Path to an HDR/EXR equirectangular environment map.
    #[serde(default)]
    pub map:               Option<String>,
    /// Scaling factor for environment map IBL contribution (default 1.0).
    #[serde(default = "one")]
    pub map_intensity:     f32,
}

fn default_ambient() -> f32 { 0.3 }

// ── Loader ────────────────────────────────────────────────────────────────────

pub struct LoadedScene {
    pub scene:       Scene,
    pub description: SceneDescription,
}

pub fn load_scene(renderer: &Renderer, scene_path: &Path) -> Result<LoadedScene> {
    let json = std::fs::read_to_string(scene_path)
        .with_context(|| format!("read scene file: {}", scene_path.display()))?;
    let desc: SceneDescription =
        serde_json::from_str(&json).context("parse scene JSON")?;

    let base = scene_path.parent().unwrap_or(Path::new("."));

    // Load and merge all models into a single scene
    let ctx = renderer.load_context();
    let mut merged: Option<Scene> = None;
    let mut node_offset = 0usize;
    let mut mesh_offset  = 0usize;
    let mut mat_offset   = 0usize;

    for model_desc in &desc.models {
        let model_path = resolve_path(base, &model_desc.path);
        let mut sub = gltf_loader::load(&ctx, &model_path)
            .with_context(|| format!("load model: {}", model_path.display()))?;

        // Apply the model's shader to all its materials
        if let Some(shader_name) = &model_desc.shader {
            for mat in &mut sub.materials {
                mat.pipeline_name = Some(shader_name.clone());
            }
        }

        // Apply the scene-level transform to all root nodes of this sub-scene
        let tf = Transform {
            translation: Vec3::from(model_desc.transform.translation),
            rotation:    Quat::from_array(model_desc.transform.rotation),
            scale:       Vec3::from(model_desc.transform.scale),
        };

        if let Some(base_scene) = merged.as_mut() {
            // Re-index nodes, meshes, materials, textures
            for node in &mut sub.nodes {
                if let Some(p) = node.parent.as_mut() { *p += node_offset; }
                node.children.iter_mut().for_each(|c| *c += node_offset);
                if let Some(m) = node.mesh.as_mut() { *m += mesh_offset; }
                // Root nodes: apply model transform on top
                if node.parent.is_none() {
                    node.local_transform = compose_transforms(&tf, &node.local_transform);
                }
            }
            for mesh in &mut sub.meshes {
                for prim in &mut mesh.primitives {
                    prim.material += mat_offset;
                }
            }

            node_offset += sub.nodes.len();
            mesh_offset += sub.meshes.len();
            mat_offset  += sub.materials.len();

            base_scene.nodes.extend(sub.nodes);
            base_scene.meshes.extend(sub.meshes);
            base_scene.materials.extend(sub.materials);
            base_scene.textures.extend(sub.textures);
            base_scene.update_world_transforms();
        } else {
            // Apply transform to root nodes
            for node in sub.nodes.iter_mut() {
                if node.parent.is_none() {
                    node.local_transform = compose_transforms(&tf, &node.local_transform);
                }
            }
            sub.update_world_transforms();

            node_offset = sub.nodes.len();
            mesh_offset = sub.meshes.len();
            mat_offset  = sub.materials.len();
            merged = Some(sub);
        }
    }

    // If no models were specified, create an empty scene
    let scene = merged.unwrap_or_else(|| {
        unsafe {
            let pool = renderer.device().create_descriptor_pool(
                &ash::vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&[]),
                None,
            ).unwrap_or(ash::vk::DescriptorPool::null());
            Scene {
                nodes: vec![],
                meshes: vec![],
                materials: vec![],
                textures: vec![],
                world_transforms: vec![],
                descriptor_pool: pool,
            }
        }
    });

    Ok(LoadedScene { scene, description: desc })
}

fn resolve_path(base: &Path, relative: &str) -> PathBuf {
    let p = Path::new(relative);
    if p.is_absolute() { p.to_path_buf() } else { base.join(p) }
}

fn compose_transforms(outer: &Transform, inner: &Transform) -> Transform {
    let m = outer.to_mat4() * inner.to_mat4();
    let (s, r, t) = m.to_scale_rotation_translation();
    Transform { translation: t, rotation: r, scale: s }
}
