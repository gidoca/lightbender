use std::path::Path;

use anyhow::{Context, Result};
use ply_rs::parser::Parser;
use ply_rs::ply::{DefaultElement, Property};

use crate::types::GpuVertex;

pub struct PlyMesh {
    pub vertices: Vec<GpuVertex>,
    pub indices:  Vec<u32>,
}

pub fn load_ply(path: &Path) -> Result<PlyMesh> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("open PLY: {}", path.display()))?;
    let mut reader = std::io::BufReader::new(file);

    let vertex_parser = Parser::<DefaultElement>::new();
    let header = vertex_parser.read_header(&mut reader)
        .with_context(|| format!("parse PLY header: {}", path.display()))?;

    let mut vertices = Vec::new();
    let mut face_indices = Vec::new();

    for (_name, element) in &header.elements {
        match element.name.as_str() {
            "vertex" => {
                let elems = vertex_parser
                    .read_payload_for_element(&mut reader, element, &header)
                    .context("read PLY vertices")?;

                for elem in &elems {
                    let px = prop_f32(elem, "x").unwrap_or(0.0);
                    let py = prop_f32(elem, "y").unwrap_or(0.0);
                    let pz = prop_f32(elem, "z").unwrap_or(0.0);

                    let nx = prop_f32(elem, "nx").unwrap_or(0.0);
                    let ny = prop_f32(elem, "ny").unwrap_or(1.0);
                    let nz = prop_f32(elem, "nz").unwrap_or(0.0);

                    let u = prop_f32(elem, "s")
                        .or_else(|| prop_f32(elem, "texture_u"))
                        .or_else(|| prop_f32(elem, "u"))
                        .unwrap_or(0.0);
                    let v = prop_f32(elem, "t")
                        .or_else(|| prop_f32(elem, "texture_v"))
                        .or_else(|| prop_f32(elem, "v"))
                        .unwrap_or(0.0);

                    vertices.push(GpuVertex {
                        position: [px, py, pz],
                        normal:   [nx, ny, nz],
                        uv:       [u, v],
                        tangent:  [1.0, 0.0, 0.0, 1.0],
                    });
                }
            }
            "face" => {
                let elems = vertex_parser
                    .read_payload_for_element(&mut reader, element, &header)
                    .context("read PLY faces")?;

                for elem in &elems {
                    if let Some(Property::ListInt(ref list)) = elem.get("vertex_indices")
                        .or_else(|| elem.get("vertex_index"))
                    {
                        triangulate_face(list, &mut face_indices);
                    } else if let Some(Property::ListUInt(ref list)) = elem.get("vertex_indices")
                        .or_else(|| elem.get("vertex_index"))
                    {
                        let signed: Vec<i32> = list.iter().map(|&v| v as i32).collect();
                        triangulate_face(&signed, &mut face_indices);
                    }
                }
            }
            _ => {
                // Skip unknown elements
                let _ = vertex_parser.read_payload_for_element(&mut reader, element, &header);
            }
        }
    }

    if vertices.is_empty() {
        anyhow::bail!("PLY file contains no vertices: {}", path.display());
    }

    // If no faces, generate sequential indices (point cloud or triangle soup)
    if face_indices.is_empty() {
        face_indices = (0..vertices.len() as u32).collect();
    }

    // Compute tangents (reuse OBJ loader's approach)
    crate::scene::obj_loader::compute_tangents(&mut vertices, &face_indices);

    Ok(PlyMesh { vertices, indices: face_indices })
}

fn prop_f32(elem: &DefaultElement, name: &str) -> Option<f32> {
    match elem.get(name)? {
        Property::Float(v) => Some(*v),
        Property::Double(v) => Some(*v as f32),
        Property::Int(v) => Some(*v as f32),
        Property::UInt(v) => Some(*v as f32),
        Property::Short(v) => Some(*v as f32),
        Property::UShort(v) => Some(*v as f32),
        Property::Char(v) => Some(*v as f32),
        Property::UChar(v) => Some(*v as f32),
        _ => None,
    }
}

/// Fan-triangulate a polygon face into the index buffer.
fn triangulate_face(indices: &[i32], out: &mut Vec<u32>) {
    if indices.len() < 3 {
        return;
    }
    let v0 = indices[0] as u32;
    for i in 1..indices.len() - 1 {
        out.push(v0);
        out.push(indices[i] as u32);
        out.push(indices[i + 1] as u32);
    }
}
