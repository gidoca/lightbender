use std::path::Path;

use anyhow::{Context, Result};

use lightbender_scene::{compute_tangents, Vertex};

pub struct ObjMesh {
    pub vertices: Vec<Vertex>,
    pub indices:  Vec<u32>,
}

pub fn load_obj(path: &Path) -> Result<Vec<ObjMesh>> {
    let (models, _materials) = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS)
        .with_context(|| format!("load OBJ: {}", path.display()))?;

    let mut meshes = Vec::with_capacity(models.len());
    for m in &models {
        let mesh = &m.mesh;
        let vertex_count = mesh.positions.len() / 3;
        let mut vertices = Vec::with_capacity(vertex_count);

        for i in 0..vertex_count {
            let position = [
                mesh.positions[i * 3],
                mesh.positions[i * 3 + 1],
                mesh.positions[i * 3 + 2],
            ];
            let normal = if mesh.normals.len() >= (i + 1) * 3 {
                [mesh.normals[i * 3], mesh.normals[i * 3 + 1], mesh.normals[i * 3 + 2]]
            } else {
                [0.0, 1.0, 0.0]
            };
            let uv = if mesh.texcoords.len() >= (i + 1) * 2 {
                [mesh.texcoords[i * 2], mesh.texcoords[i * 2 + 1]]
            } else {
                [0.0, 0.0]
            };

            vertices.push(Vertex {
                position,
                normal,
                uv,
                tangent: [1.0, 0.0, 0.0, 1.0], // computed below
            });
        }

        let indices: Vec<u32> = mesh.indices.clone();

        // Compute tangent vectors from position/UV derivatives
        compute_tangents(&mut vertices, &indices);

        meshes.push(ObjMesh { vertices, indices });
    }

    Ok(meshes)
}
