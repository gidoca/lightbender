use std::path::Path;

use anyhow::{Context, Result};

use crate::types::GpuVertex;

pub struct ObjMesh {
    pub vertices:       Vec<GpuVertex>,
    pub indices:        Vec<u32>,
    pub material_index: Option<usize>,
}

pub fn load_obj(path: &Path) -> Result<Vec<ObjMesh>> {
    let (models, _materials) = tobj::load_obj(
        path,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )
    .with_context(|| format!("load OBJ: {}", path.display()))?;

    let mut meshes = Vec::new();

    for model in &models {
        let mesh = &model.mesh;
        let num_verts = mesh.positions.len() / 3;
        let has_normals = !mesh.normals.is_empty();
        let has_uvs = !mesh.texcoords.is_empty();

        let mut vertices = Vec::with_capacity(num_verts);

        for i in 0..num_verts {
            let px = mesh.positions[i * 3];
            let py = mesh.positions[i * 3 + 1];
            let pz = mesh.positions[i * 3 + 2];

            let (nx, ny, nz) = if has_normals {
                (mesh.normals[i * 3], mesh.normals[i * 3 + 1], mesh.normals[i * 3 + 2])
            } else {
                (0.0, 1.0, 0.0)
            };

            let (u, v) = if has_uvs {
                (mesh.texcoords[i * 2], mesh.texcoords[i * 2 + 1])
            } else {
                (0.0, 0.0)
            };

            vertices.push(GpuVertex {
                position: [px, py, pz],
                normal:   [nx, ny, nz],
                uv:       [u, v],
                tangent:  [1.0, 0.0, 0.0, 1.0], // placeholder, computed below
            });
        }

        let indices: Vec<u32> = mesh.indices.clone();

        // Compute tangents from UV derivatives
        compute_tangents(&mut vertices, &indices);

        meshes.push(ObjMesh {
            vertices,
            indices,
            material_index: mesh.material_id,
        });
    }

    if meshes.is_empty() {
        anyhow::bail!("OBJ file contains no meshes: {}", path.display());
    }

    Ok(meshes)
}

/// Compute tangent vectors from position/UV derivatives using MikkTSpace-style averaging.
pub fn compute_tangents(vertices: &mut [GpuVertex], indices: &[u32]) {
    // Accumulate tangents per vertex from triangle contributions
    let mut tangents = vec![[0.0f32; 3]; vertices.len()];
    let mut bitangents = vec![[0.0f32; 3]; vertices.len()];

    for tri in indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        let p0 = vertices[i0].position;
        let p1 = vertices[i1].position;
        let p2 = vertices[i2].position;

        let uv0 = vertices[i0].uv;
        let uv1 = vertices[i1].uv;
        let uv2 = vertices[i2].uv;

        let dp1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let dp2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

        let duv1 = [uv1[0] - uv0[0], uv1[1] - uv0[1]];
        let duv2 = [uv2[0] - uv0[0], uv2[1] - uv0[1]];

        let denom = duv1[0] * duv2[1] - duv1[1] * duv2[0];
        if denom.abs() < 1e-12 {
            continue;
        }
        let r = 1.0 / denom;

        let t = [
            r * (duv2[1] * dp1[0] - duv1[1] * dp2[0]),
            r * (duv2[1] * dp1[1] - duv1[1] * dp2[1]),
            r * (duv2[1] * dp1[2] - duv1[1] * dp2[2]),
        ];
        let b = [
            r * (-duv2[0] * dp1[0] + duv1[0] * dp2[0]),
            r * (-duv2[0] * dp1[1] + duv1[0] * dp2[1]),
            r * (-duv2[0] * dp1[2] + duv1[0] * dp2[2]),
        ];

        for &idx in &[i0, i1, i2] {
            for j in 0..3 {
                tangents[idx][j] += t[j];
                bitangents[idx][j] += b[j];
            }
        }
    }

    // Orthogonalize and store
    for (i, vert) in vertices.iter_mut().enumerate() {
        let n = glam::Vec3::from(vert.normal);
        let t = glam::Vec3::from(tangents[i]);
        let b = glam::Vec3::from(bitangents[i]);

        // Gram-Schmidt orthogonalization
        let t_ortho = (t - n * n.dot(t)).normalize_or_zero();
        if t_ortho.length_squared() < 1e-6 {
            vert.tangent = [1.0, 0.0, 0.0, 1.0];
            continue;
        }

        // Handedness
        let w = if n.cross(t_ortho).dot(b) < 0.0 { -1.0 } else { 1.0 };
        vert.tangent = [t_ortho.x, t_ortho.y, t_ortho.z, w];
    }
}
