use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use ash::vk;

pub struct ShaderPair {
    pub vert: vk::ShaderModule,
    pub frag: vk::ShaderModule,
}

pub struct ShaderLibrary {
    pub pairs: HashMap<String, ShaderPair>,
}

impl ShaderLibrary {
    pub fn new() -> Self {
        Self { pairs: HashMap::new() }
    }

    /// Load a named shader pair from two SPIR-V file paths.
    pub unsafe fn load(
        &mut self,
        device: &ash::Device,
        name: &str,
        vert_path: &Path,
        frag_path: &Path,
    ) -> Result<()> {
        let vert_spv = load_spirv(vert_path)
            .with_context(|| format!("load vert shader: {}", vert_path.display()))?;
        let frag_spv = load_spirv(frag_path)
            .with_context(|| format!("load frag shader: {}", frag_path.display()))?;

        let vert = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vert_spv), None)
            .context("create vert shader module")?;
        let frag = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&frag_spv), None)
            .context("create frag shader module")?;

        // Destroy old modules if replacing
        if let Some(old) = self.pairs.remove(name) {
            device.destroy_shader_module(old.vert, None);
            device.destroy_shader_module(old.frag, None);
        }

        self.pairs.insert(name.to_string(), ShaderPair { vert, frag });
        Ok(())
    }

    pub unsafe fn destroy(&self, device: &ash::Device) {
        for pair in self.pairs.values() {
            device.destroy_shader_module(pair.vert, None);
            device.destroy_shader_module(pair.frag, None);
        }
    }
}

pub fn load_spirv(path: &Path) -> Result<Vec<u32>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("read SPIR-V: {}", path.display()))?;
    anyhow::ensure!(bytes.len() % 4 == 0, "SPIR-V not 4-byte aligned");
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    anyhow::ensure!(magic == 0x07230203, "invalid SPIR-V magic in {}", path.display());
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}
