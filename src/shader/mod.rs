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

        let vert = unsafe {
            device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vert_spv), None)
                .context("create vert shader module")?
        };
        let frag = unsafe {
            device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&frag_spv), None)
                .context("create frag shader module")?
        };

        // Destroy old modules if replacing
        if let Some(old) = self.pairs.remove(name) {
            unsafe {
                device.destroy_shader_module(old.vert, None);
                device.destroy_shader_module(old.frag, None);
            }
        }

        self.pairs.insert(name.to_string(), ShaderPair { vert, frag });
        Ok(())
    }

    pub unsafe fn destroy(&self, device: &ash::Device) {
        for pair in self.pairs.values() {
            unsafe {
                device.destroy_shader_module(pair.vert, None);
                device.destroy_shader_module(pair.frag, None);
            }
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

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn write_temp(bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f
    }

    fn spirv_magic_le() -> [u8; 4] {
        0x07230203u32.to_le_bytes()
    }

    #[test]
    fn load_spirv_valid() {
        // Minimal valid SPIR-V: magic + 7 more u32s (version, generator, bound, 0, 0, opcode...)
        let mut bytes = spirv_magic_le().to_vec();
        bytes.extend_from_slice(&[0u8; 28]); // 7 more u32s
        let f = write_temp(&bytes);
        let words = load_spirv(f.path()).unwrap();
        assert_eq!(words[0], 0x07230203);
        assert_eq!(words.len(), 8);
    }

    #[test]
    fn load_spirv_wrong_magic() {
        let bytes = [0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00, 0x00];
        let f = write_temp(&bytes);
        let err = load_spirv(f.path()).unwrap_err();
        assert!(err.to_string().contains("magic"), "{}", err);
    }

    #[test]
    fn load_spirv_misaligned() {
        let mut bytes = spirv_magic_le().to_vec();
        bytes.push(0x00); // 5 bytes total — not 4-byte aligned
        let f = write_temp(&bytes);
        let err = load_spirv(f.path()).unwrap_err();
        assert!(err.to_string().contains("4-byte"), "{}", err);
    }

    #[test]
    fn load_spirv_missing_file() {
        let err = load_spirv(Path::new("/nonexistent/path/shader.spv")).unwrap_err();
        assert!(err.to_string().contains("read SPIR-V"), "{}", err);
    }
}
