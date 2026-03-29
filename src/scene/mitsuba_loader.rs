use std::path::Path;

use anyhow::Result;

use crate::camera::OrbitalCamera;
use crate::renderer::Renderer;

use super::Scene;

pub struct LoadedMitsubaScene {
    pub scene:  Scene,
    pub camera: OrbitalCamera,
}

pub fn load_mitsuba(_renderer: &Renderer, path: &Path) -> Result<LoadedMitsubaScene> {
    let xml = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("read Mitsuba XML: {}: {e}", path.display()))?;

    // Verify this is a Mitsuba scene file
    let reader = quick_xml::Reader::from_str(&xml);
    let _ = reader; // will be used in subsequent steps

    log::info!("Found Mitsuba scene: {}", path.display());

    anyhow::bail!("Mitsuba scene loading not yet implemented")
}
