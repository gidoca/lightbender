use std::path::Path;

use anyhow::{Context, Result};
use glam::{Mat4, Vec3};
use quick_xml::events::Event;
use quick_xml::Reader;

use crate::camera::OrbitalCamera;
use crate::renderer::Renderer;
use crate::types::GpuLight;

use super::Scene;

pub struct LoadedMitsubaScene {
    pub scene:  Scene,
    pub camera: OrbitalCamera,
}

pub fn load_mitsuba(_renderer: &Renderer, path: &Path) -> Result<LoadedMitsubaScene> {
    let xml = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("read Mitsuba XML: {}: {e}", path.display()))?;

    let reader = Reader::from_str(&xml);
    let _ = reader;

    log::info!("Found Mitsuba scene: {}", path.display());

    anyhow::bail!("Mitsuba scene loading not yet implemented")
}

// ── Sensor (camera) parsing ──────────────────────────────────────────────────

struct MitsubaCameraDesc {
    fov_y:    f32, // degrees
    near:     f32,
    far:      f32,
    to_world: Mat4,
}

impl Default for MitsubaCameraDesc {
    fn default() -> Self {
        Self {
            fov_y:    45.0,
            near:     0.01,
            far:      1000.0,
            to_world: Mat4::IDENTITY,
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
                if tag.as_ref() == b"float" {
                    if let (Some(name), Some(val)) = (attr_str(e, "name"), attr_f32(e, "value")) {
                        match name.as_str() {
                            "fov" => desc.fov_y = val,
                            "near_clip" => desc.near = val,
                            "far_clip" => desc.far = val,
                            _ => {}
                        }
                    }
                }
            }
            Event::Start(ref e) => {
                let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                match tag.as_str() {
                    "transform" => {
                        desc.to_world = parse_transform(reader)?;
                    }
                    _ => {
                        // Skip film, sampler, etc.
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

/// Convert Mitsuba camera description to an OrbitalCamera.
fn camera_from_mitsuba(desc: &MitsubaCameraDesc) -> OrbitalCamera {
    // Extract camera position from to_world matrix (column 3)
    let origin = desc.to_world.col(3).truncate();
    // Forward direction is -Z in camera space
    let forward = -desc.to_world.col(2).truncate().normalize();
    // Place the look-at target 1 unit ahead in the forward direction,
    // then use the distance from origin to target for the orbital distance.
    let distance = 5.0; // default orbital distance
    let target = origin + forward * distance;

    // Compute yaw and pitch from the offset vector (origin - target)
    let offset = origin - target;
    let dist = offset.length().max(0.001);
    let pitch = (offset.y / dist).asin();
    let yaw = offset.z.atan2(offset.x) - std::f32::consts::FRAC_PI_2;

    let mut cam = OrbitalCamera::new(target, dist, yaw.to_degrees(), pitch.to_degrees());
    cam.camera.fov_y = desc.fov_y.to_radians();
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
            let world_dir = to_world.transform_vector3(direction).normalize();
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
            if let Some(v) = attr_str(e, "value") {
                if let Ok(c) = parse_rgb_value(&v) {
                    props.colors.insert(name, c);
                }
            }
        }
        "point" => {
            if let (Some(x), Some(y), Some(z)) = (attr_f32(e, "x"), attr_f32(e, "y"), attr_f32(e, "z")) {
                props.points.insert(name, Vec3::new(x, y, z));
            } else if let Some(v) = attr_str(e, "value") {
                if let Ok(p) = parse_vec3(&v) {
                    props.points.insert(name, p);
                }
            }
        }
        "vector" => {
            if let (Some(x), Some(y), Some(z)) = (attr_f32(e, "x"), attr_f32(e, "y"), attr_f32(e, "z")) {
                props.vectors.insert(name, Vec3::new(x, y, z));
            } else if let Some(v) = attr_str(e, "value") {
                if let Ok(p) = parse_vec3(&v) {
                    props.vectors.insert(name, p);
                }
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

    // Mitsuba's to_world lookat gives a camera-to-world matrix.
    // glam's look_at_rh gives world-to-camera, so we invert.
    let view = Mat4::look_at_rh(origin, target, up);
    Ok(view.inverse())
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec4;

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
        // Camera at (0,0,5) looking at origin. The to_world matrix should put
        // origin at (0,0,5) and forward along -Z (toward origin).
        let origin = m * Vec4::new(0.0, 0.0, 0.0, 1.0);
        assert!((origin.x - 0.0).abs() < 1e-5);
        assert!((origin.y - 0.0).abs() < 1e-5);
        assert!((origin.z - 5.0).abs() < 1e-5);
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

        assert!((desc.fov_y - 60.0).abs() < 1e-6);
        assert!((desc.near - 0.1).abs() < 1e-6);
        assert!((desc.far - 500.0).abs() < 1e-6);

        let cam = camera_from_mitsuba(&desc);
        // Camera should be roughly at (0, 5, 10)
        let pos = cam.camera.position;
        assert!((pos.x).abs() < 0.5, "x={}", pos.x);
        assert!((pos.y - 5.0).abs() < 0.5, "y={}", pos.y);
        assert!((pos.z - 10.0).abs() < 0.5, "z={}", pos.z);
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
}
