use std::path::{Path, PathBuf};
use std::process::Command;

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_lightbender"))
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn reference_dir() -> PathBuf {
    project_root().join("tests").join("reference")
}

/// Find the Vulkan SDK setup-env.sh and return env vars to enable validation layers.
fn vulkan_sdk_env() -> Vec<(String, String)> {
    // Check if validation layer is already available
    if std::env::var("VK_LAYER_PATH").is_ok() {
        return vec![];
    }

    // Look for Vulkan SDK in common locations
    let home = std::env::var("HOME").unwrap_or_default();
    let candidates = [
        format!("{home}/Downloads/VulkanSDK"),
        format!("{home}/VulkanSDK"),
        "/opt/VulkanSDK".to_string(),
    ];

    for base in &candidates {
        let base_path = Path::new(base);
        if !base_path.exists() {
            continue;
        }
        // Find the newest version directory
        let Ok(entries) = std::fs::read_dir(base_path) else {
            continue;
        };
        let mut versions: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        versions.sort();
        if let Some(version_dir) = versions.last() {
            let layer_path = version_dir.join("share/vulkan/explicit_layer.d");
            let lib_path = version_dir.join("lib");
            if layer_path.exists() {
                return vec![
                    ("VK_LAYER_PATH".to_string(), layer_path.to_string_lossy().to_string()),
                    ("LD_LIBRARY_PATH".to_string(), format!(
                        "{}:{}",
                        lib_path.display(),
                        std::env::var("LD_LIBRARY_PATH").unwrap_or_default()
                    )),
                ];
            }
        }
    }

    eprintln!("Warning: Vulkan SDK not found — validation layer checks may not work");
    vec![]
}

struct RenderResult {
    output_path: PathBuf,
    stderr: String,
}

/// Render a scene and return the output path and stderr.
/// Returns None if Vulkan is unavailable (test should be skipped).
fn render_scene(scene: &str) -> Option<RenderResult> {
    let scene_path = project_root().join("scenes").join(format!("{scene}.json"));
    let output_dir = std::env::temp_dir().join(format!(
        "lightbender-test-{}-{}",
        std::process::id(),
        scene
    ));
    let _ = std::fs::create_dir_all(&output_dir);
    let output_path = output_dir.join(format!("{scene}.png"));

    let mut cmd = Command::new(binary_path());
    cmd.arg(scene_path.as_os_str())
        .arg("-o")
        .arg(output_path.as_os_str())
        .env("RUST_LOG", "warn");

    for (key, value) in vulkan_sdk_env() {
        cmd.env(key, value);
    }

    let output = cmd.output().expect("failed to execute lightbender binary");

    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        let stderr_lower = stderr.to_lowercase();
        if stderr_lower.contains("vulkan") || stderr_lower.contains("vk") {
            eprintln!("Skipping test: Vulkan not available\n{stderr}");
            return None;
        }
        panic!(
            "lightbender exited with {}\nstderr:\n{stderr}",
            output.status
        );
    }

    Some(RenderResult {
        output_path,
        stderr,
    })
}

fn compare_images(actual: &Path, reference: &Path) -> (f64, u8) {
    let actual_img = image::open(actual)
        .unwrap_or_else(|e| panic!("failed to open actual image {}: {e}", actual.display()))
        .to_rgba8();
    let ref_img = image::open(reference)
        .unwrap_or_else(|e| panic!("failed to open reference image {}: {e}", reference.display()))
        .to_rgba8();

    assert_eq!(
        actual_img.dimensions(),
        ref_img.dimensions(),
        "Image dimensions differ: actual {:?} vs reference {:?}",
        actual_img.dimensions(),
        ref_img.dimensions()
    );

    let mut sum_sq: f64 = 0.0;
    let mut max_diff: u8 = 0;
    let pixel_count = (actual_img.width() * actual_img.height()) as f64;

    for (a, r) in actual_img.pixels().zip(ref_img.pixels()) {
        for i in 0..4 {
            let diff = (a.0[i] as i16 - r.0[i] as i16).unsigned_abs() as u8;
            max_diff = max_diff.max(diff);
            sum_sq += (diff as f64) * (diff as f64);
        }
    }

    let rmse = (sum_sq / (pixel_count * 4.0)).sqrt();
    (rmse, max_diff)
}

fn assert_no_validation_errors(stderr: &str) {
    for line in stderr.lines() {
        if line.contains("[Vulkan]") {
            panic!("Vulkan validation message detected:\n{line}\n\nFull stderr:\n{stderr}");
        }
    }
}

fn update_references() -> bool {
    std::env::var("LIGHTBENDER_UPDATE_REFERENCES").is_ok()
}

fn rmse_threshold() -> f64 {
    std::env::var("LIGHTBENDER_RMSE_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.5)
}

fn run_scene_test(scene: &str) {
    let Some(result) = render_scene(scene) else {
        return; // Vulkan unavailable — skip
    };

    assert_no_validation_errors(&result.stderr);

    let ref_path = reference_dir().join(format!("{scene}.png"));

    if update_references() {
        std::fs::copy(&result.output_path, &ref_path).unwrap_or_else(|e| {
            panic!(
                "failed to copy {} to {}: {e}",
                result.output_path.display(),
                ref_path.display()
            )
        });
        eprintln!("Updated reference: {}", ref_path.display());
        return;
    }

    assert!(
        ref_path.exists(),
        "Reference image not found: {}\n\
         Run with LIGHTBENDER_UPDATE_REFERENCES=1 cargo test to generate it.",
        ref_path.display()
    );

    let (rmse, max_diff) = compare_images(&result.output_path, &ref_path);
    let threshold = rmse_threshold();

    assert!(
        rmse <= threshold,
        "Image differs from reference beyond threshold.\n\
         RMSE: {rmse:.4} (threshold: {threshold})\n\
         Max per-channel diff: {max_diff}\n\
         Actual: {}\n\
         Reference: {}",
        result.output_path.display(),
        ref_path.display()
    );
}

#[test]
fn test_example() {
    run_scene_test("example");
}

#[test]
fn test_helmet() {
    run_scene_test("helmet");
}

#[test]
fn test_multi() {
    run_scene_test("multi");
}

#[test]
fn test_helmet_env() {
    // Skip if the environment map asset is not available
    let scene_path = project_root().join("scenes").join("helmet_env.json");
    let scene_content = std::fs::read_to_string(&scene_path).expect("read helmet_env.json");
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&scene_content) {
        if let Some(map_path) = json
            .get("environment")
            .and_then(|e| e.get("map"))
            .and_then(|m| m.as_str())
        {
            let resolved = if Path::new(map_path).is_absolute() {
                PathBuf::from(map_path)
            } else {
                scene_path.parent().unwrap().join(map_path)
            };
            if !resolved.exists() {
                eprintln!(
                    "Skipping test_helmet_env: environment map not found at {}",
                    resolved.display()
                );
                return;
            }
        }
    }

    run_scene_test("helmet_env");
}
