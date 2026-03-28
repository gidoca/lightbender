use std::path::Path;

fn main() {
    let shader_src = "shaders/src";
    let shader_out = "shaders/compiled";
    std::fs::create_dir_all(shader_out).unwrap();

    // Watch the directory itself so new files trigger a rebuild
    println!("cargo:rerun-if-changed={shader_src}");

    let dir = match std::fs::read_dir(shader_src) {
        Ok(d) => d,
        Err(_) => return, // no shaders yet
    };

    for entry in dir {
        let path = entry.unwrap().path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext == "vert" || ext == "frag" {
            let filename = path.file_name().unwrap().to_str().unwrap();
            let out_path = format!("{}/{}.spv", shader_out, filename);
            println!("cargo:rerun-if-changed={}", path.display());
            // Only recompile if source is newer than output
            if is_newer(&path, Path::new(&out_path)) {
                let status = std::process::Command::new("glslc")
                    .args([
                        path.to_str().unwrap(),
                        "-o",
                        &out_path,
                        "--target-env=vulkan1.0",
                    ])
                    .status()
                    .expect("failed to run glslc — is the Vulkan SDK installed?");
                assert!(status.success(), "shader compilation failed: {filename}");
            }
        }
    }
}

fn is_newer(src: &Path, dst: &Path) -> bool {
    let dst_meta = match std::fs::metadata(dst) {
        Ok(m) => m,
        Err(_) => return true, // dst doesn't exist
    };
    let src_meta = std::fs::metadata(src).unwrap();
    src_meta.modified().unwrap() > dst_meta.modified().unwrap()
}
