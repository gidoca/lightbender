
mod app;
mod camera;
mod input;
mod renderer;
mod scene;
mod shader;
mod types;
mod vulkan;

use std::path::PathBuf;

use app::{load_scene_from_input, App, InputPath};
use renderer::Renderer;
use winit::event_loop::EventLoop;

fn print_help() {
    println!(
        "Usage: lightbender [OPTIONS] [FILE]\n\
         \n\
         Arguments:\n  \
           [FILE]  Path to a .json scene file or a glTF model file\n\
         \n\
         Options:\n  \
           -o, --output <PATH>  Save rendered frame to an image file and exit\n  \
           -h, --help           Print this help message"
    );
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut input_path: Option<InputPath> = None;
    let mut output_path: Option<PathBuf> = None;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            "-o" | "--output" => {
                i += 1;
                if i >= args.len() {
                    anyhow::bail!("--output requires a file path argument");
                }
                output_path = Some(PathBuf::from(&args[i]));
            }
            arg => {
                let path = PathBuf::from(arg);
                input_path = Some(match path.extension().and_then(|e| e.to_str()) {
                    Some("json") => InputPath::Scene(path),
                    Some("xml")  => InputPath::MitsubaScene(path),
                    _            => InputPath::Model(path),
                });
            }
        }
        i += 1;
    }

    if let Some(output_path) = output_path {
        // Headless path: render to file without creating a window
        let mut renderer = Renderer::new_offscreen(1280, 720)?;
        let (scene, camera) = load_scene_from_input(&mut renderer, input_path.as_ref())?;
        renderer.draw_frame_offscreen(&camera.camera, scene.as_ref())?;
        renderer.capture_frame_to_file(&output_path)?;
        log::info!("Saved output to {}", output_path.display());
        if let Some(scene) = scene {
            unsafe {
                let _ = renderer.device_wait_idle();
                scene.destroy(renderer.device());
            }
        }
        return Ok(());
    }

    let event_loop = EventLoop::new()?;
    let mut app = App::new(input_path, None);
    event_loop.run_app(&mut app)?;
    Ok(())
}
