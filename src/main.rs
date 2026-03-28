mod app;
mod camera;
mod input;
mod renderer;
mod scene;
mod shader;
mod types;
mod vulkan;

use app::{App, InputPath};
use winit::event_loop::EventLoop;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let input_path = std::env::args().nth(1).map(|arg| {
        let path = std::path::PathBuf::from(&arg);
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            InputPath::Scene(path)
        } else {
            InputPath::Model(path)
        }
    });

    let event_loop = EventLoop::new()?;
    let mut app = App::new(input_path);
    event_loop.run_app(&mut app)?;
    Ok(())
}
