mod app;
mod camera;
mod input;
mod renderer;
mod scene;
mod types;
mod vulkan;

use app::App;
use winit::event_loop::EventLoop;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let model_path = std::env::args().nth(1).map(std::path::PathBuf::from);

    let event_loop = EventLoop::new()?;
    let mut app = App::new(model_path);
    event_loop.run_app(&mut app)?;
    Ok(())
}
