# Lightbender

A Vulkan-based 3D renderer written in Rust. Supports PBR materials, glTF models, environment mapping, and both interactive and headless rendering.

## Dependencies

- **Rust** (2024 edition) -- install via [rustup](https://rustup.rs/)
- **Vulkan driver** -- a GPU with Vulkan 1.0 support and an installed driver
- **glslc** (GLSL to SPIR-V compiler) -- part of the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) or available standalone via the `shaderc` package. Needed to compile shaders at build time.
- **Vulkan SDK** (optional, recommended) -- provides the `VK_LAYER_KHRONOS_validation` layer used for debugging and integration tests. Download from https://vulkan.lunarg.com/sdk/home and source `setup-env.sh` to make it available:
  ```
  source ~/Downloads/VulkanSDK/<version>/setup-env.sh
  ```

## Building

```
cargo build
```

The build script automatically compiles GLSL shaders in `shaders/src/` to SPIR-V using `glslc`.

## Usage

### Interactive mode

Open an interactive window with orbital camera controls (left-click rotate, right-click pan, scroll zoom):

```
cargo run -- scenes/helmet.json
cargo run -- path/to/model.glb
```

### Headless rendering

Render a single frame to an image file and exit, without creating a window:

```
cargo run -- scenes/helmet.json -o output.png
```

### Command-line options

```
Usage: lightbender [OPTIONS] [FILE]

Arguments:
  [FILE]  Path to a .json scene file or a glTF model file

Options:
  -o, --output <PATH>  Save rendered frame to an image file and exit
  -h, --help           Print this help message
```

## Tests

### Running integration tests

The integration tests render each scene using `--output` and compare the result to stored reference images. They also check for Vulkan validation layer errors.

```
cargo test
```

The tests automatically search for the Vulkan SDK in common locations (`~/Downloads/VulkanSDK`, `~/VulkanSDK`, `/opt/VulkanSDK`) to enable validation layers. If the SDK is not found, validation checks are skipped with a warning.

Tests are skipped gracefully if no Vulkan-capable GPU is available.

### Regenerating reference images

After intentional visual changes (shader updates, rendering fixes, etc.), regenerate the reference images:

```
LIGHTBENDER_UPDATE_REFERENCES=1 cargo test
```

Visually inspect the images in `tests/reference/` before committing.

### Adjusting the comparison threshold

The default RMSE threshold is 1.5 (on a 0-255 scale). To adjust it:

```
LIGHTBENDER_RMSE_THRESHOLD=2.0 cargo test
```
