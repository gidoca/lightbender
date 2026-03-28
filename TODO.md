* Code quality
    * Enable clippy linting
    * Add unit tests where it makes sense
    * Add all resources needed for tests to the repository
    * Add Github Actions
    * Update to Rust 2024 edition
    * Speed optimizations
    * Consider switching to Vulkano
* Split up into libraries:
    * Scene graph (data structures only, provides mesh object, instancing, hierarchical groups, polylines, materials)
    * Renderer (consumes scene graph, provides transformation updates, mesh updates, camera updates, texture updates, scene object creation and deletion, render modes)
    * Scene loaders (Support Mitsuba format, our own JSON based format, in the future our own binary format)
    * Object loaders (Support glTF, OBJ, PLY, PNG, JPG, EXR, HDR)
    * Interaction library (Camera, mouse picking, trackball for object transformation)
* New libraries:
    * UI toolkit (must support both 2D and 3D UI)
* Applications:
    * Simple renderer (same as current application)
    * A playground for mesh operations and algorithms (somewhat akin to Meshlab, named Rustbucket)
    * In the future a simple game engine
* Renderer improvements
    * Better tonemapping
    * Shadow mapping
    * API agnostic, so we can support WebGPU and DX12
    * Support Vulkan ray tracing
* Stuff for the far future
    * Subsurface scattering
    * Global illumination
    * OpenXR support
