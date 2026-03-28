/// Transient per-frame input state. Call `flush()` at the start of each frame.
#[derive(Default)]
pub struct InputState {
    pub left_button:   bool,
    pub right_button:  bool,
    pub middle_button: bool,
    /// Raw mouse delta accumulated since last flush (from DeviceEvent).
    pub mouse_delta: (f64, f64),
    /// Scroll delta accumulated since last flush.
    pub scroll_delta: f32,
}

impl InputState {
    /// Reset per-frame deltas (keep button states).
    pub fn flush(&mut self) {
        self.mouse_delta = (0.0, 0.0);
        self.scroll_delta = 0.0;
    }

    pub fn accumulate_mouse_delta(&mut self, dx: f64, dy: f64) {
        self.mouse_delta.0 += dx;
        self.mouse_delta.1 += dy;
    }

    pub fn accumulate_scroll(&mut self, delta: f32) {
        self.scroll_delta += delta;
    }
}
