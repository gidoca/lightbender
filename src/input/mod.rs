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
    pub key_w: bool,
    pub key_a: bool,
    pub key_s: bool,
    pub key_d: bool,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flush_resets_deltas_keeps_buttons() {
        let mut input = InputState {
            left_button: true,
            right_button: false,
            middle_button: true,
            mouse_delta: (3.0, -2.0),
            scroll_delta: 1.5,
            key_w: true,
            key_a: false,
            key_s: true,
            key_d: false,
        };
        input.flush();
        assert_eq!(input.mouse_delta, (0.0, 0.0));
        assert_eq!(input.scroll_delta, 0.0);
        assert!(input.left_button);
        assert!(input.middle_button);
        assert!(input.key_w);
        assert!(input.key_s);
        assert!(!input.key_a);
        assert!(!input.key_d);
    }

    #[test]
    fn accumulate_mouse_delta_sums() {
        let mut input = InputState::default();
        input.accumulate_mouse_delta(1.0, 2.0);
        input.accumulate_mouse_delta(-0.5, 3.0);
        assert_eq!(input.mouse_delta, (0.5, 5.0));
    }

    #[test]
    fn accumulate_scroll_sums() {
        let mut input = InputState::default();
        input.accumulate_scroll(1.0);
        input.accumulate_scroll(-0.5);
        assert!((input.scroll_delta - 0.5).abs() < 1e-6);
    }
}
