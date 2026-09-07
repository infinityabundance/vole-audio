//! Mix accumulation (frozen).
//!
//! For an observation window of `frames` × `channels`, the mixer keeps one
//! i64 accumulator per (frame, channel). Voice contributions are added in any
//! order (i64 addition is associative — reduction trees may differ across
//! scalar/SIMD/GPU). Saturation to i32 happens **exactly once**, at the output
//! boundary, per (frame, channel).

/// Window mixer: `frames * channels` i64 accumulators.
#[derive(Debug, Clone)]
pub struct Mixer {
    channels: usize,
    frames: usize,
    acc: Vec<i64>,
}

impl Mixer {
    pub fn new(channels: usize, frames: usize) -> Mixer {
        assert!((1..=crate::limits::MAX_CHANNELS as usize).contains(&channels));
        assert!(frames <= crate::limits::MAX_QUANTUM_FRAMES as usize);
        Mixer {
            channels,
            frames,
            acc: vec![0i64; channels * frames],
        }
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Add a voice contribution to `(frame, channel)`. Contributions must
    /// already be voice-bus saturated (|c| <= 2^31-1) by `gain::contribution`.
    #[inline]
    pub fn add(&mut self, frame: usize, channel: u8, contribution: i32) {
        debug_assert!(frame < self.frames && (channel as usize) < self.channels);
        self.acc[frame * self.channels + channel as usize] += i64::from(contribution);
    }

    /// Read the raw accumulator for `(frame, channel)` (tests/evidence).
    pub fn get(&self, frame: usize, channel: u8) -> i64 {
        self.acc[frame * self.channels + channel as usize]
    }

    /// Finalize into canonical interleaved output: saturate each (frame,
    /// channel) exactly once.
    pub fn finalize_interleaved(&self, out: &mut [i32]) {
        assert_eq!(out.len(), self.frames * self.channels);
        for (i, v) in self.acc.iter().enumerate() {
            out[i] = crate::universe::arithmetic::sat_i32(*v);
        }
    }

    /// Finalize into per-channel planes.
    pub fn finalize_planes(&self, planes: &mut [&mut [i32]]) {
        assert_eq!(planes.len(), self.channels);
        for p in planes.iter() {
            assert_eq!(p.len(), self.frames);
        }
        for f in 0..self.frames {
            for (c, p) in planes.iter_mut().enumerate() {
                p[f] = crate::universe::arithmetic::sat_i32(self.acc[f * self.channels + c]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_independent_and_saturates_once() {
        let mut m = Mixer::new(2, 2);
        m.add(0, 0, i32::MAX - 1);
        m.add(0, 0, i32::MAX - 1);
        m.add(0, 1, 5);
        m.add(1, 0, -5);
        let mut out = [0i32; 4];
        m.finalize_interleaved(&mut out);
        assert_eq!(out, [i32::MAX, 5, -5, 0]);
    }

    #[test]
    fn planes_finalize() {
        let mut m = Mixer::new(2, 2);
        m.add(0, 0, 1);
        m.add(1, 0, 2);
        m.add(0, 1, 3);
        m.add(1, 1, 4);
        let mut l = [0i32; 2];
        let mut r = [0i32; 2];
        m.finalize_planes(&mut [&mut l, &mut r]);
        assert_eq!(l, [1, 2]);
        assert_eq!(r, [3, 4]);
    }
}
