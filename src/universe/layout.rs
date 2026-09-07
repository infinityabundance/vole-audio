//! Channel layout semantics (frozen).
//!
//! u1 canonical observation layout is **frame-interleaved**:
//! `[ch0 f0, ch1 f0, ..., chC-1 f0, ch0 f1, ...]` — the endpoint order.
//! Evaluation internally may use SoA per-channel storage, but every canonical
//! byte form (hashing, archival payloads, endpoint writes) is interleaved
//! little-endian i32.

use crate::limits::MAX_CHANNELS;
use core::fmt;

/// Channel layout identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Layout {
    Mono = 1,
    Stereo = 2,
    /// Explicit channel count (1..=MAX_CHANNELS).
    Channels(u8),
}

impl Layout {
    pub const fn count(self) -> u8 {
        match self {
            Layout::Mono => 1,
            Layout::Stereo => 2,
            Layout::Channels(n) => n,
        }
    }

    /// Validate a channel count against the universe ceiling.
    pub const fn checked(n: u8) -> Option<Layout> {
        if n == 0 || n > MAX_CHANNELS as u8 {
            None
        } else {
            Some(match n {
                1 => Layout::Mono,
                2 => Layout::Stereo,
                n => Layout::Channels(n),
            })
        }
    }

    /// Frames needed to hold `frames` interleaved observations.
    pub const fn interleaved_len(self, frames: usize) -> Option<usize> {
        frames.checked_mul(self.count() as usize)
    }

    /// Interleave per-channel planes into the canonical frame-interleaved
    /// order. Each plane must have exactly `frames` samples.
    pub fn interleave(&self, planes: &[&[i32]], out: &mut [i32]) {
        let ch = self.count() as usize;
        assert_eq!(planes.len(), ch, "plane count mismatch");
        let frames = out.len() / ch;
        assert_eq!(out.len(), frames * ch, "output length mismatch");
        for p in planes {
            assert_eq!(p.len(), frames, "plane length mismatch");
        }
        for f in 0..frames {
            for (c, p) in planes.iter().enumerate() {
                out[f * ch + c] = p[f];
            }
        }
    }

    /// De-interleave canonical frames into per-channel planes.
    pub fn deinterleave(&self, frames: &[i32], planes: &mut [&mut [i32]]) {
        let ch = self.count() as usize;
        assert_eq!(planes.len(), ch, "plane count mismatch");
        let n = frames.len() / ch;
        assert_eq!(frames.len(), n * ch, "input length mismatch");
        for p in planes.iter() {
            assert_eq!(p.len(), n, "plane length mismatch");
        }
        for f in 0..n {
            for (c, p) in planes.iter_mut().enumerate() {
                p[f] = frames[f * ch + c];
            }
        }
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Layout::Mono => write!(f, "mono"),
            Layout::Stereo => write!(f, "stereo"),
            Layout::Channels(n) => write!(f, "{}ch", n),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_layout_validation() {
        assert_eq!(Layout::checked(0), None);
        assert_eq!(Layout::checked(1), Some(Layout::Mono));
        assert_eq!(Layout::checked(2), Some(Layout::Stereo));
        assert_eq!(Layout::checked(3), Some(Layout::Channels(3)));
        assert_eq!(
            Layout::checked(MAX_CHANNELS as u8),
            Some(Layout::Channels(32))
        );
        assert_eq!(Layout::checked(33), None);
    }

    #[test]
    fn interleave_roundtrip() {
        let l = Layout::Stereo;
        let lf = vec![1i32, 3, 5, 7];
        let rf = vec![2i32, 4, 6, 8];
        let mut inter = vec![0i32; 8];
        l.interleave(&[&lf, &rf], &mut inter);
        assert_eq!(inter, vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let mut lo = vec![0i32; 4];
        let mut ro = vec![0i32; 4];
        let mut planes: [&mut [i32]; 2] = [&mut lo, &mut ro];
        l.deinterleave(&inter, &mut planes);
        assert_eq!(lo, lf);
        assert_eq!(ro, rf);
    }

    #[test]
    fn layout_display() {
        assert_eq!(Layout::Mono.to_string(), "mono");
        assert_eq!(Layout::Stereo.to_string(), "stereo");
        assert_eq!(Layout::Channels(6).to_string(), "6ch");
    }
}
