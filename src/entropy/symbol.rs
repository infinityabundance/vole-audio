//! Reversible symbolizations over canonical sample codes (H.2.4 / H.2.10).
//!
//! A symbolization maps a page of canonical interleaved `i32` codes
//! (frame-major, channel-minor) to an ordered set of **symbol streams**
//! (byte streams), and back. Every symbolization is reversible, canonical,
//! bounded, versioned, and independently tested. rANS coding operates on the
//! streams; the symbolization itself is value-transparent for exact
//! reconstruction (`symbolize` -> `desymbolize` == identity on samples).
//!
//! Frozen ids (versioned; new candidates append, never renumber):
//!
//! * `1 identity` — one stream of the raw LE bytes of the samples. Used for
//!   RAW page bodies and byte-stream containers. Zero transform cost.
//! * `2 lane4-plain` — four streams: byte-position `p` (0..4) of every
//!   sample's LE representation, in canonical interleaved sample order.
//! * `3 lane4-zigzag` — four streams over the ZigZag-mapped (`i32 -> u32`)
//!   sample values.
//! * `4 delta-lane4` — per-channel modular first differences, ZigZag-mapped,
//!   four byte-position streams (channels concatenated within a lane so
//!   per-channel delta continuity is preserved).
//!
//! Lane-stream lengths are always `frames * channels` per lane (all four
//! lanes equal). The ZigZag map and the modular first-difference are exact
//! and documented in this file's tests and in `docs/RANS.md`.
//!
//! High-entropy/random controls must not be forced through elaborate
//! symbolizations: the RAW/literal fallback competes on complete bytes and
//! wins for incompressible material (H.2.5/H.2.31).

use crate::entropy::transform;
use crate::error::{Error, Kind, Result};

pub use crate::entropy::transform::{byte_of_le, modular_delta, unzigzag, zigzag};

/// Versioned symbolization ids (frozen; see module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Symbolization {
    /// Identity byte stream (raw LE sample bytes); RAW page bodies.
    Identity = 1,
    /// Four byte-position lanes over LE sample bytes.
    Lane4Plain = 2,
    /// Four byte-position lanes over ZigZag-mapped samples.
    Lane4ZigZag = 3,
    /// Per-channel modular first difference + ZigZag, four lanes.
    DeltaLane4 = 4,
}

impl Symbolization {
    pub const fn code(self) -> u8 {
        self as u8
    }

    pub const fn from_code(c: u8) -> Option<Symbolization> {
        match c {
            1 => Some(Symbolization::Identity),
            2 => Some(Symbolization::Lane4Plain),
            3 => Some(Symbolization::Lane4ZigZag),
            4 => Some(Symbolization::DeltaLane4),
            _ => None,
        }
    }

    /// Lane/stream count produced by the symbolization.
    pub const fn stream_count(self) -> usize {
        match self {
            Symbolization::Identity => 1,
            Symbolization::Lane4Plain | Symbolization::Lane4ZigZag | Symbolization::DeltaLane4 => 4,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Symbolization::Identity => "identity",
            Symbolization::Lane4Plain => "lane4_plain",
            Symbolization::Lane4ZigZag => "lane4_zigzag",
            Symbolization::DeltaLane4 => "delta_lane4",
        }
    }
}

/// Symbolize `samples` (canonical interleaved codes, `frames * channels`
/// long) into byte streams per the symbolization. Lane stream lengths are
/// documented above; `channels` is the interleave factor (1..=MAX_CHANNELS).
pub fn symbolize(sym: Symbolization, samples: &[i32], channels: usize) -> Result<Vec<Vec<u8>>> {
    if channels == 0 || channels > crate::limits::MAX_CHANNELS as usize {
        return Err(Error::new(Kind::Malformed, "channel count out of domain"));
    }
    if !samples.len().is_multiple_of(channels) {
        return Err(Error::new(
            Kind::Malformed,
            "sample count not divisible by channel count",
        ));
    }
    let frames = samples.len() / channels;
    match sym {
        Symbolization::Identity => {
            let mut bytes = Vec::with_capacity(samples.len() * 4);
            for s in samples {
                bytes.extend_from_slice(&s.to_le_bytes());
            }
            Ok(vec![bytes])
        }
        Symbolization::Lane4Plain => Ok(lane_bytes(samples, |s| *s as u32)),
        Symbolization::Lane4ZigZag => Ok(lane_bytes(samples, |s| zigzag(*s))),
        Symbolization::DeltaLane4 => {
            // Per-channel modular first difference. Within each lane, channel
            // streams are concatenated channel-major (c0 frames, then c1
            // frames, ...) so per-channel delta continuity is preserved and
            // every lane still has exactly `frames * channels` symbols.
            let mut out = vec![Vec::new(); 4];
            for c in 0..channels {
                let mut prev = 0i32;
                for f in 0..frames {
                    let s = samples[f * channels + c];
                    let d = if f == 0 { s } else { modular_delta(prev, s) };
                    prev = s;
                    let z = zigzag(d);
                    for (p, lane) in out.iter_mut().enumerate() {
                        lane.push(byte_of_le(z, p));
                    }
                }
            }
            Ok(out)
        }
    }
}

/// Desymbolize byte streams back into canonical interleaved codes.
pub fn desymbolize(sym: Symbolization, streams: &[Vec<u8>], channels: usize) -> Result<Vec<i32>> {
    let expected = sym.stream_count();
    if streams.len() != expected {
        return Err(Error::new(
            Kind::Malformed,
            format!(
                "symbolization {} needs {expected} streams, got {}",
                sym.code(),
                streams.len()
            ),
        ));
    }
    // One shared implementation (transform::samples_from_streams) is used by
    // the host, the SIMD path, and the device kernels: decoded symbols always
    // reconstruct identical samples everywhere.
    let refs: Vec<&[u8]> = streams.iter().map(|s| s.as_slice()).collect();
    let total = match sym {
        Symbolization::Identity => streams[0].len() / 4,
        Symbolization::Lane4Plain | Symbolization::Lane4ZigZag | Symbolization::DeltaLane4 => {
            streams[0].len()
        }
    };
    let mut out = vec![0i32; total];
    if !transform::samples_from_streams(sym.code(), &refs, channels, &mut out) {
        return Err(Error::new(
            Kind::Malformed,
            "stream shape invalid for symbolization",
        ));
    }
    Ok(out)
}
fn lane_bytes(samples: &[i32], map: impl Fn(&i32) -> u32) -> Vec<Vec<u8>> {
    let mut out = vec![Vec::new(); 4];
    for s in samples {
        let v = map(s);
        for (p, b) in out.iter_mut().enumerate() {
            b.push(((v >> (8 * p)) & 0xff) as u8);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_values(rng: &mut impl FnMut() -> u64, n: usize) -> Vec<i32> {
        // Deterministic LCG; produces a spread of magnitudes incl. negatives.
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let s = rng();
            let hi = (s >> 33) as u32;
            let lo = (s >> 1) as u32;
            let v = (hi ^ lo) as i32;
            // skew toward small magnitudes so delta coding has structure
            out.push(v.wrapping_mul((s >> 40) as i32 & 0xff).wrapping_div(257));
        }
        out
    }

    fn lcg(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s
        }
    }

    #[test]
    fn zigzag_roundtrip_is_bijective() {
        for v in [0i32, 1, -1, 2, -2, i32::MAX, i32::MIN, 12345, -99999] {
            assert_eq!(unzigzag(zigzag(v)), v);
        }
        // Bijection over a large deterministic sample.
        let mut r = lcg(7);
        for _ in 0..100_000 {
            let v = (r() >> 32) as i32;
            assert_eq!(unzigzag(zigzag(v)), v);
        }
    }

    #[test]
    fn modular_delta_roundtrip() {
        let mut r = lcg(11);
        let mut prev = 0i32;
        for _ in 0..100_000 {
            let cur = (r() >> 32) as i32;
            let d = modular_delta(prev, cur);
            assert_eq!(prev.wrapping_add(d), cur);
            prev = cur;
        }
    }

    #[test]
    fn identity_roundtrip() {
        let mut r = lcg(3);
        for len in [0usize, 1, 7, 1024] {
            let samples = sample_values(&mut r, len);
            let streams = symbolize(Symbolization::Identity, &samples, 1).unwrap();
            assert_eq!(streams.len(), 1);
            assert_eq!(streams[0].len(), len * 4);
            let back = desymbolize(Symbolization::Identity, &streams, 1).unwrap();
            assert_eq!(back, samples);
        }
    }

    #[test]
    fn lane4_plain_roundtrip() {
        let mut r = lcg(13);
        let samples = sample_values(&mut r, 4096);
        let streams = symbolize(Symbolization::Lane4Plain, &samples, 1).unwrap();
        assert_eq!(streams.len(), 4);
        for s in &streams {
            assert_eq!(s.len(), samples.len());
        }
        let back = desymbolize(Symbolization::Lane4Plain, &streams, 1).unwrap();
        assert_eq!(back, samples);
        // Lane 0 must be the LE low bytes.
        assert_eq!(streams[0][0], (samples[0] as u32 & 0xff) as u8);
    }

    #[test]
    fn lane4_zigzag_roundtrip() {
        let mut r = lcg(17);
        let samples = sample_values(&mut r, 512);
        let streams = symbolize(Symbolization::Lane4ZigZag, &samples, 1).unwrap();
        let back = desymbolize(Symbolization::Lane4ZigZag, &streams, 1).unwrap();
        assert_eq!(back, samples);
    }

    #[test]
    fn delta_lane4_roundtrip_stereo() {
        let mut r = lcg(23);
        let frames = 512;
        let ch = 2usize;
        let samples: Vec<i32> = (0..frames * ch)
            .map(|k| {
                let f = k / ch;
                let c = k % ch;
                // Smooth per-channel signal: correlated ramp + small noise.
                let base = (f as i32) * 1000 + if c == 0 { 50_000 } else { -50_000 };
                let jitter = (r() >> 33) as i32 % 256;
                base.wrapping_add(jitter)
            })
            .collect();
        let streams = symbolize(Symbolization::DeltaLane4, &samples, ch).unwrap();
        assert_eq!(streams.len(), 4);
        for s in &streams {
            assert_eq!(s.len(), frames * ch);
        }
        let back = desymbolize(Symbolization::DeltaLane4, &streams, ch).unwrap();
        assert_eq!(back, samples);
        // Delta coding must compress the smooth part: per-sample high byte
        // lanes carry little information.
        assert!(streams[3].iter().filter(|&&b| b != 0).count() < streams[3].len());
    }

    #[test]
    fn desymbolize_validates_stream_shape() {
        let mut r = lcg(29);
        let samples = sample_values(&mut r, 64);
        let streams = symbolize(Symbolization::Lane4Plain, &samples, 1).unwrap();
        // Wrong stream count.
        assert!(desymbolize(Symbolization::Lane4Plain, &streams[..3], 1).is_err());
        // Non-4-aligned identity stream.
        let bad = vec![vec![1u8, 2, 3]];
        assert!(desymbolize(Symbolization::Identity, &bad, 1).is_err());
        // Channel split mismatch.
        assert!(desymbolize(Symbolization::DeltaLane4, &streams, 3).is_err());
    }

    #[test]
    fn symbolization_ids_are_stable() {
        for c in 1..=4u8 {
            assert_eq!(Symbolization::from_code(c).unwrap().code(), c);
        }
        assert!(Symbolization::from_code(0).is_none());
        assert!(Symbolization::from_code(5).is_none());
        assert_eq!(Symbolization::Identity.name(), "identity");
        assert_eq!(Symbolization::DeltaLane4.name(), "delta_lane4");
    }
}
