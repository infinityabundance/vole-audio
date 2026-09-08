//! No_std sample-domain transforms (Phase H.2) — pure, slice-based, shared
//! by the host symbolization layer and the device entropy decoder.
//!
//! The transforms here are value-transparent: they map between canonical
//! interleaved `i32` codes and per-lane byte streams exactly as documented
//! in `symbol.rs` (which wraps these primitives for the `Vec`-based host
//! API). Device kernels and SIMD paths use exactly this code, so decoded
//! symbols always reconstruct identical samples everywhere.
//!
//! Symbolization stream conventions (frozen):
//!
//! * `Identity` (1): one stream of raw LE sample bytes.
//! * `Lane4Plain` (2) / `Lane4ZigZag` (3): four byte-position lanes over the
//!   sample values (zigzag-mapped for 3). Lane length = frames * channels.
//! * `DeltaLane4` (4): per-channel modular first differences (zigzag-coded);
//!   within each lane, channel streams concatenate channel-major.
//!
//! All functions are `no_std`-clean, infallible-by-contract, and return
//! `bool`/write into caller-provided buffers.

use crate::limits::MAX_CHANNELS;

/// ZigZag-mapped sample: `i32` -> `u32` bijectively (`0, -1, 1, -2, 2, ...`).
#[inline]
pub fn zigzag(v: i32) -> u32 {
    ((v << 1) ^ (v >> 31)) as u32
}

/// Inverse ZigZag: `u32` -> `i32`.
#[inline]
pub fn unzigzag(z: u32) -> i32 {
    ((z >> 1) as i32) ^ -((z & 1) as i32)
}

/// Modular first difference in the code domain (exact under `mod 2^32`);
/// `s_cur == prev.wrapping_add(delta)` always reconstructs `s_cur`.
#[inline]
pub fn modular_delta(prev: i32, cur: i32) -> i32 {
    cur.wrapping_sub(prev)
}

/// Extract one byte lane from a u32 LE value at byte position `p`.
#[inline]
pub fn byte_of_le(v: u32, p: usize) -> u8 {
    ((v >> (8 * p)) & 0xff) as u8
}

/// Reconstruct interleaved samples from decoded symbol streams.
///
/// `streams` holds the per-stream symbol bytes (length `frames * channels`
/// per lane stream; `1` stream for Identity with `frames*channels*4` bytes).
/// Writes exactly `frames * channels` interleaved codes into `out`.
/// Returns `false` on shape mismatch (hostile input).
pub fn samples_from_streams(sym: u8, streams: &[&[u8]], channels: usize, out: &mut [i32]) -> bool {
    let mut lanes = [&[][..]; 4];
    for (i, slot) in lanes.iter_mut().enumerate() {
        if let Some(s) = streams.get(i) {
            *slot = s;
        }
    }
    samples_from_lanes(sym, lanes, channels, out)
}

/// No-alloc reconstruction core (device kernels): up to four lane streams.
/// See [`samples_from_streams`] for the stream conventions.
pub fn samples_from_lanes(sym: u8, lanes: [&[u8]; 4], channels: usize, out: &mut [i32]) -> bool {
    if channels == 0 || channels > MAX_CHANNELS as usize {
        return false;
    }
    match sym {
        1 => {
            // Identity: raw LE sample bytes.
            let b = lanes[0];
            if !b.len().is_multiple_of(4) {
                return false;
            }
            let n = b.len() / 4;
            if out.len() != n {
                return false;
            }
            for (i, s) in out.iter_mut().enumerate() {
                *s = i32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]);
            }
            true
        }
        2 | 3 => {
            // Four byte-position lanes (plain or zigzag).
            let n = lanes[0].len();
            if lanes[1].len() != n || lanes[2].len() != n || lanes[3].len() != n {
                return false;
            }
            if out.len() != n {
                return false;
            }
            if sym == 2 {
                for (i, s) in out.iter_mut().enumerate() {
                    *s = i32::from_le_bytes([lanes[0][i], lanes[1][i], lanes[2][i], lanes[3][i]]);
                }
            } else {
                for (i, s) in out.iter_mut().enumerate() {
                    let z =
                        u32::from_le_bytes([lanes[0][i], lanes[1][i], lanes[2][i], lanes[3][i]]);
                    *s = unzigzag(z);
                }
            }
            true
        }
        4 => {
            // Per-channel modular first difference. Streams are four lanes;
            // within a lane, channel streams concatenate channel-major.
            let total = lanes[0].len();
            if lanes[1].len() != total || lanes[2].len() != total || lanes[3].len() != total {
                return false;
            }
            if !total.is_multiple_of(channels) {
                return false;
            }
            if out.len() != total {
                return false;
            }
            let frames = total / channels;
            for c in 0..channels {
                let mut prev = 0i32;
                for f in 0..frames {
                    let k = c * frames + f;
                    let z =
                        u32::from_le_bytes([lanes[0][k], lanes[1][k], lanes[2][k], lanes[3][k]]);
                    let d = unzigzag(z);
                    let s = if f == 0 { d } else { prev.wrapping_add(d) };
                    out[f * channels + c] = s;
                    prev = s;
                }
            }
            true
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitive_maps_roundtrip() {
        for v in [0i32, -1, 1, i32::MAX, i32::MIN] {
            assert_eq!(unzigzag(zigzag(v)), v);
        }
        assert_eq!(modular_delta(100, -5), -105);
        assert_eq!(100i32.wrapping_add(modular_delta(100, -5)), -5);
    }

    #[test]
    fn stream_reconstruction_matches_host_symbolization() {
        // Cross-check against the Vec-based host symbolizer: build streams
        // with `symbolize`, then reconstruct with the slice primitive.
        let samples: Vec<i32> = (0..200u32)
            .map(|i| (i.wrapping_mul(0x9e37_79b1).rotate_left(13)) as i32)
            .collect();
        for (code, sym) in [
            (1u8, crate::entropy::symbol::Symbolization::Identity),
            (2, crate::entropy::symbol::Symbolization::Lane4Plain),
            (3, crate::entropy::symbol::Symbolization::Lane4ZigZag),
            (4, crate::entropy::symbol::Symbolization::DeltaLane4),
        ] {
            let streams = crate::entropy::symbol::symbolize(sym, &samples, 1).unwrap();
            let refs: Vec<&[u8]> = streams.iter().map(|s| s.as_slice()).collect();
            let mut out = vec![0i32; samples.len()];
            assert!(samples_from_streams(code, &refs, 1, &mut out), "sym {code}");
            assert_eq!(out, samples, "sym {code} exact");
        }
    }

    #[test]
    fn hostile_shapes_rejected() {
        let mut out = [0i32; 8];
        assert!(!samples_from_streams(2, &[], 1, &mut out));
        assert!(!samples_from_streams(1, &[&[0u8; 3]], 1, &mut out)); // not 4-aligned
        assert!(!samples_from_streams(4, &[&[0u8; 8]], 2, &mut out)); // < 4 lanes
        assert!(!samples_from_streams(9, &[&[0u8; 8]], 1, &mut out)); // bad sym
        assert!(!samples_from_streams(
            2,
            &[&[0u8; 4][..], &[0u8; 4][..], &[0u8; 4][..], &[0u8; 4][..]],
            0,
            &mut out
        )); // zero channels
    }
}
