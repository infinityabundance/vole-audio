//! Frozen entropy development corpus (H.2.30).
//!
//! A small, hard-to-game corpus for the entropy courts. Everything is a
//! **reproducible generated fixture** (this repository contains no external
//! recordings; the license-clean public-corpus requirement is documented in
//! `docs/PHASE_H2.md` and any external object must carry source/license/hash/
//! conversion path before admission — courts report external entries as
//! `NOT_AVAILABLE` rather than inventing data).
//!
//! The corpus deliberately spans the structure axis from perfectly
//! deterministic (silence, DC, tones) through stochastic-but-structured to
//! incompressible controls (white noise, cryptographic/random-like, bit-
//! scrambled). Procedural/entropy hypotheses must win on the former and lose
//! (fall back toward RAW / literal) on the latter; that asymmetry is the
//! phase's research result, and the corpus is frozen so the court cannot be
//! tuned around it.

use crate::hash::sha256::Sha256;
use crate::object::descriptor::{ObjectDescriptor, Representation};
use crate::universe::layout::Layout;

/// One corpus fixture: canonical interleaved i32 codes.
#[derive(Debug, Clone)]
pub struct Fixture {
    pub name: &'static str,
    /// Class label (structural/negative-control category).
    pub kind: &'static str,
    pub channels: u8,
    pub samples: Vec<i32>,
    /// Generator identity + parameters (the "source" of a generated fixture).
    pub source: &'static str,
}

impl Fixture {
    pub fn frames(&self) -> usize {
        self.samples.len() / usize::from(self.channels)
    }

    /// Canonical U1 content hash of the fixture samples.
    pub fn u1_hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for s in &self.samples {
            hasher.update(&s.to_le_bytes());
        }
        hasher.finalize()
    }

    /// Semantic literal descriptor for the fixture.
    pub fn descriptor(&self) -> Option<ObjectDescriptor> {
        let layout = Layout::checked(self.channels)?;
        ObjectDescriptor::new(Representation::Literal, self.frames() as u64, layout, None)
    }
}

/// Deterministic splitmix64 stream (documented generator; not the semantic
/// universe PRNG — fixtures only).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Fixture definitions. `frames` is the number of frames per fixture.
fn build(name: &'static str, kind: &'static str, channels: u8, frames: usize) -> Fixture {
    let ch = usize::from(channels);
    let mut state: u64 = 0x6d6574612d75776f; // "meta-uwo" domain separation
    for b in name.bytes() {
        state = state.wrapping_mul(31).wrapping_add(u64::from(b));
    }
    let mut samples = Vec::with_capacity(frames * ch);
    let table = crate::universe::phase::sine_table_i32();
    let sine = |phase: usize| -> i64 { i64::from(table[phase & 4095]) };
    for f in 0..frames {
        let fi = f as i64;
        let v: i64 = match name {
            "silence" => 0,
            "dc" => 1 << 23,
            "dc-negative" => -(1 << 23),
            "single-sine" => (sine(f * 64) * (1 << 10)) >> 30,
            "harmonic-tone" => {
                (sine(f * 64) * (1 << 12) + sine(f * 197) * (1 << 9) + sine(f * 331) * (1 << 7))
                    >> 30
            }
            "quasi-periodic" => {
                let wob = (16 + ((fi * 7) % 128)) as usize;
                (sine(f * wob) * (1 << 14)) >> 30
            }
            "impulse-train" => {
                if fi % 512 == 0 {
                    1 << 24
                } else {
                    0
                }
            }
            "transient-heavy" => {
                let burst = if fi % 4096 < 96 { 1 << 22 } else { 0 };
                let decay = if fi % 4096 < 96 { 96 - fi % 4096 } else { 0 };
                burst + ((decay * decay) >> 3) - (if fi % 8192 == 0 { 1 << 21 } else { 0 })
            }
            "am-signal" => {
                let carrier = sine(f * 96);
                let modr = 1 + ((sine(f * 3) >> 30) + (1 << 16)) / 2;
                (carrier * modr) >> 30
            }
            "fm-signal" => {
                let dev = 24 + ((sine(f * 5) >> 30) as usize) % 64;
                (sine(f * dev + f / 32) * (1 << 14)) >> 30
            }
            "stereo-correlated" => {
                // Mono content; per-channel handling below.
                (sine(f * 71) * (1 << 12)) >> 30
            }
            "white-noise" => {
                // Full 32-bit spread: byte-flat negative control (a limited
                // amplitude would let byte-lane models compress sign- and
                // zero-extension bytes — not a true incompressible control).
                let r = splitmix64(&mut state);
                (r as u32) as i64 - (1i64 << 31)
            }
            "random-control" => {
                // Full 32-bit spread: every LE byte is near-uniform.
                let r = splitmix64(&mut state);
                (r as u32) as i64 - (1i64 << 31)
            }
            "scrambled-control" => {
                // Bit-reversal scramble of a noise stream: uniform but with
                // destroyed byte structure (compressed-like control).
                let r = splitmix64(&mut state).reverse_bits();
                (r as u32) as i64 - (1i64 << 31)
            }
            _ => 0,
        };
        // Per-channel shaping.
        for c in 0..ch {
            let sample = if name == "stereo-correlated" {
                if c == 0 {
                    v
                } else {
                    // Correlated with a fixed phase offset and slight level
                    // difference (reversible stereo structure).
                    (sine((f + 4000) * 71) * (1 << 11)) >> 30
                }
            } else if name == "white-noise" {
                v
            } else {
                if c == 0 {
                    v
                } else if name == "harmonic-tone" || name == "single-sine" {
                    -v // anticorrelated second channel
                } else {
                    v
                }
            };
            samples.push(sample.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
        }
    }
    Fixture {
        name,
        kind,
        channels,
        samples,
        source: "generated: deterministic closed-form/splitmix64 (corpus.rs)",
    }
}

/// The full frozen corpus (iterated by the entropy courts in fixed order).
pub fn all() -> Vec<Fixture> {
    // Structural/deterministic content first; negative controls last.
    let names: &[(&str, &str, u8, usize)] = &[
        ("silence", "structural", 1, 16384),
        ("dc", "structural", 1, 16384),
        ("dc-negative", "structural", 1, 16384),
        ("single-sine", "structural", 2, 16384),
        ("harmonic-tone", "structural", 2, 16384),
        ("quasi-periodic", "structural", 1, 16384),
        ("impulse-train", "structural", 1, 16384),
        ("transient-heavy", "transient", 1, 16384),
        ("am-signal", "structured", 1, 16384),
        ("fm-signal", "structured", 1, 16384),
        ("stereo-correlated", "structured", 2, 16384),
        ("white-noise", "negative-control", 1, 16384),
        ("random-control", "negative-control", 2, 16384),
        ("scrambled-control", "negative-control", 1, 16384),
    ];
    names
        .iter()
        .map(|&(name, kind, channels, frames)| build(name, kind, channels, frames))
        .collect()
}

/// Look up one fixture by name.
pub fn named(name: &str) -> Option<Fixture> {
    all().into_iter().find(|f| f.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_is_frozen_and_deterministic() {
        let a = all();
        let b = all();
        assert!(!a.is_empty());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.name, y.name);
            assert_eq!(x.samples, y.samples);
            assert_eq!(x.u1_hash(), y.u1_hash());
        }
        // Names unique.
        let names: std::collections::BTreeSet<_> = a.iter().map(|f| f.name).collect();
        assert_eq!(names.len(), a.len());
        // Negative controls exist and are distinct from structural content.
        assert!(a.iter().any(|f| f.kind == "negative-control"));
    }

    #[test]
    fn structural_content_is_low_entropy() {
        // Sanity anchor: a perfect sine (4096-table periodic, 64-step
        // increment) repeats every 64 frames *within* one period alignment;
        // measure byte entropy of the low byte lane (0) — must be far below
        // 8 bits for the delta/zigzag representations to win later.
        let sine = named("single-sine").unwrap();
        let low: Vec<u8> = sine
            .samples
            .iter()
            .take(4096)
            .map(|s| (s.wrapping_mul(31)) as u8)
            .collect();
        let mut counts = [0u64; 256];
        for &b in &low {
            counts[b as usize] += 1;
        }
        let n = low.len() as f64;
        let h: f64 = counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / n;
                -p * p.log2()
            })
            .sum();
        assert!(h < 7.9, "structured fixture must not be flat (h={h})");
    }

    #[test]
    fn negative_controls_are_high_entropy() {
        let noise = named("random-control").unwrap();
        let bytes: Vec<u8> = noise
            .samples
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .take(65536)
            .collect();
        let mut counts = [0u64; 256];
        for &b in &bytes {
            counts[b as usize] += 1;
        }
        let n = bytes.len() as f64;
        let h: f64 = counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / n;
                -p * p.log2()
            })
            .sum();
        assert!(h > 7.95, "random control must be near-uniform (h={h})");
    }

    #[test]
    fn fixtures_have_legal_descriptors_and_hashes() {
        for f in all() {
            let d = f.descriptor().expect("legal layout");
            assert_eq!(d.extent_frames as usize, f.frames());
            assert_eq!(f.u1_hash().len(), 32);
        }
    }
}
