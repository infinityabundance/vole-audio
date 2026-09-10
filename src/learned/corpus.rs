//! Phase O frozen corpora (`O.16`, `O.37`, `O.38`).
//!
//! Two deterministic synthetic corpora, plus adversarial controls:
//!
//! * an **intrinsic** corpus (what a learned predictor must explain on its own),
//! * a **transfer** corpus (paired source/target relationships a learned
//!   operator must beat simple analytic transforms on).
//!
//! Both are generated, license-clean and byte-reproducible. The real-recording
//! stratum of Phase M remains `VACANT_DECLARED`; nothing here pretends to be
//! recorded material, and no corpus is constructed so that a learned family is
//! expected to win by design.

use crate::hash::sha256::{Sha256, hex};

/// Corpus generation version (frozen with the court hashes).
pub const LEARNED_CORPUS_VERSION: u32 = 1;

/// Frames per intrinsic case (bounded so every court stays fast and exact).
pub const INTRINSIC_FRAMES: usize = 4096;

/// One intrinsic case.
#[derive(Debug, Clone)]
pub struct IntrinsicCase {
    pub id: &'static str,
    /// Frozen descriptive class.
    pub class: &'static str,
    /// Coarse grouping for the stratified surface.
    pub group: &'static str,
    pub channels: u8,
    pub sample_rate_hz: u32,
    pub samples: Vec<i32>,
}

/// One paired transfer case.
#[derive(Debug, Clone)]
pub struct TransferPair {
    pub id: &'static str,
    /// The deterministic relationship applied to the source.
    pub transform: &'static str,
    pub source_channels: u8,
    pub target_channels: u8,
    pub sample_rate_hz: u32,
    pub source: Vec<i32>,
    pub target: Vec<i32>,
}

const Q: i64 = 1 << 24; // amplitude scale for generated code-domain content

fn xs(u: &mut u64) -> u32 {
    *u ^= *u << 13;
    *u ^= *u >> 7;
    *u ^= *u << 17;
    (*u >> 32) as u32
}

fn clamp_code(v: i64) -> i32 {
    v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

fn sine_table(n: usize, period: f64, amp: f64, phase: f64, scale: i64) -> Vec<i32> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            (amp * (2.0 * std::f64::consts::PI * t / period + phase).sin() * scale as f64) as i32
        })
        .collect()
}

fn noise(n: usize, seed: u64, amp: i32) -> Vec<i32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            // Full-scale uniform in ±2^23, scaled by `amp` in Q24.
            let r = (xs(&mut s) as i32) >> 8;
            clamp_code((i64::from(r) * i64::from(amp)) >> 24)
        })
        .collect()
}

fn one_pole_noise(n: usize, seed: u64, a: f64) -> Vec<i32> {
    let raw = noise(n, seed, 1 << 20);
    let mut y = 0f64;
    raw.iter()
        .map(|&x| {
            y = a * y + (1.0 - a) * f64::from(x);
            clamp_code(y as i64)
        })
        .collect()
}

/// The frozen intrinsic corpus.
pub fn intrinsic_cases() -> Vec<IntrinsicCase> {
    let mut v = Vec::new();
    let mk = |id, class, group, channels, rate, samples| IntrinsicCase {
        id,
        class,
        group,
        channels,
        sample_rate_hz: rate,
        samples,
    };
    let n = INTRINSIC_FRAMES;

    // --- periodic / tonal ---
    v.push(mk(
        "sine-440",
        "pure_tone",
        "periodic",
        1,
        48_000,
        sine_table(n, 109.09, 0.8, 0.0, Q),
    ));
    v.push(mk(
        "triangle-220",
        "periodic",
        "periodic",
        1,
        48_000,
        (0..n)
            .map(|i| {
                let p = (i as i64) % 218;
                let t = if p < 109 { p } else { 218 - p };
                clamp_code((t - 54) * (Q / 64))
            })
            .collect(),
    ));
    v.push(mk(
        "square-mix",
        "periodic_rich",
        "periodic",
        1,
        48_000,
        (0..n)
            .map(|i| {
                let t = i as f64;
                let s = (2.0 * std::f64::consts::PI * t / 256.0).sin()
                    + 0.5 * (4.0 * std::f64::consts::PI * t / 256.0).sin()
                    + 0.25 * (6.0 * std::f64::consts::PI * t / 256.0).sin();
                clamp_code((s * 0.5 * Q as f64) as i64)
            })
            .collect(),
    ));
    v.push(mk(
        "stereo-unison",
        "identical_stereo",
        "periodic",
        2,
        48_000,
        {
            let mono = sine_table(n, 200.0, 0.7, 0.3, Q);
            let mut s = Vec::with_capacity(n * 2);
            for x in mono {
                s.push(x);
                s.push(x);
            }
            s
        },
    ));

    // --- quasi-periodic ---
    v.push(mk(
        "vibrato-tone",
        "quasi_periodic",
        "quasi_periodic",
        1,
        48_000,
        (0..n)
            .map(|i| {
                let t = i as f64;
                let f = 1.0 + 0.02 * (2.0 * std::f64::consts::PI * t / 1024.0).sin();
                clamp_code(
                    (0.7 * (2.0 * std::f64::consts::PI * t * f / 128.0).sin() * Q as f64) as i64,
                )
            })
            .collect(),
    ));
    v.push(mk(
        "detuned-partials",
        "quasi_periodic",
        "quasi_periodic",
        1,
        44_100,
        (0..n)
            .map(|i| {
                let t = i as f64;
                let s = (2.0 * std::f64::consts::PI * t / 100.0).sin()
                    + 0.7 * (2.0 * std::f64::consts::PI * t / 100.7).sin()
                    + 0.4 * (2.0 * std::f64::consts::PI * t / 149.3).sin();
                clamp_code((s * 0.4 * Q as f64) as i64)
            })
            .collect(),
    ));

    // --- transient / pluck-like ---
    v.push(mk(
        "pluck-decay",
        "transient_decay",
        "transient",
        1,
        48_000,
        {
            let base = sine_table(n, 120.0, 1.0, 0.0, Q);
            base.iter()
                .enumerate()
                .map(|(i, &x)| {
                    let env = (-(i as f64) / 1500.0).exp();
                    clamp_code((f64::from(x) * env) as i64)
                })
                .collect()
        },
    ));
    v.push(mk(
        "sparse-clicks",
        "sparse_transient",
        "transient",
        1,
        48_000,
        {
            let mut s = vec![0i32; n];
            let mut seed = 0xC0FFEEu64;
            for _ in 0..48 {
                let at = (xs(&mut seed) as usize) % n;
                let amp = ((xs(&mut seed) % 4) as i64 + 1) * Q / 4;
                for k in 0..64 {
                    let idx = at + k;
                    if idx >= n {
                        break;
                    }
                    let env = (-(k as f64) / 12.0).exp();
                    s[idx] = clamp_code(i64::from(s[idx]) + (amp as f64 * env) as i64);
                }
            }
            s
        },
    ));

    // --- noise / random ---
    v.push(mk(
        "white-full",
        "full_width_random",
        "noise",
        1,
        48_000,
        noise(n, 0x1234_5678, 1 << 24),
    ));
    v.push(mk(
        "pink-ish",
        "filtered_random",
        "noise",
        1,
        48_000,
        one_pole_noise(n, 0xABCD_EF01, 0.97),
    ));
    v.push(mk(
        "lowbyte-random",
        "reduced_amplitude_random",
        "noise",
        1,
        48_000,
        noise(n, 0x0BAD_F00D, 1 << 8),
    ));

    // --- speech/music-like ---
    v.push(mk(
        "formant-ish",
        "speech_like",
        "complex",
        1,
        48_000,
        (0..n)
            .map(|i| {
                let t = i as f64;
                let env = 0.5 + 0.5 * (2.0 * std::f64::consts::PI * t / 2048.0).sin();
                let s = (2.0 * std::f64::consts::PI * t / 180.0).sin()
                    * (1.0 + 0.4 * (2.0 * std::f64::consts::PI * t / 60.0).sin());
                clamp_code((s * env * 0.6 * Q as f64) as i64)
            })
            .collect(),
    ));
    v.push(mk(
        "chord-stack",
        "music_like",
        "complex",
        1,
        48_000,
        (0..n)
            .map(|i| {
                let t = i as f64;
                let s = (2.0 * std::f64::consts::PI * t / 240.0).sin()
                    + (2.0 * std::f64::consts::PI * t / 302.0).sin()
                    + (2.0 * std::f64::consts::PI * t / 359.0).sin()
                    + (2.0 * std::f64::consts::PI * t / 455.0).sin();
                clamp_code((s * 0.25 * Q as f64) as i64)
            })
            .collect(),
    ));

    // --- stationary vs nonstationary ---
    v.push(mk(
        "stationary-dc-tone",
        "stationary",
        "steady",
        1,
        48_000,
        {
            let mut s = sine_table(n, 256.0, 0.4, 0.0, Q);
            for x in s.iter_mut() {
                *x = clamp_code(i64::from(*x) + Q / 8);
            }
            s
        },
    ));
    v.push(mk("regime-switch", "nonstationary", "steady", 1, 48_000, {
        let a = sine_table(n, 256.0, 0.6, 0.0, Q);
        let b = noise(n, 0xFEED_FACE, 1 << 22);
        a.iter()
            .zip(b.iter())
            .enumerate()
            .map(|(i, (&x, &y))| if i % 1024 < 512 { x } else { y })
            .collect()
    }));

    // --- adversarial / hostile controls ---
    v.push(mk(
        "hostile-full-random",
        "adversarial_random",
        "adversarial",
        1,
        48_000,
        noise(n, 0xDEAD_BEEF_CAFEu64, 1 << 24),
    ));
    v.push(mk(
        "hostile-max-toggle",
        "adversarial_toggle",
        "adversarial",
        1,
        48_000,
        (0..n)
            .map(|i| if i % 2 == 0 { i32::MAX } else { i32::MIN })
            .collect(),
    ));
    v.push(mk(
        "hostile-antitone",
        "adversarial_phase",
        "adversarial",
        1,
        48_000,
        (0..n)
            .map(|i| {
                let p = (xs(&mut (0x9E37_79B9u64.wrapping_add(i as u64))) % 64) as i64;
                clamp_code((p - 32) * (Q / 32))
            })
            .collect(),
    ));

    v
}

/// Deterministic canonical hash of the intrinsic corpus.
pub fn intrinsic_corpus_sha256() -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"vole.audio.learned.corpus.intrinsic.v1");
    for c in intrinsic_cases() {
        h.update(c.id.as_bytes());
        h.update(&[0]);
        h.update(c.class.as_bytes());
        h.update(&[0]);
        h.update(&[c.channels]);
        h.update(&c.sample_rate_hz.to_le_bytes());
        h.update(&(c.samples.len() as u64).to_le_bytes());
        for s in &c.samples {
            h.update(&s.to_le_bytes());
        }
    }
    h.finalize()
}

// ---------------------------------------------------------------------------
// Transfer corpus
// ---------------------------------------------------------------------------

fn source_signal(n: usize, seed: u64) -> Vec<i32> {
    // A colored, broadband, non-trivial source.
    let a = one_pole_noise(n, seed, 0.6);
    let b = sine_table(n, 331.0, 0.3, 0.4, Q);
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| clamp_code(i64::from(x) / 2 + i64::from(y)))
        .collect()
}

fn fir3(x: &[i32], taps: [f64; 3]) -> Vec<i32> {
    let mut out = vec![0i32; x.len()];
    for t in 0..x.len() {
        let mut acc = 0f64;
        for (k, &a) in taps.iter().enumerate() {
            if t >= k {
                acc += a * f64::from(x[t - k]);
            }
        }
        out[t] = clamp_code(acc as i64);
    }
    out
}

fn iir1(x: &[i32], a: f64, b: f64) -> Vec<i32> {
    let mut y = 0f64;
    x.iter()
        .map(|&v| {
            y = a * y + b * f64::from(v);
            clamp_code(y as i64)
        })
        .collect()
}

/// The frozen transfer corpus: deliberately simple relationships so a learned
/// operator must compete against strong analytic baselines (`O.37`).
pub fn transfer_pairs() -> Vec<TransferPair> {
    let n = INTRINSIC_FRAMES;
    let s1 = source_signal(n, 0x1357_9BDF);
    let s2 = source_signal(n, 0x2468_ACE0);

    // A short-room impulse response (causal, 64 taps).
    let mut room = vec![0f64; 64];
    room[0] = 0.7;
    room[7] = 0.35;
    room[23] = 0.2;
    room[41] = -0.12;
    room[59] = 0.07;
    let convolve = |x: &[i32], ir: &[f64]| -> Vec<i32> {
        let mut out = vec![0i32; x.len()];
        for t in 0..x.len() {
            let mut acc = 0f64;
            for (k, &h) in ir.iter().enumerate() {
                if t >= k {
                    acc += h * f64::from(x[t - k]);
                }
            }
            out[t] = clamp_code(acc as i64);
        }
        out
    };

    let mk = |id, transform, source: Vec<i32>, target: Vec<i32>| TransferPair {
        id,
        transform,
        source_channels: 1,
        target_channels: 1,
        sample_rate_hz: 48_000,
        source,
        target,
    };

    let mut v = Vec::new();
    v.push(mk("identity", "identity", s1.clone(), s1.clone()));
    v.push(mk(
        "gain-0.5",
        "fixed_gain",
        s1.clone(),
        s1.iter().map(|&x| x / 2).collect(),
    ));
    v.push(mk(
        "gain-1.5",
        "fixed_gain",
        s1.clone(),
        s1.iter()
            .map(|&x| clamp_code(i64::from(x) * 3 / 2))
            .collect(),
    ));
    v.push(mk(
        "affine",
        "affine",
        s1.clone(),
        s1.iter()
            .map(|&x| clamp_code(i64::from(x) / 2 + 100_000))
            .collect(),
    ));
    v.push(mk("fir3", "fir", s1.clone(), fir3(&s1, [0.6, 0.3, 0.1])));
    v.push(mk("iir1", "iir", s1.clone(), iir1(&s1, 0.8, 0.4)));
    v.push(mk("room", "convolution", s1.clone(), convolve(&s1, &room)));
    v.push(mk("delay-13", "delay", s1.clone(), {
        let mut out = vec![0i32; n];
        out[13..].copy_from_slice(&s1[..n - 13]);
        out
    }));
    v.push(mk(
        "poly-softclip",
        "static_polynomial",
        s1.clone(),
        s1.iter()
            .map(|&x| {
                let u = f64::from(x) / (Q as f64);
                clamp_code((Q as f64 * (u - u * u * u / 3.0)) as i64)
            })
            .collect(),
    ));
    v.push(mk(
        "piecewise-halfwave",
        "piecewise_linear",
        s1.clone(),
        s1.iter()
            .map(|&x| if x > 0 { x / 2 } else { x / 4 })
            .collect(),
    ));
    v.push(mk("bandlimit-b0.25", "bandwidth_limit", s1.clone(), {
        // A length-9 moving average: band-limiting.
        let mut out = vec![0i32; n];
        for t in 0..n {
            let mut acc = 0i64;
            for k in 0..9 {
                if t >= k {
                    acc += i64::from(s1[t - k]);
                }
            }
            out[t] = clamp_code(acc / 9);
        }
        out
    }));
    v.push(mk("spatial-delay-mix", "spatial", s2.clone(), {
        let mut out = vec![0i32; n];
        for t in 0..n {
            let d = if t >= 31 { s2[t - 31] } else { 0 };
            out[t] = clamp_code(i64::from(s2[t]) / 2 + i64::from(d) / 2);
        }
        out
    }));
    v.push(mk(
        "dynamic-envelope",
        "dynamics",
        s2.clone(),
        s2.iter()
            .enumerate()
            .map(|(i, &x)| {
                let env = 0.6 + 0.4 * (i as f64 / 512.0).sin();
                clamp_code((f64::from(x) * env) as i64)
            })
            .collect(),
    ));
    v.push(mk(
        "procedural-to-sampled",
        "procedural_manifestation",
        s1.clone(),
        fir3(&iir1(&s1, 0.5, 0.7), [0.8, 0.2, -0.05]),
    ));
    v
}

/// Deterministic canonical hash of the transfer corpus.
pub fn transfer_corpus_sha256() -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"vole.audio.learned.corpus.transfer.v1");
    for p in transfer_pairs() {
        h.update(p.id.as_bytes());
        h.update(&[0]);
        h.update(p.transform.as_bytes());
        h.update(&[0]);
        h.update(&[p.source_channels, p.target_channels]);
        h.update(&p.sample_rate_hz.to_le_bytes());
        for s in &p.source {
            h.update(&s.to_le_bytes());
        }
        for s in &p.target {
            h.update(&s.to_le_bytes());
        }
    }
    h.finalize()
}

/// Human-readable corpus identity, for receipts.
pub fn intrinsic_corpus_hex() -> String {
    hex(&intrinsic_corpus_sha256())
}

/// Human-readable transfer corpus identity, for receipts.
pub fn transfer_corpus_hex() -> String {
    hex(&transfer_corpus_sha256())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpora_are_deterministic_and_well_formed() {
        let a = intrinsic_corpus_sha256();
        let b = intrinsic_corpus_sha256();
        assert_eq!(a, b);
        assert_eq!(transfer_corpus_sha256(), transfer_corpus_sha256());
        assert_ne!(a, transfer_corpus_sha256());
        for c in intrinsic_cases() {
            let ch = usize::from(c.channels);
            assert_eq!(c.samples.len(), INTRINSIC_FRAMES * ch, "{}", c.id);
            assert!(c.samples.iter().any(|&s| s != 0), "{} is degenerate", c.id);
        }
        for p in transfer_pairs() {
            assert_eq!(p.source.len(), p.target.len(), "{}", p.id);
            assert!(p.source.iter().any(|&s| s != 0), "{}", p.id);
        }
    }

    #[test]
    fn classes_cover_the_required_families() {
        let classes: Vec<&str> = intrinsic_cases().iter().map(|c| c.class).collect();
        for want in [
            "full_width_random",
            "reduced_amplitude_random",
            "periodic",
            "quasi_periodic",
            "sparse_transient",
            "nonstationary",
            "adversarial_random",
        ] {
            assert!(classes.contains(&want), "missing class {want}");
        }
        let transforms: Vec<&str> = transfer_pairs().iter().map(|p| p.transform).collect();
        for want in [
            "identity",
            "fixed_gain",
            "affine",
            "fir",
            "iir",
            "convolution",
            "static_polynomial",
            "piecewise_linear",
            "bandwidth_limit",
        ] {
            assert!(transforms.contains(&want), "missing transform {want}");
        }
    }
}
