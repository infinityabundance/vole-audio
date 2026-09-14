//! Packet-loss concealment for `vole.audio.stream.voice.exp1` (7C.5).
//!
//! A missing packet means the source samples are **unknown**. Concealment is
//! therefore never described as reconstruction anywhere in this crate: it is
//! the best causal continuation the decoder can produce from state it actually
//! holds.
//!
//! ```text
//! voiced (lag > 0)   : pitch-period continuation through the last good
//!                      short-term filter, with the long-term gain decayed
//! unvoiced           : shaped noise at the last quantiser step, through the
//!                      last good short-term filter
//! long runs          : fade to the comfort-noise class, so concealment cannot
//!                      ring forever on stale pitch
//! ```
//!
//! The decay is applied to the *excitation* (long-term gain and noise
//! amplitude), never to the output samples, so the decoder's reconstruction
//! ring and the audio it has already emitted stay consistent.

use crate::lossy::predict as lp;
use crate::voice::predict as vp;
use crate::voice::predict::{FrameModel, LTPG_LEVELS, VoiceState};

/// The last good frame's model, as concealment sees it.
pub struct LastGood<'a> {
    /// The frame's model.
    pub model: FrameModel,
    /// The frame's quantised reflection codes.
    pub k_q: &'a [i32],
    /// The frame's residual gain code.
    pub gain: i32,
}

/// Concealment runs beyond this many frames abandon pitch repetition.
pub const VOICED_TAIL: u32 = 8;
/// Per-concealed-frame excitation decay.
pub const DECAY: f64 = 0.85;
/// Floor of the excitation decay.
pub const DECAY_FLOOR: f64 = 0.12;

/// The comfort spectral class used when no frame was ever decoded.
const COMFORT_K: [i32; 4] = [-6, 4, -2, 1];

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn prng_unit(state: &mut u64) -> f64 {
    let v = splitmix64(state) >> 11;
    (v as f64) / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// The excitation decay applied after `run` consecutive concealed frames.
pub fn decay_for(run: u32) -> f64 {
    DECAY.powi(run.min(64) as i32).max(DECAY_FLOOR)
}

/// The first four reflection codes of a frame, at width 6.
fn spectral4(k_q: &[i32], width: u8) -> Vec<i32> {
    let scale = 1.0 / f64::from(1u32 << vp::shift_of(width));
    let k: Vec<f64> = k_q.iter().map(|&q| f64::from(q) * scale).collect();
    vp::quantise_k(&k, 6).into_iter().take(4).collect()
}

fn noise_symbols(seed: u64, n: usize, amp: f64) -> Vec<i32> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let v = (prng_unit(&mut s) * amp).round();
        out.push(v.clamp(-32768.0, 32767.0) as i32);
    }
    out
}

/// Conceal one frame of `n` samples, advancing `state` exactly as a decoded
/// frame would.
///
/// The voiced path repeats the **excitation ring** at the last good pitch
/// period. That is not an approximation of pitch-period repetition — it *is*
/// pitch-period repetition, because the excitation ring is what the long-term
/// predictor reads on a healthy frame too. Repeating reconstructed *output*
/// samples instead would apply the short-term filter twice per period and
/// diverge within a few frames.
pub fn conceal(
    state: &mut VoiceState,
    last: Option<LastGood<'_>>,
    run: u32,
    seed: u64,
    n: usize,
    comfort_rms: f64,
) -> Vec<f64> {
    let decay = decay_for(run);
    match last {
        Some(g) => {
            let voiced = g.model.lag > 0 && run < VOICED_TAIL;
            let step = lp::step_of(g.gain);
            if voiced {
                let lag = (g.model.lag as usize).clamp(vp::MIN_LAG, vp::MAX_LAG);
                // Capture the ring before it advances, then extend it
                // periodically with a per-period decay.
                let src = state.excitation_chronological(lag);
                let mut excitation = Vec::with_capacity(n);
                for i in 0..n {
                    let periods = (i / lag) as i32 + 1;
                    excitation.push(src[i % lag] * decay.powi(periods));
                }
                let ltpg = lp::dequantize_ltpg(g.model.ltpg_q.clamp(0, LTPG_LEVELS - 1)) * decay;
                let ltpg_q = lp::quantize_ltpg(ltpg.clamp(0.0, 1.5));
                vp::synthesize_excitation(
                    state,
                    g.k_q,
                    g.model.width,
                    g.model.lag,
                    ltpg_q,
                    &excitation,
                )
            } else {
                // Unvoiced (or a long run): shaped noise at the last quantiser
                // step, through the four lowest-order coefficients only — a
                // stale high-order filter must not colour a noise burst.
                let k4 = spectral4(g.k_q, g.model.width);
                let amp = step * decay;
                let excitation: Vec<f64> = noise_symbols(seed, n, amp)
                    .into_iter()
                    .map(f64::from)
                    .collect();
                vp::synthesize_excitation(state, &k4, 6, 0, 0, &excitation)
            }
        }
        None => {
            // Nothing was ever decoded: procedural comfort noise only.
            let amp = (comfort_rms * decay).max(0.0);
            let symbols = noise_symbols(seed, n, amp);
            let excitation: Vec<f64> = symbols.into_iter().map(f64::from).collect();
            vp::synthesize_excitation(state, &COMFORT_K, 6, 0, 0, &excitation)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn last_model() -> FrameModel {
        FrameModel {
            estimator: 0,
            order: 8,
            width: 6,
            lag: 80,
            ltpg_q: 24,
        }
    }

    #[test]
    fn decay_is_monotone_and_bounded() {
        let mut prev = 1.0f64;
        for run in 0..40u32 {
            let d = decay_for(run);
            assert!(d <= prev + 1e-12, "decay must not grow at run {run}");
            assert!(d >= DECAY_FLOOR);
            prev = d;
        }
        assert_eq!(decay_for(0), 1.0);
        assert_eq!(decay_for(1000), DECAY_FLOOR);
    }

    #[test]
    fn concealment_advances_state_and_stays_finite() {
        let mut state = VoiceState::new();
        for i in 0..600 {
            state.push(3000.0 * ((i as f64) * 0.2).sin(), 0.0);
        }
        let k_q = vp::quantise_k(&[0.8, -0.4, 0.2, -0.1, 0.05, -0.02, 0.01, 0.0], 6);
        let model = last_model();
        for run in 0..40u32 {
            let out = conceal(
                &mut state,
                Some(LastGood {
                    model,
                    k_q: &k_q,
                    gain: 8,
                }),
                run,
                crate::voice::comfort_seed(u64::from(run)),
                160,
                120.0,
            );
            assert_eq!(out.len(), 160);
            for v in out {
                assert!(v.is_finite() && v.abs() <= vp::SATURATION);
            }
        }
    }

    #[test]
    fn concealment_without_history_uses_comfort_noise() {
        let mut state = VoiceState::new();
        let out = conceal(&mut state, None, 0, 7, 160, 500.0);
        assert_eq!(out.len(), 160);
        let energy: f64 = out.iter().map(|v| v * v).sum();
        assert!(energy > 0.0, "comfort noise must not be silent");
    }

    #[test]
    fn long_runs_stop_repeating_stale_pitch() {
        // The voiced path must be abandoned after VOICED_TAIL frames: a stale
        // pitch track cannot be allowed to ring indefinitely.
        let mut a = VoiceState::new();
        let mut b = VoiceState::new();
        for i in 0..600 {
            let v = 3000.0 * ((i as f64) * 0.2).sin();
            a.push(v, 0.0);
            b.push(v, 0.0);
        }
        let k_q = vp::quantise_k(&[0.8, -0.4, 0.2, -0.1, 0.05, -0.02, 0.01, 0.0], 6);
        let model = last_model();
        let early = conceal(
            &mut a,
            Some(LastGood {
                model,
                k_q: &k_q,
                gain: 8,
            }),
            1,
            11,
            160,
            120.0,
        );
        let late = conceal(
            &mut b,
            Some(LastGood {
                model,
                k_q: &k_q,
                gain: 8,
            }),
            VOICED_TAIL,
            11,
            160,
            120.0,
        );
        // The late frame is the noise path; it must not equal the periodic one.
        assert_ne!(early, late);
    }
}
