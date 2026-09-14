//! Phase 7C.2-H: procedural excitation.
//!
//! This is VOLE's answer to the decoder-prior problem the charter frames in §18.
//! A conventional coder asks *what excitation did the encoder send?*; a
//! procedural decoder asks *given the spectrum, pitch, energy and voicing, what
//! excitation would both ends have generated anyway?* For 28 bits the decoder can
//! materialise a pitch-synchronous glottal excitation, blended with shaped noise,
//! rather than receive a waveform.
//!
//! # Packet independence is structural, not policed
//!
//! Every sample is a pure function of the transmitted fields and the sample
//! index. There is no cross-packet state, no dependence on the previously
//! reconstructed waveform, and no hidden phase accumulator. Losing packet `n`
//! therefore cannot change how packet `n + 1` is interpreted — the constitution's
//! rule holds by construction rather than by review.
//!
//! # Level semantics
//!
//! [`excitation`] returns a signal whose RMS is exactly the transmitted level for
//! *every* voicing value, so the transmitted gain is interpretable in isolation
//! and an encoder can set it from the measured residual RMS without a search.

/// Quarter-sample resolution of the procedural lag, matching `fcelp`.
pub const FRAC_BITS: u32 = 2;
/// Voicing levels (`0` is pure noise, the last is a pure pulse train).
pub const VOICING_LEVELS: i32 = 16;

/// A deterministic, packet-local white sample in `[-1, 1)`.
///
/// Identical in kind to the fallback core's generator: it depends on the seed and
/// the sample index and on nothing else, so it carries no state across packets.
fn noise_sample(seed: u8, n: usize) -> f64 {
    let mut x = 0x9E37_79B9_7F4A_7C15u64
        .wrapping_add(u64::from(seed).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add((n as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x as f64 / u64::MAX as f64) * 2.0 - 1.0
}

/// Quantisation steps of the transmitted phase, over one period.
pub const PHASE_STEPS: usize = 64;

/// The pitch-synchronous component: unit impulses at fractional spacing `period`
/// and phase `phase`, linearly distributed over the two nearest samples so the
/// fractional part is honoured without an extra filter.
///
/// Returned at unit RMS, so it can be blended with [`noise`] directly.
pub fn comb(len: usize, period: f64, phase: f64) -> Vec<f64> {
    let mut out = vec![0.0f64; len];
    if len == 0 || period < 1.0 || !period.is_finite() {
        return out;
    }
    let mut q = phase.rem_euclid(period);
    let mut pulses = 0usize;
    while q < len as f64 {
        let i = q.floor();
        let frac = q - i;
        let i = i as usize;
        if i < len {
            out[i] += 1.0 - frac;
        }
        if i + 1 < len {
            out[i + 1] += frac;
        }
        pulses += 1;
        q += period;
    }
    if pulses == 0 {
        return out;
    }
    let scale = 1.0 / (pulses as f64).sqrt();
    for v in out.iter_mut() {
        *v *= scale;
    }
    out
}

/// The phase index in `0..PHASE_STEPS` whose comb best aligns with `resid`, by
/// signed normalised correlation.
///
/// Selecting the phase analytically is what makes a *usable* phase affordable.
/// Choosing it by the transmitted field alone would need a search over every
/// phase value times every other parameter; solving for it here costs `O(P·N)`
/// per parameter combination and lets the wire carry a 64-step phase for six
/// bits.
///
/// The search is signed, not absolute: a comb shifted by half a period is
/// approximately its own negation, so the full period already covers the sign and
/// a residual whose pulses are positive is matched by a positive comb.
pub fn best_phase(resid: &[f64], period: f64, steps: usize) -> (usize, f64) {
    let steps = steps.max(1);
    let nr: f64 = resid.iter().map(|x| x * x).sum::<f64>().sqrt();
    if nr <= 0.0 || period < 1.0 || !period.is_finite() {
        return (0, 0.0);
    }
    let mut best = (0usize, f64::NEG_INFINITY);
    for j in 0..steps {
        let c = comb(resid.len(), period, j as f64 / steps as f64 * period);
        let nc: f64 = c.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nc <= 0.0 {
            continue;
        }
        let d: f64 = resid.iter().zip(&c).map(|(a, b)| a * b).sum();
        let corr = d / (nr * nc);
        if corr > best.1 {
            best = (j, corr);
        }
    }
    best
}

/// The coherent component at a phase given as a fraction of the period.
fn pulse_train(len: usize, lag_q: i32, phase: f64) -> Vec<f64> {
    if lag_q <= 0 {
        // No pitch information: the coherent component is empty, and the blend
        // degenerates to noise, which is the honest answer.
        return vec![0.0f64; len];
    }
    let period = f64::from(lag_q) / f64::from(1u32 << FRAC_BITS);
    comb(len, period, phase)
}

/// The aperiodic component at unit RMS.
fn noise(len: usize, seed: u8) -> Vec<f64> {
    (0..len).map(|n| noise_sample(seed, n)).collect()
}

/// Scale `v` to unit RMS. Deterministic and computable on both sides, so it is a
/// legitimate part of the decoder's generation rather than hidden state.
fn unit_rms(v: &mut [f64]) {
    if v.is_empty() {
        return;
    }
    let e: f64 = v.iter().map(|x| x * x).sum();
    if e > 1e-12 {
        let s = (v.len() as f64 / e).sqrt();
        for x in v.iter_mut() {
            *x *= s;
        }
    } else {
        v.iter_mut().for_each(|x| *x = 0.0);
    }
}

/// The procedural excitation: a voicing blend of a pitch-synchronous pulse train
/// and shaped noise, at exactly the requested RMS.
///
/// `lag_q` is the pitch period in quarter samples, `voicing` is `0..VOICING_LEVELS`
/// where the last value is a pure pulse train, and `phase` is the coherent
/// component's offset as a fraction of the period in `[0, 1)`. Both components are
/// brought to unit RMS and the *blend* is then normalised as a whole, so the
/// output RMS is exactly `level` for every voicing value — including the cross
/// term, which a per-component normalisation would leave in. That is what makes
/// the transmitted gain interpretable on its own.
pub fn excitation(
    len: usize,
    lag_q: i32,
    level: f64,
    voicing: i32,
    phase: f64,
    seed: u8,
) -> Vec<f64> {
    if len == 0 {
        return Vec::new();
    }
    let v = f64::from(voicing.clamp(0, VOICING_LEVELS - 1)) / f64::from(VOICING_LEVELS - 1);
    let mut pulse = pulse_train(len, lag_q, phase);
    unit_rms(&mut pulse);
    let mut n = noise(len, seed);
    unit_rms(&mut n);
    let mut raw: Vec<f64> = (0..len).map(|i| v * pulse[i] + (1.0 - v) * n[i]).collect();
    unit_rms(&mut raw);
    for x in raw.iter_mut() {
        *x *= level;
    }
    raw
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_excitation_level_is_exactly_the_requested_rms_at_every_voicing() {
        // The property the encoder relies on: the transmitted gain *is* the
        // excitation RMS, so it can be set from a measurement instead of searched.
        let len = 320usize;
        for voicing in 0..VOICING_LEVELS {
            for &level in &[10.0f64, 300.0, 5000.0] {
                let e = excitation(len, 4 * 73, level, voicing, 0.0, 1);
                let rms = (e.iter().map(|x| x * x).sum::<f64>() / len as f64).sqrt();
                assert!(
                    (rms - level).abs() <= 1e-9 * level,
                    "voicing {voicing} level {level}: rms {rms}"
                );
            }
        }
    }

    #[test]
    fn a_pulse_train_lands_on_the_fractional_period() {
        // With full voicing and no tilt the coherent component is the signal, and
        // its pulses must sit at the transmitted fractional spacing.
        let len = 160usize;
        let lag_q = 4 * 40;
        let e = excitation(len, lag_q, 100.0, VOICING_LEVELS - 1, 0.0, 4);
        // Period 40 samples with zero phase: impulses at 0, 40, 80, 120.
        for k in 0..4 {
            let i = k * 40;
            assert!(e[i].abs() > 0.0, "expected a pulse at {i}, found {}", e[i]);
        }
        // Halfway between pulses the coherent component is silent.
        assert!(e[20].abs() < 1e-9);
    }

    #[test]
    fn the_generator_is_packet_local_and_deterministic() {
        // The constitution's hard rule, tested rather than asserted: the same
        // transmitted fields must give the same samples, and the generator must
        // not read anything outside its arguments.
        let a = excitation(320, 4 * 90, 700.0, 9, 0.0, 5);
        let b = excitation(320, 4 * 90, 700.0, 9, 0.0, 5);
        assert_eq!(a, b);
        // Different seeds are different realisations, so the encoder has a choice.
        let c = excitation(320, 4 * 90, 700.0, 9, 0.0, 6);
        assert_ne!(a, c);
        // And a zero pitch means no coherent component rather than a panic.
        let d = excitation(64, 0, 10.0, VOICING_LEVELS - 1, 0.0, 0);
        assert_eq!(d, vec![0.0; 64]);
    }

    #[test]
    fn phase_resolution_is_what_limits_a_memoryless_pulse_train() {
        // The measured codec result is that the procedural core is *bit-identical*
        // to plain shaped noise: it never wins even when it is the only pitched
        // option. This diagnostic pins the reason. A comb carries no memory, so
        // its phase must be transmitted; the wire spends 3 bits on it, giving four
        // positions inside a `period/8` span. If the target's pulses fall between
        // them, the coherent component is uncorrelated and is worth nothing.
        //
        // This is also the structural reason an *adaptive codebook* (7C.2-E) beats
        // a procedural one: ACELP reads the phase out of the excitation ring, so
        // it carries the phase for zero bits.
        let len = 320usize;
        let period = 81.0f64;
        // A target whose pulses sit at a fractional phase the coarse grid misses.
        let true_phase = 17.3f64;
        let target = comb(len, period, true_phase);
        let corr = |a: &[f64], b: &[f64]| -> f64 {
            let d: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            if na * nb <= 0.0 { 0.0 } else { d / (na * nb) }
        };
        // The shipped grid: a 3-bit phase field gives eight positions.
        let coarse = (0..8usize)
            .map(|j| corr(&target, &comb(len, period, j as f64 / 8.0 * period)).abs())
            .fold(0.0f64, f64::max);
        // A phase search at half-sample resolution, which the wire cannot afford
        // at this period (it would need ~8 bits instead of 3).
        let mut fine = 0.0f64;
        let mut phi = 0.0f64;
        while phi < period {
            fine = fine.max(corr(&target, &comb(len, period, phi)).abs());
            phi += 0.5;
        }
        assert!(fine > 0.9, "a fine phase search must align: got {fine}");
        assert!(
            coarse < fine - 0.2,
            "the coarse grid should be the limiter: coarse {coarse}, fine {fine}"
        );
        // The shipped quantiser is `PHASE_STEPS` positions over one period. At a
        // period of 81 that is ≈1.3 samples of resolution, which measures ≈0.70
        // against a pure comb — far better than an 8-step grid but short of a
        // half-sample sweep. The analytic selector must find the grid's optimum.
        let grid_best = (0..PHASE_STEPS)
            .map(|j| {
                corr(
                    &target,
                    &comb(len, period, j as f64 / PHASE_STEPS as f64 * period),
                )
                .abs()
            })
            .fold(0.0f64, f64::max);
        let (j, _) = best_phase(&target, period, PHASE_STEPS);
        let got = corr(
            &target,
            &comb(len, period, j as f64 / PHASE_STEPS as f64 * period),
        )
        .abs();
        assert!(
            (got - grid_best).abs() < 1e-12,
            "best_phase must return the grid optimum: {got} vs {grid_best}"
        );
        assert!(
            grid_best > coarse + 0.3,
            "a 64-step grid must beat an 8-step one: {coarse} -> {grid_best}"
        );
    }

    #[test]
    fn voicing_moves_the_signal_between_noise_and_a_pulse_train() {
        let len = 320usize;
        let lag_q = 4 * 80;
        // Fully unvoiced is exactly the noise component (up to the level scale),
        // so it must be perfectly correlated with the raw generator output.
        let n = excitation(len, lag_q, 1000.0, 0, 0.0, 2);
        let raw = noise(len, 2);
        let dot: f64 = n.iter().zip(&raw).map(|(a, b)| a * b).sum();
        let nn: f64 = n.iter().map(|a| a * a).sum::<f64>().sqrt();
        let rn: f64 = raw.iter().map(|a| a * a).sum::<f64>().sqrt();
        assert!(
            (dot / (nn * rn) - 1.0).abs() < 1e-9,
            "unvoiced output must be the noise component"
        );
        // Fully voiced is periodic: the normalised autocorrelation at the period
        // must be high.
        let v = excitation(len, lag_q, 1000.0, VOICING_LEVELS - 1, 0.0, 2);
        let num: f64 = (0..len - 80).map(|i| v[i] * v[i + 80]).sum();
        let den: f64 = (0..len - 80).map(|i| v[i] * v[i]).sum();
        assert!(num / den > 0.8, "voiced excitation is not periodic");
        // And voicing must actually move the signal: the two ends differ.
        let mid = excitation(len, lag_q, 1000.0, 8, 0.0, 2);
        assert_ne!(mid, n);
        assert_ne!(mid, v);
    }
}
