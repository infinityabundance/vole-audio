//! Analytic ADSR envelope (frozen).
//!
//! The envelope is an exact function of `(params, trigger_frame,
//! note_off_frame, t)` — **no per-frame sequential state**. This makes random
//! access, chunking, GPU parallelism, and checkpoints trivial: any frame can
//! be evaluated independently.
//!
//! Freeze (U1_SPEC §"Sampler transforms — envelope"): piecewise linear,
//! values in Q16 `[0, 1<<16]`.
//!
//! * attack  `A`: frames `k = t - t_on` in `[0, A)`:
//!   `level = round(unity * k / A)` (reaches unity exactly at `k = A`);
//! * decay   `D`: frames `[0, D)` from unity to sustain:
//!   `level = unity - round((unity - sustain) * k / D)`;
//! * sustain: held until note-off;
//! * release `R`: after note-off from the level *at* note-off:
//!   `level = L0 - round(L0 * k / R)`; once `k >= R` the voice is silent.
//!
//! Zero-length segments jump instantly (A=0 ⇒ unity at `t_on`; D=0 ⇒ sustain
//! at `t_on + A`; R=0 ⇒ 0 at note-off). If note-off occurs during attack or
//! decay, release starts from the level at the note-off frame.
//! Rounding is round-half-up on positive operands.

/// Unity envelope level (Q16).
pub const ENV_UNITY: i32 = 1 << 16;

/// Exact `round-half-up(num / den)` for `num >= 0, den > 0`.
#[inline]
fn div_round_up(num: i64, den: i64) -> i64 {
    debug_assert!(num >= 0 && den > 0);
    (num + den / 2) / den
}

/// ADSR segment lengths (frames) and sustain level (Q16).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnvelopeParams {
    pub attack_frames: u32,
    pub decay_frames: u32,
    /// Sustain level in Q16, `[0, 1<<16]`.
    pub sustain_q16: i32,
    pub release_frames: u32,
}

impl EnvelopeParams {
    pub const fn new(
        attack_frames: u32,
        decay_frames: u32,
        sustain_q16: i32,
        release_frames: u32,
    ) -> Option<EnvelopeParams> {
        if sustain_q16 < 0 || sustain_q16 > ENV_UNITY {
            return None;
        }
        Some(EnvelopeParams {
            attack_frames,
            decay_frames,
            sustain_q16,
            release_frames,
        })
    }

    /// Default 1 ms attack / 40 ms decay / -6 dB sustain / 60 ms release at
    /// the given rate, truncated to whole frames (const, no closures).
    pub const fn default_at(rate_hz: u32) -> EnvelopeParams {
        EnvelopeParams {
            attack_frames: (rate_hz + 500) / 1000,
            decay_frames: (40 * rate_hz + 500) / 1000,
            sustain_q16: ENV_UNITY / 2,
            release_frames: (60 * rate_hz + 500) / 1000,
        }
    }

    /// True if the voice is silent at `t` (never triggered or fully
    /// released).
    pub fn silent_at(&self, t_on: i64, t_off: Option<i64>, t: i64) -> bool {
        if t < t_on {
            return true;
        }
        match t_off {
            None => false,
            Some(off) => {
                if t < off {
                    return false;
                }
                if self.release_frames == 0 {
                    return true;
                }
                (t - off) >= i64::from(self.release_frames)
            }
        }
    }

    /// Envelope level (Q16) at frame `t`.
    pub fn level_at(&self, t_on: i64, t_off: Option<i64>, t: i64) -> i32 {
        if t < t_on {
            return 0;
        }
        match t_off {
            Some(off) if t >= off => {
                // Release phase; start level is the value at the off frame.
                let k = t - off;
                let r = i64::from(self.release_frames);
                if r == 0 || k >= r {
                    return 0;
                }
                let l0 = i64::from(self.level_at(t_on, None, off));
                let drop = div_round_up(l0 * k, r).min(l0);
                (l0 - drop) as i32
            }
            _ => {
                let k = t - t_on;
                let a = i64::from(self.attack_frames);
                if a == 0 || k >= a {
                    // Attack over: decay then sustain.
                    let kd = k - a;
                    let d = i64::from(self.decay_frames);
                    if d == 0 || kd >= d {
                        return self.sustain_q16;
                    }
                    let span = i64::from(ENV_UNITY) - i64::from(self.sustain_q16);
                    (i64::from(ENV_UNITY) - div_round_up(span * kd, d)) as i32
                } else {
                    // Attack ramp.
                    div_round_up(i64::from(ENV_UNITY) * k, a) as i32
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(a: u32, d: u32, s: i32, r: u32) -> EnvelopeParams {
        EnvelopeParams::new(a, d, s, r).unwrap()
    }

    #[test]
    fn validation() {
        assert!(EnvelopeParams::new(0, 0, ENV_UNITY, 0).is_some());
        assert!(EnvelopeParams::new(0, 0, ENV_UNITY + 1, 0).is_none());
        assert!(EnvelopeParams::new(0, 0, -1, 0).is_none());
    }

    #[test]
    fn attack_reaches_unity_and_is_linear() {
        let e = p(4, 0, ENV_UNITY, 0);
        assert_eq!(e.level_at(100, None, 99), 0); // before trigger
        assert_eq!(e.level_at(100, None, 100), 0); // k=0 -> 0
        assert_eq!(e.level_at(100, None, 101), ENV_UNITY / 4);
        assert_eq!(e.level_at(100, None, 102), ENV_UNITY / 2);
        assert_eq!(e.level_at(100, None, 103), 3 * ENV_UNITY / 4);
        assert_eq!(e.level_at(100, None, 104), ENV_UNITY); // sustain
    }

    #[test]
    fn decay_to_sustain() {
        let e = p(0, 2, ENV_UNITY / 2, 0);
        assert_eq!(e.level_at(0, None, 0), ENV_UNITY); // attack 0
        assert_eq!(e.level_at(0, None, 1), 3 * ENV_UNITY / 4);
        assert_eq!(e.level_at(0, None, 2), ENV_UNITY / 2); // sustain
        assert_eq!(e.level_at(0, None, 1000), ENV_UNITY / 2);
    }

    #[test]
    fn release_from_sustain_and_from_attack() {
        let e = p(0, 0, ENV_UNITY, 4);
        assert_eq!(e.level_at(0, Some(10), 9), ENV_UNITY);
        assert_eq!(e.level_at(0, Some(10), 10), ENV_UNITY); // k=0
        assert_eq!(e.level_at(0, Some(10), 11), 3 * ENV_UNITY / 4);
        assert_eq!(e.level_at(0, Some(10), 13), ENV_UNITY / 4);
        assert_eq!(e.level_at(0, Some(10), 14), 0);
        assert!(e.silent_at(0, Some(10), 14));
        assert!(!e.silent_at(0, Some(10), 13));

        // Release during attack starts from the current level: at off=10 with
        // A=100 the level is round(10*65536/100) = 6554; one frame later
        // level = 6554 - round(6554/10) = 6554 - 655 = 5899.
        let e2 = p(100, 0, ENV_UNITY, 10);
        assert_eq!(e2.level_at(0, Some(10), 10), 6554);
        assert_eq!(e2.level_at(0, Some(10), 11), 5899);
    }

    #[test]
    fn monotonic_segments() {
        let e = p(8, 8, ENV_UNITY / 4, 8);
        let mut prev = 0;
        for t in 0..8 {
            let v = e.level_at(0, Some(64), t);
            assert!(v >= prev);
            prev = v;
        }
        for t in 64..72 {
            let v = e.level_at(0, Some(64), t);
            assert!(v <= prev);
            prev = v;
        }
    }

    #[test]
    fn default_params_valid() {
        let e = EnvelopeParams::default_at(48_000);
        assert_eq!(e.attack_frames, 48);
        assert_eq!(e.decay_frames, 1920);
        assert_eq!(e.sustain_q16, ENV_UNITY / 2);
        assert_eq!(e.release_frames, 2880);
    }
}
