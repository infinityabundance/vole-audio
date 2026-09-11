//! Residual entropy anatomy (fourth-pass **Seal E0**, diagnostic only).
//!
//! The S8 real-speech result already beats FLAC-8; the open question is how much
//! *conditional* structure the exact residual still carries. This module
//! binarizes an exact residual into a canonical prefix bitstream and measures
//! the empirical conditional entropy `H(bit | context)` under several
//! decoder-visible contexts, so the E-ladder knows which contexts actually carry
//! information before any coder is built.
//!
//! Binarization (frozen): for each residual `r`,
//!
//! ```text
//! sign bit   = 1 if r < 0 else 0
//! magnitude  = Exp-Golomb(0) of |r|:  let v = |r| + 1, n = bit_length(v)
//!              emit (n-1) zero bits, then the n bits of v (MSB first)
//! ```
//!
//! A residual sample of `0` therefore costs a `0` sign bit plus the single
//! Exp-Golomb bit `1` (the value field of `v = 1`); a nonzero magnitude costs the
//! sign plus `bit_length(v) - 1` unary zeros plus the `bit_length(v)` value bits.
//! The mapping is exactly invertible and keeps zero runs to two bits each.
//!
//! Every context is computable from already-decoded information plus immutable
//! object state: the current token prefix, the previous one/two residual
//! magnitudes and sign, a small magnitude FSM, a decoder-visible matched lag, a
//! local-energy bucket, and the disagreement between the selected predictor and
//! a cheap previous-sample alternative.
//!
//! ## Estimator and its caveat
//!
//! `H(bit | context)` is the plug-in (maximum-likelihood) empirical conditional
//! entropy, in **bits per residual sample**. Conditioning on any context can
//! never exceed the order-0 rate, because the plug-in mutual information is a KL
//! divergence and is therefore non-negative. The plug-in estimator is optimistic
//! when a context is sparse, so every result also reports the number of distinct
//! context values actually observed ([`Anatomy::contexts`]); a context whose
//! cardinality approaches the number of bits is measuring noise, not structure.
//!
//! ## Oracle context
//!
//! The matched lag is chosen per residual by the encoder from the residual's own
//! autocorrelation. In a real coder that lag would have to be transmitted (or
//! derived from reconstructed history); here it is an **upper-bound oracle** for
//! the lagged-residual context, and is reported as such.

use std::collections::HashMap;
use std::hash::Hash;

/// Number of magnitude bit-length buckets (bit lengths `0..=20`).
pub const MAG_BUCKETS: usize = 21;

/// Number of conditional contexts measured (the length of
/// [`Anatomy::contexts`], in the same order as the `by_*` fields).
pub const CONTEXT_COUNT: usize = 11;

/// Human names for the measured contexts, aligned with [`CONTEXT_COUNT`] and the
/// order in which the `by_*` fields appear.
pub const CONTEXT_NAMES: [&str; CONTEXT_COUNT] = [
    "order0",
    "pos",
    "prefix",
    "prev_mag",
    "prev2_mag",
    "prev_pair",
    "prev_sign",
    "fsm",
    "lag",
    "energy",
    "disagree",
];

/// Conditional-entropy results, in **bits per residual sample**.
#[derive(Debug, Clone, PartialEq)]
pub struct Anatomy {
    /// Residual samples analysed.
    pub residual_samples: u64,
    /// Total canonical bitstream length (all tokens) in bits.
    pub bits: u64,
    /// `H(bit)` — order-0 entropy.
    pub order0: f64,
    /// `H(bit | bit position within the token)`.
    pub by_pos: f64,
    /// `H(bit | full already-decoded token prefix)`.
    pub by_prefix: f64,
    /// `H(bit | previous residual magnitude bucket)`.
    pub by_prev_mag: f64,
    /// `H(bit | second-previous residual magnitude bucket)`.
    pub by_prev2_mag: f64,
    /// `H(bit | previous two magnitude buckets jointly)`.
    pub by_prev_pair: f64,
    /// `H(bit | previous residual sign)`.
    pub by_prev_sign: f64,
    /// `H(bit | magnitude-class finite-state-machine state)`.
    pub by_fsm: f64,
    /// `H(bit | matched-lag residual magnitude bucket)` (oracle lag).
    pub by_lag: f64,
    /// `H(bit | trailing local-energy bucket)`.
    pub by_energy: f64,
    /// `H(bit | selected-predictor vs previous-sample disagreement bucket)`.
    pub by_disagree: f64,
    /// Fraction of residual samples equal to zero.
    pub zero_rate: f64,
    /// Fraction of residual samples strictly negative.
    pub neg_rate: f64,
    /// Mean `|r|` over the residual.
    pub mean_abs: f64,
    /// Maximum `|r|` over the residual.
    pub max_abs: u64,
    /// The oracle matched lag chosen for this residual.
    pub lag: u32,
    /// Magnitude (bit-length) histogram, buckets `0..=20`.
    pub mag_hist: [u64; MAG_BUCKETS],
    /// Distinct context values observed, aligned with [`CONTEXT_NAMES`].
    pub contexts: [u64; CONTEXT_COUNT],
}

impl Default for Anatomy {
    fn default() -> Self {
        Anatomy {
            residual_samples: 0,
            bits: 0,
            order0: 0.0,
            by_pos: 0.0,
            by_prefix: 0.0,
            by_prev_mag: 0.0,
            by_prev2_mag: 0.0,
            by_prev_pair: 0.0,
            by_prev_sign: 0.0,
            by_fsm: 0.0,
            by_lag: 0.0,
            by_energy: 0.0,
            by_disagree: 0.0,
            zero_rate: 0.0,
            neg_rate: 0.0,
            mean_abs: 0.0,
            max_abs: 0,
            lag: 0,
            mag_hist: [0; MAG_BUCKETS],
            contexts: [0; CONTEXT_COUNT],
        }
    }
}

#[inline]
fn bit_length_u64(v: u64) -> u32 {
    // 64 - leading_zeros == 0 for v == 0, and the true bit length otherwise.
    64 - v.leading_zeros()
}

/// Magnitude bucket (`0..=20`), i.e. the bit length clamped to the histogram.
#[inline]
fn mag_bucket(m: u64) -> u64 {
    u64::from(bit_length_u64(m)).min((MAG_BUCKETS - 1) as u64)
}

/// Coarse magnitude class for the residual finite-state machine.
#[inline]
fn mag_class(m: u64) -> u64 {
    if m == 0 {
        0
    } else if m <= 2 {
        1
    } else if m <= 16 {
        2
    } else if m <= 256 {
        3
    } else {
        4
    }
}

fn feed<K: Eq + Hash>(map: &mut HashMap<K, (u64, u64)>, key: K, bit: u64) {
    let e = map.entry(key).or_insert((0, 0));
    if bit == 0 {
        e.0 += 1;
    } else {
        e.1 += 1;
    }
}

/// Empirical conditional entropy of a bit stream given per-bit keys, averaged
/// over the bit positions (so the result is in bits per emitted bit).
///
/// Contexts are visited in ascending key order so the floating-point summation
/// is byte-for-byte deterministic regardless of `HashMap` iteration order.
fn conditional_bits<K: Eq + Hash + Ord>(counts: &HashMap<K, (u64, u64)>) -> f64 {
    let mut total = 0u64;
    for &(z, o) in counts.values() {
        total += z + o;
    }
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f64;
    let mut items: Vec<(&K, &(u64, u64))> = counts.iter().collect();
    items.sort_unstable_by(|a, b| a.0.cmp(b.0));
    let mut h = 0.0f64;
    for &(_, &(z, o)) in &items {
        let n = z + o;
        if n == 0 {
            continue;
        }
        let p = n as f64 / total_f;
        let mut he = 0.0f64;
        if z > 0 {
            let pz = z as f64 / n as f64;
            he -= pz * pz.log2();
        }
        if o > 0 {
            let po = o as f64 / n as f64;
            he -= po * po.log2();
        }
        h += p * he;
    }
    h
}

/// Choose a decoder-visible matched lag `L` in `1..=128` from the residual's own
/// autocorrelation (largest positive mean lagged product).
fn matched_lag(residual: &[i32]) -> u32 {
    let n = residual.len();
    if n < 256 {
        return 1;
    }
    let max_lag = 128usize.min(n / 4).max(1);
    let sample = n.min(8192);
    let mut best_l = 1u32;
    let mut best = i128::MIN;
    for l in 1..=max_lag {
        let mut acc: i128 = 0;
        let mut cnt = 0u64;
        let mut t = l;
        while t < sample {
            acc += i128::from(residual[t]) * i128::from(residual[t - l]);
            cnt += 1;
            t += 1;
        }
        if cnt == 0 {
            continue;
        }
        let norm = acc / i128::from(cnt);
        if norm > best {
            best = norm;
            best_l = l as u32;
        }
    }
    best_l
}

/// Per-token contexts that are fixed for every bit of the token.
#[derive(Clone, Copy)]
struct TokenCtx {
    prev: u64,
    prev2: u64,
    pair: u64,
    sign: u64,
    fsm: u64,
    lag: u64,
    energy: u64,
    disagree: u64,
}

#[derive(Default)]
struct Buckets {
    order0: HashMap<u64, (u64, u64)>,
    pos: HashMap<u64, (u64, u64)>,
    prefix: HashMap<u128, (u64, u64)>,
    prev: HashMap<u64, (u64, u64)>,
    prev2: HashMap<u64, (u64, u64)>,
    pair: HashMap<u64, (u64, u64)>,
    sign: HashMap<u64, (u64, u64)>,
    fsm: HashMap<u64, (u64, u64)>,
    lag: HashMap<u64, (u64, u64)>,
    energy: HashMap<u64, (u64, u64)>,
    disagree: HashMap<u64, (u64, u64)>,
}

/// Analyse one exact residual (with its source, for the disagreement context).
pub fn anatomy(residual: &[i32], source: &[i32]) -> Anatomy {
    let n = residual.len();
    let mut a = Anatomy {
        residual_samples: n as u64,
        lag: matched_lag(residual),
        ..Default::default()
    };
    if n == 0 {
        return a;
    }
    let lag = a.lag as usize;
    let src_len = source.len();

    let mut buckets = Buckets::default();
    let mut prev_mag: u64 = 0;
    let mut prev2_mag: u64 = 0;
    let mut prev_sign: u64 = 0;
    let mut prev_class: u64 = 0;
    let mut prev2_class: u64 = 0;
    let mut zero: u64 = 0;
    let mut neg: u64 = 0;
    let mut abs_sum: u128 = 0;

    for t in 0..n {
        let r = residual[t];
        let sign = u64::from(r < 0);
        let m = u64::from(r.unsigned_abs());
        let v = m + 1; // Exp-Golomb code_num
        let nbits = bit_length_u64(v);

        if m == 0 {
            zero += 1;
        }
        if r < 0 {
            neg += 1;
        }
        abs_sum += u128::from(m);
        a.max_abs = a.max_abs.max(m);
        a.mag_hist[mag_bucket(m) as usize] += 1;

        // Local energy: mean |r| over a trailing window of 64 (decoder-visible:
        // it uses only already-decoded residual samples).
        let lo = t.saturating_sub(64);
        let mut esum: u64 = 0;
        for &x in &residual[lo..=t] {
            esum += u64::from(x.unsigned_abs());
        }
        let energy = mag_bucket(esum / (t - lo + 1) as u64);

        let ctx = TokenCtx {
            prev: mag_bucket(prev_mag),
            prev2: mag_bucket(prev2_mag),
            pair: mag_bucket(prev_mag) * (MAG_BUCKETS as u64) + mag_bucket(prev2_mag),
            sign: prev_sign,
            fsm: prev_class * 5 + prev2_class,
            lag: if t >= lag {
                mag_bucket(u64::from(residual[t - lag].unsigned_abs()))
            } else {
                MAG_BUCKETS as u64 // "no lag yet" bucket
            },
            energy,
            disagree: {
                // The selected predictor's hypothesis is `source[t] - r`; the
                // cheap alternative is the previous reconstructed sample. Both
                // are known to the decoder before the current residual bit.
                let h_sel = source.get(t).map_or(0i64, |&s| i64::from(s)) - i64::from(r);
                let h_prev = if t >= 1 && t - 1 < src_len {
                    i64::from(source[t - 1])
                } else {
                    0
                };
                mag_bucket((h_sel - h_prev).unsigned_abs())
            },
        };

        let emit = |b: &mut Buckets, bit: u64, pos: u64, prefix: u128, plen: u64| {
            feed(&mut b.order0, 0, bit);
            feed(&mut b.pos, pos, bit);
            feed(&mut b.prefix, (u128::from(plen) << 64) | prefix, bit);
            feed(&mut b.prev, ctx.prev, bit);
            feed(&mut b.prev2, ctx.prev2, bit);
            feed(&mut b.pair, ctx.pair, bit);
            feed(&mut b.sign, ctx.sign, bit);
            feed(&mut b.fsm, ctx.fsm, bit);
            feed(&mut b.lag, ctx.lag, bit);
            feed(&mut b.energy, ctx.energy, bit);
            feed(&mut b.disagree, ctx.disagree, bit);
        };

        // sign
        let mut pos: u64 = 0;
        let mut prefix: u128 = 0;
        let mut plen: u64 = 0;
        emit(&mut buckets, sign, pos, prefix, plen);
        pos += 1;
        prefix = u128::from(sign);
        plen = 1;
        // unary zeros
        for _ in 0..(nbits - 1) {
            emit(&mut buckets, 0, pos, prefix, plen);
            pos += 1;
            prefix <<= 1;
            plen += 1;
        }
        // v's nbits, MSB first
        for i in (0..nbits).rev() {
            let bit = (v >> i) & 1;
            emit(&mut buckets, bit, pos, prefix, plen);
            pos += 1;
            prefix = (prefix << 1) | u128::from(bit);
            plen += 1;
        }
        a.bits += pos;

        prev2_mag = prev_mag;
        prev_mag = m;
        prev_sign = sign;
        prev2_class = prev_class;
        prev_class = mag_class(m);
    }

    let nsamp = n as f64;
    let total_bits = a.bits as f64;
    // `conditional_bits` returns the average entropy per emitted bit; multiply
    // by the bitstream length and divide by the sample count to express every
    // result as bits per residual sample.
    let rate = |c: f64| c * total_bits / nsamp;
    a.order0 = rate(conditional_bits(&buckets.order0));
    a.by_pos = rate(conditional_bits(&buckets.pos));
    a.by_prefix = rate(conditional_bits(&buckets.prefix));
    a.by_prev_mag = rate(conditional_bits(&buckets.prev));
    a.by_prev2_mag = rate(conditional_bits(&buckets.prev2));
    a.by_prev_pair = rate(conditional_bits(&buckets.pair));
    a.by_prev_sign = rate(conditional_bits(&buckets.sign));
    a.by_fsm = rate(conditional_bits(&buckets.fsm));
    a.by_lag = rate(conditional_bits(&buckets.lag));
    a.by_energy = rate(conditional_bits(&buckets.energy));
    a.by_disagree = rate(conditional_bits(&buckets.disagree));
    a.zero_rate = zero as f64 / nsamp;
    a.neg_rate = neg as f64 / nsamp;
    a.mean_abs = abs_sum as f64 / nsamp;
    a.contexts = [
        buckets.order0.len() as u64,
        buckets.pos.len() as u64,
        buckets.prefix.len() as u64,
        buckets.prev.len() as u64,
        buckets.prev2.len() as u64,
        buckets.pair.len() as u64,
        buckets.sign.len() as u64,
        buckets.fsm.len() as u64,
        buckets.lag.len() as u64,
        buckets.energy.len() as u64,
        buckets.disagree.len() as u64,
    ];
    a
}

/// Merge `add` into `acc`.
///
/// Every `by_*` field is a rate in bits per residual sample. Because
/// `rate = total_conditional_bits / samples`, the exact aggregate rate is the
/// sample-weighted mean of the per-object rates, so the merge is exact rather
/// than an approximation. Counts, the bit total, the magnitude histogram and the
/// context cardinalities are summed. `lag` is deliberately left unchanged: a
/// single lag is not meaningful across residuals and is reported per object.
pub fn merge_into(acc: &mut Anatomy, add: &Anatomy) {
    let w0 = acc.residual_samples as f64;
    let w1 = add.residual_samples as f64;
    let denom = (w0 + w1).max(1.0);
    let mix = |a: f64, b: f64| (a * w0 + b * w1) / denom;
    acc.order0 = mix(acc.order0, add.order0);
    acc.by_pos = mix(acc.by_pos, add.by_pos);
    acc.by_prefix = mix(acc.by_prefix, add.by_prefix);
    acc.by_prev_mag = mix(acc.by_prev_mag, add.by_prev_mag);
    acc.by_prev2_mag = mix(acc.by_prev2_mag, add.by_prev2_mag);
    acc.by_prev_pair = mix(acc.by_prev_pair, add.by_prev_pair);
    acc.by_prev_sign = mix(acc.by_prev_sign, add.by_prev_sign);
    acc.by_fsm = mix(acc.by_fsm, add.by_fsm);
    acc.by_lag = mix(acc.by_lag, add.by_lag);
    acc.by_energy = mix(acc.by_energy, add.by_energy);
    acc.by_disagree = mix(acc.by_disagree, add.by_disagree);
    acc.zero_rate = mix(acc.zero_rate, add.zero_rate);
    acc.neg_rate = mix(acc.neg_rate, add.neg_rate);
    acc.mean_abs = mix(acc.mean_abs, add.mean_abs);
    acc.max_abs = acc.max_abs.max(add.max_abs);
    for i in 0..MAG_BUCKETS {
        acc.mag_hist[i] += add.mag_hist[i];
    }
    for i in 0..CONTEXT_COUNT {
        acc.contexts[i] += add.contexts[i];
    }
    acc.residual_samples += add.residual_samples;
    acc.bits += add.bits;
}

/// Empirical conditional entropy of one binarized residual given one context
/// value per residual sample, in **bits per residual sample**.
///
/// Uses the same frozen binarization as [`anatomy`] (sign, then Exp-Golomb(0) of
/// `|r|`). `contexts` must be aligned with `residual`; it is the general entry
/// point for measuring the value of a candidate decoder-visible feature (for
/// example a multi-hypothesis disagreement bucket).
#[allow(clippy::needless_range_loop)]
pub fn conditional_bits_per_sample(residual: &[i32], contexts: &[u64]) -> f64 {
    let n = residual.len();
    if n == 0 || contexts.len() != n {
        return 0.0;
    }
    let mut counts: HashMap<u64, (u64, u64)> = HashMap::new();
    let mut total_bits = 0u64;
    for (t, &r) in residual.iter().enumerate() {
        let key = contexts[t];
        let sign = u64::from(r < 0);
        let m = u64::from(r.unsigned_abs());
        let v = m + 1;
        let nb = bit_length_u64(v);
        let bits = (0..nb)
            .map(|i| u64::from(i + 1 == nb))
            .chain((0..nb - 1).rev().map(|i| (v >> i) & 1));
        feed(&mut counts, key, sign);
        total_bits += 1;
        for b in bits {
            feed(&mut counts, key, b);
            total_bits += 1;
        }
    }
    if total_bits == 0 {
        return 0.0;
    }
    conditional_bits(&counts) * (total_bits as f64) / (n as f64)
}

/// Order-0 entropy of the same binarization, in bits per residual sample.
pub fn order0_bits_per_sample(residual: &[i32]) -> f64 {
    let zeros = vec![0u64; residual.len()];
    conditional_bits_per_sample(residual, &zeros)
}

/// The bits-per-residual saved by a context relative to order-0, as a JSON
/// object keyed by [`CONTEXT_NAMES`].
pub fn savings_json(a: &Anatomy) -> serde_json::Value {
    let values = [
        a.by_pos,
        a.by_prefix,
        a.by_prev_mag,
        a.by_prev2_mag,
        a.by_prev_pair,
        a.by_prev_sign,
        a.by_fsm,
        a.by_lag,
        a.by_energy,
        a.by_disagree,
    ];
    let mut map = serde_json::Map::new();
    for (name, v) in CONTEXT_NAMES.iter().skip(1).zip(values) {
        map.insert((*name).to_string(), serde_json::json!(a.order0 - v));
    }
    serde_json::Value::Object(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anatomy_is_deterministic_and_bounded() {
        let r: Vec<i32> = (0..4096).map(|i| ((i * 37) % 51) - 25).collect();
        let src: Vec<i32> = (0..4096).map(|i| (i * 3) % 500).collect();
        let a = anatomy(&r, &src);
        let b = anatomy(&r, &src);
        assert_eq!(a, b);
        // Conditioning cannot raise the plug-in entropy above order-0.
        assert!(a.by_pos <= a.order0 + 1e-9);
        assert!(a.by_prefix <= a.order0 + 1e-9);
        assert!(a.by_prev_mag <= a.order0 + 1e-9);
        assert!(a.by_prev2_mag <= a.order0 + 1e-9);
        assert!(a.by_prev_pair <= a.order0 + 1e-9);
        assert!(a.by_prev_sign <= a.order0 + 1e-9);
        assert!(a.by_fsm <= a.order0 + 1e-9);
        assert!(a.by_lag <= a.order0 + 1e-9);
        assert!(a.by_energy <= a.order0 + 1e-9);
        assert!(a.by_disagree <= a.order0 + 1e-9);
        assert!(a.order0 > 0.0);
        assert!(a.bits > 0);
        assert_eq!(a.mag_hist.iter().sum::<u64>(), a.residual_samples);
    }

    #[test]
    fn all_zero_residual_has_zero_conditional_entropy() {
        let r = vec![0i32; 1024];
        let src = vec![0i32; 1024];
        let a = anatomy(&r, &src);
        // Each zero binarizes to "0" then "1", so order-0 is 2 bits/residual and
        // the position context removes all of it.
        assert_eq!(a.order0, 2.0);
        assert_eq!(a.by_pos, 0.0);
        assert_eq!(a.by_prefix, 0.0);
        assert_eq!(a.bits, 2 * a.residual_samples);
        assert_eq!(a.zero_rate, 1.0);
    }

    #[test]
    fn merge_is_a_sample_weighted_average() {
        let r0: Vec<i32> = (0..1000).map(|i| ((i * 7) % 13) - 6).collect();
        let r1: Vec<i32> = (0..3000).map(|i| ((i * 29) % 4001) - 2000).collect();
        let s0 = r0.clone();
        let s1 = r1.clone();
        let a0 = anatomy(&r0, &s0);
        let a1 = anatomy(&r1, &s1);
        let mut merged = Anatomy::default();
        merge_into(&mut merged, &a0);
        merge_into(&mut merged, &a1);
        assert_eq!(merged.residual_samples, 4000);
        assert_eq!(merged.bits, a0.bits + a1.bits);
        let expected = (a0.order0 * 1000.0 + a1.order0 * 3000.0) / 4000.0;
        assert!((merged.order0 - expected).abs() < 1e-9);
        assert_eq!(merged.mag_hist.iter().sum::<u64>(), 4000);
    }
}
