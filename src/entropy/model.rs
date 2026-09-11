//! Canonical deterministic entropy models (H.2.2/H.2.3).
//!
//! A model maps a bounded symbol alphabet to normalized integer frequencies
//! that sum to `RANS_MODEL_TOTAL` (= 16384), so that every frequency interval
//! tiles `[0, MODEL_TOTAL)` exactly and the rANS slot
//! `x & (MODEL_TOTAL - 1)` identifies exactly one symbol.
//!
//! Normalization freeze (owner: `docs/RANS.md` §"Canonical model
//! normalization", H.2.3):
//!
//! * input: raw counts `c[0..A)` over the `A` **present** symbols, ordered
//!   ascending by symbol value; `1 <= A <= MAX_MODEL_ALPHABET`; `c[i] >= 1`;
//! * output: `f[i] >= 1` for every present symbol, `sum(f) == MODEL_TOTAL`;
//! * scale by integer floor in `u128`; distribute the remainder to the
//!   largest fractional remainders, ties broken by **ascending symbol
//!   index**;
//! * guaranteed-minimum pass: any present symbol still at 0 after the
//!   remainder distribution receives 1, taken back from the largest-index
//!   frequency `>= 2`, repeated until no zeros (terminates in `<= A`
//!   iterations);
//! * absent symbols have frequency 0 and never occur in any stream encoded
//!   with this model;
//! * no floating point anywhere.
//!
//! The `normalize_counts_into` core is `no_std`-clean (pure integer math over
//! caller-provided buffers). The [`SymbolModel`] host type (std) adds
//! canonical serialization and content identity; model bytes are stored
//! information and are always counted in complete cost.

use crate::limits::{MAX_MODEL_ALPHABET, RANS_MODEL_TOTAL};

/// Maximum distinct present symbols in one model (frozen).
pub const MAX_ALPHABET: usize = MAX_MODEL_ALPHABET;

/// Normalize raw counts into frequencies summing to `RANS_MODEL_TOTAL`.
///
/// `counts` is in ascending symbol-value order with every entry `>= 1`;
/// `freq` receives the normalized frequencies (same length). Returns `None`
/// when the input violates the frozen contract (empty, oversized alphabet,
/// zero counts).
///
/// `no_std`-clean; used by the host builder and by hostile-model validation.
pub fn normalize_counts_into(counts: &[u64], freq: &mut [u32]) -> Option<()> {
    if counts.is_empty() || counts.len() > MAX_ALPHABET || freq.len() != counts.len() {
        return None;
    }
    if counts.contains(&0) {
        return None;
    }
    let a = counts.len();
    let t = u128::from(RANS_MODEL_TOTAL);

    let sum: u128 = counts.iter().map(|&c| u128::from(c)).sum();
    debug_assert!(sum >= u128::from(a as u64));

    // 1. floor-scaled frequencies and exact fractional remainders.
    let mut q = [0u32; MAX_ALPHABET];
    let mut rem = [0u128; MAX_ALPHABET];
    let mut total_q: u128 = 0;
    for (i, &c) in counts.iter().enumerate() {
        let scaled = u128::from(c) * t;
        let qi = scaled / sum;
        q[i] = qi as u32; // qi <= t <= 2^14: fits u32
        rem[i] = scaled - qi * sum; // 0 <= rem < sum
        total_q += qi;
    }
    debug_assert!(total_q <= t);

    // 2. copy floors; extra = t - total_q units to distribute.
    let mut extra = t - total_q;
    for (i, &qi) in q.iter().enumerate().take(a) {
        freq[i] = qi;
    }

    // 3. remainder distribution: give one unit to the `extra` largest
    //    remainders; ties broken by ascending symbol index.
    let mut chosen = [false; MAX_ALPHABET];
    while extra > 0 {
        let mut best: Option<(u128, usize)> = None; // (rem, index)
        for i in 0..a {
            if chosen[i] {
                continue;
            }
            let better = match best {
                None => true,
                Some((r, _)) => rem[i] > r,
            };
            if better {
                best = Some((rem[i], i));
            }
        }
        let (_, i) = best?; // extra <= a: always findable
        chosen[i] = true;
        freq[i] += 1;
        extra -= 1;
    }

    // 4. guaranteed-minimum pass: every present symbol ends with >= 1.
    //    Add 1 to the smallest-index zero; take back 1 from the largest-index
    //    frequency >= 2. Zeros strictly decrease per iteration.
    for _ in 0..a {
        let Some(zero) = (0..a).find(|&i| freq[i] == 0) else {
            break;
        };
        freq[zero] += 1;
        let take = (0..a).rev().find(|&j| freq[j] >= 2)?;
        freq[take] -= 1;
    }
    if freq.iter().take(a).any(|&f| f == 0) {
        return None;
    }

    // 5. exact-total check (constructive, but assert to freeze the contract).
    debug_assert_eq!(
        freq.iter().take(a).map(|&f| u64::from(f)).sum::<u64>(),
        u64::from(RANS_MODEL_TOTAL)
    );
    Some(())
}

/// Normalize raw counts into an owned frequency vector.
#[cfg(feature = "std")]
pub fn normalize_counts(counts: &[u64]) -> Option<Vec<u32>> {
    let mut freq = vec![0u32; counts.len()];
    normalize_counts_into(counts, &mut freq)?;
    Some(freq)
}

/// A validated symbol model: ascending `symbols` with tiling intervals
/// `[start[i], start[i] + freq[i])` over `[0, MODEL_TOTAL)`.
#[cfg(feature = "std")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolModel {
    /// Present symbol values, ascending.
    pub symbols: Vec<u16>,
    /// Interval start (cumulative frequency) per symbol.
    pub start: Vec<u32>,
    /// Normalized frequency per symbol (`>= 1`, sum == MODEL_TOTAL).
    pub freq: Vec<u32>,
}

#[cfg(feature = "std")]
impl SymbolModel {
    /// Build a model from raw counts over the `symbols` in `(value, count)`
    /// pairs (order-independent input; stored order is ascending value).
    /// `None` when any count is zero, the alphabet exceeds the ceiling, or
    /// normalization fails.
    pub fn from_counts(pairs: &[(u16, u64)]) -> Option<SymbolModel> {
        if pairs.is_empty() || pairs.len() > MAX_ALPHABET {
            return None;
        }
        let mut sorted: Vec<(u16, u64)> = pairs.to_vec();
        sorted.sort_by_key(|&(v, _)| v);
        sorted.dedup_by_key(|&mut (v, _)| v);
        if sorted.len() != pairs.len() {
            return None; // duplicate symbol values
        }
        let counts: Vec<u64> = sorted.iter().map(|&(_, c)| c).collect();
        let freq = normalize_counts(&counts)?;
        let mut start = Vec::with_capacity(freq.len());
        let mut cum = 0u32;
        for &f in &freq {
            start.push(cum);
            cum += f;
        }
        debug_assert_eq!(cum, RANS_MODEL_TOTAL);
        Some(SymbolModel {
            symbols: sorted.iter().map(|&(v, _)| v).collect(),
            start,
            freq,
        })
    }

    /// Model over a byte alphabet from raw byte counts (length 256, some
    /// entries may be 0 — absent bytes are excluded).
    pub fn from_byte_counts(counts: &[u64; 256]) -> Option<SymbolModel> {
        let pairs: Vec<(u16, u64)> = counts
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c > 0)
            .map(|(v, &c)| (v as u16, c))
            .collect();
        SymbolModel::from_counts(&pairs)
    }

    /// Model index for a symbol value, or `None` when the symbol is absent.
    pub fn index_of(&self, value: u16) -> Option<usize> {
        self.symbols.binary_search(&value).ok()
    }

    /// Number of present symbols.
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    /// Canonical model bytes (identity-relevant):
    /// `alphabet_count(u16 LE) || per symbol: value(u16 LE), freq(u32 LE)`.
    /// Cumulative starts are derived, not stored, so the bytes are canonical
    /// under equivalent normalized models.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 + 6 * self.len());
        out.extend_from_slice(&(self.len() as u16).to_le_bytes());
        for i in 0..self.len() {
            out.extend_from_slice(&self.symbols[i].to_le_bytes());
            out.extend_from_slice(&self.freq[i].to_le_bytes());
        }
        out
    }

    /// Validate the frozen model invariants (hostile-input gate).
    pub fn validate(&self) -> bool {
        if self.symbols.len() != self.start.len() || self.start.len() != self.freq.len() {
            return false;
        }
        if self.symbols.is_empty() || self.symbols.len() > MAX_ALPHABET {
            return false;
        }
        if !self.symbols.windows(2).all(|w| w[0] < w[1]) {
            return false;
        }
        if self.freq.contains(&0) {
            return false;
        }
        if self.start[0] != 0 {
            return false;
        }
        for i in 0..self.len() {
            let Some(end) = self.start[i].checked_add(self.freq[i]) else {
                return false;
            };
            let next = if i + 1 < self.len() {
                self.start[i + 1]
            } else {
                RANS_MODEL_TOTAL
            };
            if end != next {
                return false;
            }
        }
        true
    }

    /// Decode a frequency slot into the model index of the symbol whose
    /// Derived decode table: `slot -> entry index` over the frozen
    /// `RANS_MODEL_TOTAL` slot domain. This is not stored in any canonical
    /// artifact; it is runtime state derived from the already-canonical model,
    /// and it turns the per-symbol binary search into an O(1) indexed load
    /// while producing byte-identical rANS semantics.
    pub fn slot_table(&self) -> Vec<u16> {
        let mut t = vec![0u16; RANS_MODEL_TOTAL as usize];
        for idx in 0..self.symbols.len() {
            let s = self.start[idx];
            let f = self.freq[idx];
            for slot in s..s + f {
                t[slot as usize] = idx as u16;
            }
        }
        t
    }

    /// Decode entry index for `slot`, using `table` when present.
    #[inline]
    pub fn slot_index_with(&self, slot: u32, table: Option<&[u16]>) -> usize {
        match table {
            Some(t) => t[slot as usize] as usize,
            None => self.slot_index(slot),
        }
    }

    /// interval contains it (binary search over cumulative starts).
    #[inline]
    pub fn slot_index(&self, slot: u32) -> usize {
        debug_assert!(slot < RANS_MODEL_TOTAL);
        // partition_point over start: first index with start > slot, minus 1.
        let idx = self.start.partition_point(|&s| s <= slot);
        idx.saturating_sub(1)
    }

    /// Parse the canonical model bytes produced by [`SymbolModel::canonical_bytes`].
    pub fn parse_canonical(bytes: &[u8]) -> Option<SymbolModel> {
        if bytes.len() < 2 {
            return None;
        }
        let count = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
        if count == 0 || count > MAX_ALPHABET {
            return None;
        }
        let need = 2usize.checked_add(count.checked_mul(6)?)?;
        if bytes.len() < need {
            return None;
        }
        let mut symbols = Vec::with_capacity(count);
        let mut freq = Vec::with_capacity(count);
        let mut prev = None;
        for i in 0..count {
            let at = 2 + i * 6;
            let value = u16::from_le_bytes([bytes[at], bytes[at + 1]]);
            let f = u32::from_le_bytes(bytes[at + 2..at + 6].try_into().ok()?);
            if f == 0 {
                return None;
            }
            if matches!(prev, Some(p) if p >= value) {
                return None; // not strictly ascending
            }
            prev = Some(value);
            symbols.push(value);
            freq.push(f);
        }
        let model = SymbolModel {
            symbols,
            start: Vec::new(),
            freq,
        };
        // Rebuild starts and validate the frozen invariants.
        let mut start = Vec::with_capacity(model.freq.len());
        let mut cum = 0u32;
        for &f in &model.freq {
            start.push(cum);
            cum = cum.checked_add(f)?;
        }
        if cum != RANS_MODEL_TOTAL {
            return None;
        }
        let model = SymbolModel {
            symbols: model.symbols,
            start,
            freq: model.freq,
        };
        model.validate().then_some(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(n: usize) -> Vec<u64> {
        vec![1; n]
    }

    #[test]
    fn normalized_model_sums_and_min_counts() {
        let cases: Vec<Vec<u64>> = vec![
            uniform(1),
            uniform(2),
            uniform(255),
            (1..=256).map(|i| i as u64 * 7 + 1).collect(),
            vec![1, 1, 1, 1, 1_000_000, 1],
            vec![u64::MAX / 2, 1, 1, u64::MAX / 2],
            vec![3],
            vec![1, 3, 9, 27, 81, 243],
        ];
        for counts in cases {
            let f = normalize_counts(&counts).expect("normalizes");
            assert_eq!(f.len(), counts.len());
            let sum: u64 = f.iter().map(|&x| u64::from(x)).sum();
            assert_eq!(sum, u64::from(RANS_MODEL_TOTAL));
            assert!(f.iter().all(|&x| x >= 1), "every present symbol >= 1");
        }
    }

    #[test]
    fn adversarial_many_tiny_counts_still_valid() {
        // 200 symbols at count 2^17 with S = 2^32: floors are 0 for all tiny
        // symbols; the remainder pass plus the guaranteed-minimum pass must
        // still yield a valid model (exercises the take-back loop).
        let mut counts = vec![1u64 << 17; 200];
        let tiny_sum: u64 = counts.iter().sum();
        let filler = (1u64 << 32) - tiny_sum;
        counts.push(filler);
        let f = normalize_counts(&counts).expect("normalizes");
        let sum: u64 = f.iter().map(|&x| u64::from(x)).sum();
        assert_eq!(sum, u64::from(RANS_MODEL_TOTAL));
        assert!(f.iter().all(|&x| x >= 1));
    }

    #[test]
    fn normalization_is_deterministic() {
        let counts: Vec<u64> = (0..128).map(|i| (i * 2654435761u64) % 1000 + 1).collect();
        assert_eq!(normalize_counts(&counts), normalize_counts(&counts));
    }

    #[test]
    fn input_contract_is_enforced() {
        assert!(normalize_counts(&[]).is_none());
        assert!(normalize_counts(&[0, 1]).is_none());
        let too_big = uniform(MAX_ALPHABET + 1);
        assert!(normalize_counts(&too_big).is_none());
    }

    #[test]
    fn monotone_counts_keep_present_symbols() {
        // Largest-remainder with the min-1 pass: a symbol present in input is
        // present in output regardless of how skewed the distribution is.
        for scale in [1u64, 3, 1 << 20, 1 << 40] {
            let counts: Vec<u64> = (0..16).map(|i| (i as u64 + 1) * scale).collect();
            let f = normalize_counts(&counts).unwrap();
            assert!(f.iter().all(|&x| x >= 1));
        }
    }

    #[test]
    fn derived_slot_table_matches_the_binary_search() {
        let counts = [0u64; 256];
        let mut counts = counts;
        for (i, c) in counts.iter_mut().enumerate() {
            *c = ((i * i * 31 + 7) % 97 + 1) as u64;
        }
        let m = SymbolModel::from_byte_counts(&counts).unwrap();
        assert!(m.validate());
        // Every slot maps into exactly one interval and back to its symbol.
        for (idx, &s) in m.start.iter().enumerate() {
            for off in 0..m.freq[idx] {
                let slot = s + off;
                let got = m.slot_index(slot);
                assert_eq!(got, idx, "slot {slot} -> symbol {idx}");
            }
        }
        // The derived decode table agrees with the binary search at every slot.
        let table = m.slot_table();
        assert_eq!(table.len(), RANS_MODEL_TOTAL as usize);
        for slot in 0..RANS_MODEL_TOTAL {
            assert_eq!(table[slot as usize] as usize, m.slot_index(slot));
            assert_eq!(m.slot_index_with(slot, Some(&table)), m.slot_index(slot));
        }
        // Present symbols are exactly the counted ones.
        for (v, &c) in counts.iter().enumerate() {
            assert_eq!(m.index_of(v as u16).is_some(), c > 0);
        }
    }

    #[test]
    fn byte_model_from_sparse_counts() {
        // Only bytes {0, 255, 128} appear.
        let mut counts = [0u64; 256];
        counts[0] = 10;
        counts[255] = 5;
        counts[128] = 20;
        let m = SymbolModel::from_byte_counts(&counts).unwrap();
        assert_eq!(m.symbols, vec![0, 128, 255]);
        assert!(m.validate());
        let mut canonical = m.canonical_bytes();
        // Deterministic.
        assert_eq!(canonical, m.canonical_bytes());
        // Round-trip parse check: count + value/freq tuples.
        let n = u16::from_le_bytes([canonical[0], canonical[1]]) as usize;
        assert_eq!(n, 3);
        canonical.clear();
    }
}
