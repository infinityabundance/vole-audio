//! Decision-trace rANS (Phase 6 mechanism 6, `DecisionTraceRans`).
//!
//! rANS is a stack: an encoder pushes symbols and the decoder pops them in the
//! reverse order. A **forward-adaptive** model is therefore awkward — the
//! interval for symbol `t` depends on the symbols before it, which the encoder
//! has not yet emitted when it walks backward.
//!
//! `DecisionTraceRans` resolves this by separating the two passes:
//!
//! 1. run the adaptive model **forward** over the symbols, recording each
//!    symbol's `(start, freq)` interval under the model state at that point;
//! 2. emit the recorded intervals **backward** through rANS.
//!
//! The decoder replays the model forward, deriving each interval from the
//! symbols it has already decoded, so it reconstructs exactly the intervals the
//! encoder recorded — without a single bit of model state being transmitted.
//!
//! The mechanism is generic over [`AdaptiveCategorical`], so the same trace
//! drives an order-0 model or a context-adaptive one whose context is a
//! function of the symbols already coded. [`crate::learned::residual_codec2`]'s
//! `EmaRans` (id 25) predates this module and inlines the order-0 pattern;
//! `DecisionTrace` (id 26) is the context-adaptive instantiation built on the
//! primitive here.
//!
//! ## Why a *history* function rather than "the previous symbol"
//!
//! A context model is only as good as its context. Passing the already-coded
//! prefix as a slice lets the caller condition on several previous symbols
//! (magnitude bands, zero runs, sign transitions) without the primitive having
//! to guess how much history is useful. The encoder and decoder call the same
//! closure with the same prefix at the same position, so the derived context is
//! identical on both sides by construction.

use crate::entropy::rans::{
    BackSink, FwdReader, RansState, SCALE_BITS, dec_advance, dec_init, dec_slot, enc_flush, enc_put,
};
use crate::error::{Error, Result};

/// An adaptive categorical model with a fixed alphabet, driven symbol by symbol.
///
/// `context` is a decoder-visible index selected from the symbols already
/// coded; an order-0 model reports [`AdaptiveCategorical::contexts`] `== 1` and
/// ignores it.
pub trait AdaptiveCategorical {
    /// Number of contexts (at least one).
    fn contexts(&self) -> usize;
    /// Number of symbols in the alphabet.
    fn alphabet(&self) -> usize;
    /// `(start, freq)` of `sym` under context `ctx` in the current state.
    fn interval(&self, ctx: usize, sym: usize) -> (u32, u32);
    /// Advance the model after coding `sym` under context `ctx`.
    fn update(&mut self, ctx: usize, sym: usize);
    /// The symbol owning `slot` under context `ctx` in the current state.
    fn symbol_for(&self, ctx: usize, slot: u32) -> usize;
}

/// Encode one chunk of symbols with a forward-recorded, backward-emitted trace.
///
/// `ctx_of` receives the prefix of symbols already coded **within this chunk**
/// (empty for the first symbol) and selects the context for the next one. The
/// model is advanced by the forward pass exactly as the decoder will advance it.
/// Returns the encoded region (`state || renorm bytes`), or `None` when `out` is
/// too small.
pub fn encode_chunk<'a, M, F>(
    model: &mut M,
    symbols: &[usize],
    ctx_of: F,
    out: &'a mut [u8],
) -> Option<&'a [u8]>
where
    M: AdaptiveCategorical,
    F: Fn(&[usize]) -> usize,
{
    let n = symbols.len();
    let written = {
        // Clamp once so the recorded history is byte-identical to what the
        // decoder will rebuild (`symbol_for` only ever yields in-range symbols).
        let last = model.alphabet() - 1;
        let syms: Vec<usize> = symbols.iter().map(|&s| s.min(last)).collect();
        let mut starts = vec![0u32; n];
        let mut freqs = vec![0u32; n];
        for i in 0..n {
            let ctx = ctx_of(&syms[..i]).min(model.contexts() - 1);
            let (s, f) = model.interval(ctx, syms[i]);
            starts[i] = s;
            freqs[i] = f;
            model.update(ctx, syms[i]);
        }
        let mut sink = BackSink::new(out);
        let mut state = RansState::new();
        for i in (0..n).rev() {
            if !enc_put(&mut state, &mut sink, starts[i], freqs[i], SCALE_BITS) {
                return None;
            }
        }
        if !enc_flush(&state, &mut sink) {
            return None;
        }
        sink.bytes_written()
    };
    Some(&out[out.len() - written..])
}

/// Decode one chunk produced by [`encode_chunk`], appending `n` symbols to
/// `out`. Only the symbols appended by this call form the context history; the
/// model is advanced identically to the encoder's forward pass.
pub fn decode_chunk<M, F>(
    model: &mut M,
    bytes: &[u8],
    n: usize,
    ctx_of: F,
    out: &mut Vec<usize>,
) -> Result<()>
where
    M: AdaptiveCategorical,
    F: Fn(&[usize]) -> usize,
{
    let base = out.len();
    let mut reader = FwdReader::new(bytes);
    let mut state =
        dec_init(&mut reader).ok_or_else(|| Error::malformed("decision trace is truncated"))?;
    for _ in 0..n {
        let ctx = ctx_of(&out[base..]).min(model.contexts() - 1);
        let slot = dec_slot(&state, SCALE_BITS);
        let sym = model.symbol_for(ctx, slot).min(model.alphabet() - 1);
        let (s, f) = model.interval(ctx, sym);
        if !dec_advance(&mut state, &mut reader, s, f, SCALE_BITS) {
            return Err(Error::malformed("decision trace is truncated"));
        }
        out.push(sym);
        model.update(ctx, sym);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny order-0 count-based model, used only to exercise the trace.
    struct CountModel {
        counts: Vec<u64>,
    }

    impl CountModel {
        fn new(k: usize) -> Self {
            CountModel { counts: vec![1; k] }
        }
        fn cdf(&self) -> (Vec<u32>, u32) {
            let total: u64 = self.counts.iter().sum();
            let scale = u64::from(crate::entropy::rans::MODEL_TOTAL);
            let mut cdf = Vec::with_capacity(self.counts.len() + 1);
            cdf.push(0u32);
            let mut acc = 0u64;
            for &c in &self.counts {
                let mut f = (c * scale) / total;
                if f == 0 {
                    f = 1;
                }
                acc += f;
                cdf.push(acc as u32);
            }
            // The last point must strictly exceed its predecessor so every
            // frequency is positive; the test model only needs self-consistency.
            let last = cdf.len() - 1;
            cdf[last] = cdf[last].max(cdf[last - 1] + 1);
            let total = cdf[last];
            (cdf, total)
        }
    }

    impl AdaptiveCategorical for CountModel {
        fn contexts(&self) -> usize {
            1
        }
        fn alphabet(&self) -> usize {
            self.counts.len()
        }
        fn interval(&self, _ctx: usize, sym: usize) -> (u32, u32) {
            let (cdf, _) = self.cdf();
            (cdf[sym], cdf[sym + 1] - cdf[sym])
        }
        fn update(&mut self, _ctx: usize, sym: usize) {
            self.counts[sym] += 4;
        }
        fn symbol_for(&self, _ctx: usize, slot: u32) -> usize {
            let (cdf, _) = self.cdf();
            let mut s = 0usize;
            while s + 1 < cdf.len() - 1 && cdf[s + 1] <= slot {
                s += 1;
            }
            s
        }
    }

    #[test]
    fn forward_recorded_trace_round_trips() {
        let symbols: Vec<usize> = (0..5000).map(|i| (i * 7 + i / 3) % 5).collect();
        let mut enc = CountModel::new(5);
        let mut buf = vec![0u8; crate::entropy::rans::encode_capacity(5000).max(16)];
        let bytes = encode_chunk(&mut enc, &symbols, |_| 0, &mut buf)
            .unwrap()
            .to_vec();
        let mut dec = CountModel::new(5);
        let mut out = Vec::new();
        decode_chunk(&mut dec, &bytes, symbols.len(), |_| 0, &mut out).unwrap();
        assert_eq!(out, symbols);
    }

    #[test]
    fn history_function_is_applied_on_both_sides() {
        // A context-adaptive model whose context is the previous symbol.
        struct Ctx {
            tables: Vec<Vec<u64>>,
            k: usize,
        }
        impl AdaptiveCategorical for Ctx {
            fn contexts(&self) -> usize {
                self.k
            }
            fn alphabet(&self) -> usize {
                self.k
            }
            fn interval(&self, ctx: usize, sym: usize) -> (u32, u32) {
                let total: u64 = self.tables[ctx].iter().sum();
                let scale = u64::from(crate::entropy::rans::MODEL_TOTAL);
                let start: u64 = self.tables[ctx][..sym].iter().sum();
                let f = ((self.tables[ctx][sym] * scale) / total).max(1);
                let s = ((start * scale) / total) as u32;
                (s, f as u32)
            }
            fn update(&mut self, ctx: usize, sym: usize) {
                self.tables[ctx][sym] += 4;
            }
            fn symbol_for(&self, ctx: usize, slot: u32) -> usize {
                let total: u64 = self.tables[ctx].iter().sum();
                let scale = u64::from(crate::entropy::rans::MODEL_TOTAL);
                let mut acc = 0u64;
                for (s, &c) in self.tables[ctx].iter().enumerate() {
                    let f = ((c * scale) / total).max(1);
                    if u64::from(slot) < acc + f {
                        return s;
                    }
                    acc += f;
                }
                self.k - 1
            }
        }
        let k = 4;
        let symbols: Vec<usize> = (0..3000).map(|i| (i / 2) % k).collect();
        let mut enc = Ctx {
            tables: vec![vec![1; k]; k],
            k,
        };
        let mut buf = vec![0u8; crate::entropy::rans::encode_capacity(3000).max(16)];
        let bytes = encode_chunk(
            &mut enc,
            &symbols,
            |h| h.last().copied().unwrap_or(0),
            &mut buf,
        )
        .unwrap()
        .to_vec();
        let mut dec = Ctx {
            tables: vec![vec![1; k]; k],
            k,
        };
        let mut out = Vec::new();
        decode_chunk(
            &mut dec,
            &bytes,
            symbols.len(),
            |h| h.last().copied().unwrap_or(0),
            &mut out,
        )
        .unwrap();
        assert_eq!(out, symbols);
    }
}
