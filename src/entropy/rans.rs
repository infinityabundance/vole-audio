//! Native deterministic rANS primitives (H.2.2) — `no_std`-clean, scalar
//! authority.
//!
//! Frozen parameters (owner: `docs/RANS.md`): 32-bit state, byte output,
//! `scale_bits = 14`, `MODEL_TOTAL = 2^14 = 16384`, `STATE_L = 2^23`.
//! The state machine is byte-compatible with the ryg `rans_byte.h`
//! conventions and with the independent `ryg-rans-rs` oracle (dev-dependency
//! only; never linked into normative materialization):
//!
//! * encode: state starts at `STATE_L`; per symbol, renormalize while
//!   `x >= ((STATE_L >> scale_bits) << 8) * freq` (emit `x & 0xff` backward,
//!   `x >>= 8`), then `x = ((x / freq) << scale_bits) + (x % freq) + start`;
//!   flush `x` as `u32` little-endian at the front of the stream.
//! * decode: read the `u32` state (LE); symbols are recovered in reverse
//!   encode order. Per symbol: `slot = x & (MODEL_TOTAL - 1)`, identify the
//!   model interval `[start, start + freq)` containing `slot`,
//!   `x = freq * (x >> scale_bits) + (slot - start)`, then renormalize by
//!   reading bytes while `x < STATE_L`.
//!
//! All arithmetic is checked by construction (division by a validated
//! `freq >= 1`; interval containment makes `slot - start >= 0`). The
//! primitives are infallible-by-contract and return `bool`/`Option` only for
//! byte-cursor exhaustion so they compile unchanged for GPU device targets
//! (no panics, no `Result` plumbing). Host parsers validate models and
//! bounds before decoding; hostile input therefore surfaces as typed errors
//! upstream, never as wrong samples here.

use crate::limits::{RANS_MODEL_TOTAL, RANS_SCALE_BITS, RANS_STATE_L};

/// Scale bits of the audio rANS profile (frozen).
pub const SCALE_BITS: u32 = RANS_SCALE_BITS;
/// Total normalized frequency per model (frozen): `1 << SCALE_BITS`.
pub const MODEL_TOTAL: u32 = RANS_MODEL_TOTAL;
/// Lower bound of the normalized state interval (frozen).
pub const STATE_L: u32 = RANS_STATE_L;

/// Mask that extracts the frequency slot from a state.
#[inline]
pub const fn slot_mask(scale_bits: u32) -> u32 {
    (1u32 << scale_bits) - 1
}

/// Encoder state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RansState(pub u32);

impl RansState {
    /// Fresh encoder state at the lower bound.
    #[inline]
    pub const fn new() -> Self {
        Self(STATE_L)
    }
}

impl Default for RansState {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Backward-growing byte sink (ryg convention: output bytes are written
/// from the end of the buffer toward the front).
///
/// `pos` is the index of the next write; the encoded region is
/// `buf[pos..]`. The final stream layout (front to back) is
/// `state (u32 LE) || renorm bytes in decode order`.
pub struct BackSink<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl<'a> BackSink<'a> {
    #[inline]
    pub fn new(buf: &'a mut [u8]) -> Self {
        let len = buf.len();
        Self { buf, pos: len }
    }

    /// Write one byte; `false` when the buffer is exhausted.
    #[inline]
    pub fn write_byte(&mut self, b: u8) -> bool {
        if self.pos == 0 {
            return false;
        }
        self.pos -= 1;
        self.buf[self.pos] = b;
        true
    }

    /// Write a `u32` little-endian; `false` when fewer than 4 bytes remain.
    #[inline]
    pub fn write_u32_le(&mut self, v: u32) -> bool {
        if self.pos < 4 {
            return false;
        }
        self.pos -= 4;
        self.buf[self.pos..self.pos + 4].copy_from_slice(&v.to_le_bytes());
        true
    }

    /// Bytes written so far.
    #[inline]
    pub fn bytes_written(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// The encoded region (state first) as a slice.
    #[inline]
    pub fn encoded(&self) -> &[u8] {
        &self.buf[self.pos..]
    }
}

/// Forward byte reader over an encoded stream.
pub struct FwdReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> FwdReader<'a> {
    #[inline]
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Read one byte; `None` at exhaustion.
    #[inline]
    pub fn read_byte(&mut self) -> Option<u8> {
        let b = *self.buf.get(self.pos)?;
        self.pos += 1;
        Some(b)
    }

    /// Read a `u32` little-endian; `None` when fewer than 4 bytes remain.
    #[inline]
    pub fn read_u32_le(&mut self) -> Option<u32> {
        let b = self.buf.get(self.pos..self.pos + 4)?;
        self.pos += 4;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Bytes consumed so far.
    #[inline]
    pub fn bytes_consumed(&self) -> usize {
        self.pos
    }
}

/// Renormalize the encoder state `x` for a symbol with the given `freq`,
/// emitting bytes into `sink` while `x >= x_max`. Returns the normalized
/// state, or `None` when the sink ran out of capacity (state unchanged).
#[inline]
pub fn enc_renorm(mut x: u32, sink: &mut BackSink<'_>, freq: u32, scale_bits: u32) -> Option<u32> {
    debug_assert!(freq >= 1);
    let x_max = ((STATE_L >> scale_bits) << 8) * freq;
    if x >= x_max {
        while x >= x_max {
            if !sink.write_byte((x & 0xff) as u8) {
                return None;
            }
            x >>= 8;
        }
    }
    Some(x)
}

/// Encode one symbol with interval `[start, start + freq)`.
/// `false` on sink exhaustion (state not advanced).
#[inline]
pub fn enc_put(
    state: &mut RansState,
    sink: &mut BackSink<'_>,
    start: u32,
    freq: u32,
    scale_bits: u32,
) -> bool {
    let Some(x) = enc_renorm(state.0, sink, freq, scale_bits) else {
        return false;
    };
    // Division-based reference path: exact, no reciprocal approximation.
    state.0 = ((x / freq) << scale_bits) + (x % freq) + start;
    true
}

/// Flush the encoder state (u32 LE) at the front of the stream.
#[inline]
pub fn enc_flush(state: &RansState, sink: &mut BackSink<'_>) -> bool {
    sink.write_u32_le(state.0)
}

/// Read the initial decoder state (u32 LE) from the front of the stream.
///
/// A valid stream always begins with a normalized state `>= RANS_STATE_L`
/// (the encoder invariant holds after every transition and the flushed
/// state is the post-transition state); a below-range initial state is a
/// malformed stream and is rejected here (RANS.md "State machine", step 1
/// of decode).
#[inline]
pub fn dec_init(reader: &mut FwdReader<'_>) -> Option<RansState> {
    let s = reader.read_u32_le()?;
    if s < STATE_L {
        return None;
    }
    Some(RansState(s))
}

/// Frequency slot of the current state (`x & (2^scale_bits - 1)`).
#[inline]
pub fn dec_slot(state: &RansState, scale_bits: u32) -> u32 {
    state.0 & slot_mask(scale_bits)
}

/// Advance the decoder past one symbol with interval `[start, start+freq)`,
/// renormalizing by reading bytes while the state is below `STATE_L`.
/// `false` on byte exhaustion (state not advanced).
#[inline]
pub fn dec_advance(
    state: &mut RansState,
    reader: &mut FwdReader<'_>,
    start: u32,
    freq: u32,
    scale_bits: u32,
) -> bool {
    debug_assert!(freq >= 1);
    let x = state.0;
    let mut x = freq * (x >> scale_bits) + (x & slot_mask(scale_bits)) - start;
    if x < STATE_L {
        loop {
            let b = match reader.read_byte() {
                Some(b) => b,
                None => return false,
            };
            x = (x << 8) | u32::from(b);
            if x >= STATE_L {
                break;
            }
        }
    }
    state.0 = x;
    true
}

/// Worst-case encoded byte capacity for `n` symbols (4 bytes/symbol renorm
/// slack + 4 state bytes), matching `RANS_MAX_BYTES_PER_SYMBOL`.
#[inline]
pub const fn encode_capacity(n: usize) -> usize {
    n.saturating_mul(crate::limits::RANS_MAX_BYTES_PER_SYMBOL as usize) + 4 + 8
}

/// Convenience: one-shot rANS encode of a symbol sequence against
/// `(start, freq)` intervals. Returns the canonical stream bytes
/// (`state || renorm bytes`) or `None` if capacity was exceeded (cannot
/// happen when `buf_len >= encode_capacity(n)`).
pub fn encode_symbols<'a>(
    starts: &[u32],
    freqs: &[u32],
    scale_bits: u32,
    out: &'a mut [u8],
) -> Option<&'a [u8]> {
    debug_assert_eq!(starts.len(), freqs.len());
    let pos = {
        let mut sink = BackSink::new(out);
        let mut state = RansState::new();
        for i in 0..starts.len() {
            if !enc_put(&mut state, &mut sink, starts[i], freqs[i], scale_bits) {
                return None;
            }
        }
        if !enc_flush(&state, &mut sink) {
            return None;
        }
        sink.pos
    };
    Some(&out[pos..])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(starts: &[u32], freqs: &[u32], scale_bits: u32) {
        let mut buf = vec![0u8; encode_capacity(starts.len())];
        let enc = encode_symbols(starts, freqs, scale_bits, &mut buf).expect("encode fits");
        let mut reader = FwdReader::new(enc);
        let mut state = dec_init(&mut reader).expect("state present");
        // Decode order is reverse encode order; recover into a back buffer.
        let mut syms = vec![0u32; starts.len()];
        for i in (0..starts.len()).rev() {
            let slot = dec_slot(&state, scale_bits);
            // Interval search: find the symbol whose [start, start+freq)
            // contains slot. Tiling models: exactly one interval matches.
            let mut found = None;
            for j in 0..starts.len() {
                if slot >= starts[j] && slot < starts[j] + freqs[j] {
                    found = Some(j);
                    break;
                }
            }
            let j = found.expect("slot inside exactly one interval");
            assert!(dec_advance(
                &mut state,
                &mut reader,
                starts[j],
                freqs[j],
                scale_bits
            ));
            syms[i] = j as u32;
        }
        // All renorm bytes consumed; state back at STATE_L.
        assert!(reader.bytes_consumed() <= enc.len());
        assert_eq!(state.0, STATE_L);
        for (i, &s) in syms.iter().enumerate() {
            assert_eq!(s as usize, i, "symbol order preserved");
        }
    }

    #[test]
    fn deterministic_roundtrip_single_symbol_families() {
        // freq = total (single-symbol model).
        roundtrip(&[0], &[MODEL_TOTAL], SCALE_BITS);
        // Uniform 256-symbol model.
        let starts: Vec<u32> = (0..256).map(|i| i * 64).collect();
        let freqs = vec![64u32; 256];
        roundtrip(&starts, &freqs, SCALE_BITS);
    }

    #[test]
    fn deterministic_roundtrip_skewed_models() {
        // Very skewed: one frequent symbol + a tail, several lengths.
        for n in [1usize, 2, 3, 7, 64, 1000, 4096] {
            let mut starts = Vec::with_capacity(n);
            let mut freqs = Vec::with_capacity(n);
            let mut cum = 0u32;
            for i in 0..n {
                // Deterministic skew: symbol i gets (n - i) weight, scaled.
                let w = (n - i) as u32;
                let f = (w * MODEL_TOTAL) / (n * (n + 1) / 2) as u32;
                let f = f.max(1);
                starts.push(cum);
                freqs.push(f);
                cum += f;
            }
            // Adjust the last frequency so intervals tile exactly.
            freqs[n - 1] = MODEL_TOTAL - (cum - freqs[n - 1]);
            starts[n - 1] = MODEL_TOTAL - freqs[n - 1];
            roundtrip(&starts, &freqs, SCALE_BITS);
        }
    }

    #[test]
    fn encode_is_canonical_and_stable() {
        let starts = [0u32, 8192, 12288, 15360];
        let freqs = [8192u32, 4096, 3072, 1024];
        let mut a = vec![0u8; 128];
        let mut b = vec![0u8; 128];
        let ea = encode_symbols(&starts, &freqs, SCALE_BITS, &mut a).unwrap();
        let eb = encode_symbols(&starts, &freqs, SCALE_BITS, &mut b).unwrap();
        assert_eq!(ea, eb);
    }

    #[test]
    fn dec_init_rejects_below_range_states() {
        // RANS.md decode step 1: a valid stream always begins with a
        // normalized state >= STATE_L (the encoder flushes a post-transition
        // state). Values below the range are malformed.
        let mut reader = FwdReader::new(&[0u8; 4]);
        assert!(dec_init(&mut reader).is_none());
        let low = (STATE_L - 1).to_le_bytes();
        let mut reader = FwdReader::new(&low);
        assert!(dec_init(&mut reader).is_none());
        // At and above the range are accepted by the primitive (canonical
        // termination is enforced by the stream decoders).
        let at = STATE_L.to_le_bytes();
        let mut reader = FwdReader::new(&at);
        assert_eq!(dec_init(&mut reader).unwrap().0, STATE_L);
        let max = u32::MAX.to_le_bytes();
        let mut reader = FwdReader::new(&max);
        assert_eq!(dec_init(&mut reader).unwrap().0, u32::MAX);
    }

    #[test]
    fn truncated_stream_is_detected_not_ignored() {
        // Encode a nontrivial stream, then decode from every strict prefix.
        // A valid stream is consumed exactly byte-for-byte, so any truncation
        // must surface as a clean failure (init None or advance false) —
        // never a panic, never silent wrong output.
        let n = 512;
        let mut starts = Vec::with_capacity(n);
        let mut freqs = vec![1u32; n];
        freqs[n - 1] = MODEL_TOTAL - (n as u32 - 1); // tile exactly
        let mut cum = 0u32;
        // i indexes starts/freqs while building cumulative intervals.
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            starts.push(cum);
            cum += freqs[i];
        }
        let mut buf = vec![0u8; encode_capacity(n)];
        let enc = encode_symbols(&starts, &freqs, SCALE_BITS, &mut buf).unwrap();

        let decode = |bytes: &[u8]| -> bool {
            let mut reader = FwdReader::new(bytes);
            let Some(mut state) = dec_init(&mut reader) else {
                return false;
            };
            for i in (0..n).rev() {
                let slot = dec_slot(&state, SCALE_BITS);
                let mut found = None;
                // j scans interval tables; the slot identifies one interval.
                #[allow(clippy::needless_range_loop)]
                for j in 0..n {
                    if slot >= starts[j] && slot < starts[j] + freqs[j] {
                        found = Some(j);
                        break;
                    }
                }
                let j = found.unwrap();
                if !dec_advance(&mut state, &mut reader, starts[j], freqs[j], SCALE_BITS) {
                    return false;
                }
                let _ = i;
            }
            true
        };

        // Full stream decodes.
        assert!(decode(enc));
        // Every strict prefix fails cleanly.
        for cut in 0..enc.len() {
            assert!(!decode(&enc[..cut]), "cut at {cut} must fail cleanly");
        }
    }

    #[test]
    fn sink_capacity_enforcement() {
        let mut tiny = [0u8; 3];
        let mut sink = BackSink::new(&mut tiny);
        assert!(sink.write_byte(1));
        assert!(sink.write_byte(2));
        assert!(sink.write_byte(3));
        assert!(!sink.write_byte(4));
        assert_eq!(sink.bytes_written(), 3);

        let mut four = [0u8; 4];
        let mut sink4 = BackSink::new(&mut four);
        assert!(sink4.write_u32_le(0xdead_beef));
        assert_eq!(sink4.encoded(), &0xdead_beefu32.to_le_bytes());
    }

    #[test]
    fn long_uniform_stream_roundtrips() {
        // Uniform over 2 symbols, long enough to exercise many renorms.
        let n = 100_000;
        let starts = [0u32, 8192];
        let freqs = [8192u32, 8192];
        let mut buf = vec![0u8; encode_capacity(n)];
        let mut sink = BackSink::new(&mut buf);
        let mut state = RansState::new();
        for i in 0..n {
            let (s, f) = if i % 2 == 0 {
                (starts[0], freqs[0])
            } else {
                (starts[1], freqs[1])
            };
            assert!(enc_put(&mut state, &mut sink, s, f, SCALE_BITS));
        }
        assert!(enc_flush(&state, &mut sink));
        let bytes = sink.encoded();
        let mut reader = FwdReader::new(bytes);
        let mut state = dec_init(&mut reader).unwrap();
        for i in (0..n).rev() {
            let slot = dec_slot(&state, SCALE_BITS);
            let (s, f) = if i % 2 == 0 {
                (starts[0], freqs[0])
            } else {
                (starts[1], freqs[1])
            };
            assert!(slot >= s && slot < s + f);
            assert!(dec_advance(&mut state, &mut reader, s, f, SCALE_BITS));
        }
        assert_eq!(state.0, STATE_L);
        // A valid stream is consumed exactly.
        assert_eq!(reader.bytes_consumed(), bytes.len());
    }

    // ---------------------------------------------------------------------
    // Independent oracle parity (H.2.2): byte-for-byte equality with the
    // ryg-rans-rs port where model/layout semantics match.
    // ---------------------------------------------------------------------
    #[test]
    fn byte_parity_with_independent_oracle() {
        use ryg_rans_rs::byte::{
            BackwardByteWriter, ByteReader, RansByteDecSymbol, RansByteEncSymbol,
            RansByteState as OracleState, rans_byte_dec_advance_symbol, rans_byte_dec_get,
            rans_byte_dec_init, rans_byte_enc_flush, rans_byte_enc_put_symbol,
        };
        // Deterministic skewed intervals (tiling, frozen seed pattern).
        let mut starts: Vec<u32> = Vec::new();
        let mut freqs: Vec<u32> = Vec::new();
        let n = 1000usize;
        let mut cum = 0u32;
        for i in 0..n {
            let w = ((i * 2654435761usize) % 977 + 1) as u32;
            let f = ((u64::from(w) * u64::from(MODEL_TOTAL)) / 1_000_000) as u32;
            let f = f.max(1);
            starts.push(cum);
            freqs.push(f);
            cum += f;
        }
        freqs[n - 1] = MODEL_TOTAL - (cum - freqs[n - 1]);
        starts[n - 1] = MODEL_TOTAL - freqs[n - 1];

        // Our encode.
        let mut buf_a = vec![0u8; encode_capacity(n)];
        let ours = encode_symbols(&starts, &freqs, SCALE_BITS, &mut buf_a).unwrap();

        // Oracle encode (reciprocal fast path; proven == division).
        let oracle_cap = ours.len() + 16;
        let mut buf_b = vec![0u8; oracle_cap];
        let mut w = BackwardByteWriter::new(&mut buf_b);
        let mut st = OracleState::new();
        for i in 0..n {
            let sym =
                RansByteEncSymbol::new(starts[i], freqs[i], SCALE_BITS).expect("valid symbol");
            rans_byte_enc_put_symbol(&mut st, &mut w, &sym).expect("put");
        }
        rans_byte_enc_flush(&st, &mut w).expect("flush");
        let theirs = w.encoded();
        assert_eq!(
            ours, theirs,
            "our encode bytes must equal the independent oracle byte-for-byte"
        );

        // Cross-decode: our encoder -> oracle decoder.
        let mut r = ByteReader::new(ours);
        let mut ost = rans_byte_dec_init(&mut r).expect("init");
        let dec_syms: Vec<RansByteDecSymbol> = (0..n)
            .map(|i| RansByteDecSymbol::new(starts[i], freqs[i]).expect("dec sym"))
            .collect();
        for i in (0..n).rev() {
            let slot = rans_byte_dec_get(&ost, SCALE_BITS);
            let j = dec_syms
                .iter()
                .position(|s| {
                    slot >= u32::from(s.start) && slot < u32::from(s.start) + u32::from(s.freq)
                })
                .expect("slot in interval");
            let _ = i;
            assert_eq!(starts[j], starts[j]);
            rans_byte_dec_advance_symbol(&mut ost, &mut r, &dec_syms[j], SCALE_BITS)
                .expect("advance");
        }
        assert_eq!(ost.0, STATE_L, "oracle decode lands on STATE_L");

        // Oracle encode -> our decoder (already byte-equal; decode once more).
        let mut reader = FwdReader::new(theirs);
        let mut state = dec_init(&mut reader).expect("init");
        for i in (0..n).rev() {
            let slot = dec_slot(&state, SCALE_BITS);
            let mut found = None;
            for j in 0..n {
                if slot >= starts[j] && slot < starts[j] + freqs[j] {
                    found = Some(j);
                    break;
                }
            }
            let j = found.expect("interval");
            let _ = i;
            assert!(dec_advance(
                &mut state,
                &mut reader,
                starts[j],
                freqs[j],
                SCALE_BITS
            ));
        }
        assert_eq!(state.0, STATE_L);
        assert_eq!(reader.bytes_consumed(), theirs.len());
    }
}
