//! `RecoilCheckpoints` — decoder-chosen rANS parallelism (Phase 6 mechanism 11).
//!
//! A single rANS stream is inherently sequential: the decoder must replay it
//! from the front. `RecoilCheckpoints` spends a small, explicit amount of
//! metadata to remove that constraint. At chosen symbol boundaries the encoder
//! records the decoder's `(state, rANS byte position)`; a decoder can then start
//! at any checkpoint and process a disjoint range independently, with no change
//! to the coded symbols themselves.
//!
//! This is an honest size/speed tradeoff: the stream is neither larger nor
//! smaller in its symbols, but the checkpoint index costs bytes. The payoff is
//! scalable decode parallelism chosen by the *decoder* rather than baked into
//! the format.
//!
//! The module operates on a static frequency table (the same class of model as
//! the archive/residual static rANS): both encoder and decoder are given the
//! table, and checkpoints capture only the rANS machine state and reader
//! position.

use crate::entropy::rans::{
    BackSink, FwdReader, MODEL_TOTAL, RansState, SCALE_BITS, dec_advance, dec_init, dec_slot,
    enc_flush, enc_put, encode_capacity,
};
use crate::error::{Error, Result};

/// One rANS checkpoint: the decoder state and reader position at a symbol
/// boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecoilCheckpoint {
    /// Index of the first symbol a decoder would produce from this checkpoint.
    pub symbol_index: u64,
    /// The rANS decoder state at that boundary.
    pub state: u32,
    /// The rANS reader position (bytes consumed) at that boundary.
    pub rans_byte_pos: u32,
}

/// Normalize a histogram into a static frequency table summing exactly
/// [`MODEL_TOTAL`], with every symbol at frequency at least one.
pub fn normalize_histogram(counts: &[u64]) -> Vec<u32> {
    let k = counts.len();
    debug_assert!(k >= 1 && (k as u32) < MODEL_TOTAL);
    let total: u64 = counts.iter().sum::<u64>().max(1);
    let budget = u64::from(MODEL_TOTAL) - k as u64;
    let mut f = vec![1u32; k];
    let mut used = 0u64;
    for (slot, &c) in f.iter_mut().zip(counts) {
        let add = c.saturating_mul(budget) / total;
        *slot += add as u32;
        used += add;
    }
    let mut rem = budget.saturating_sub(used);
    // Distribute the remainder by largest fractional part (ties by index).
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| {
        let fa = counts[a].saturating_mul(budget) % total;
        let fb = counts[b].saturating_mul(budget) % total;
        fb.cmp(&fa).then(a.cmp(&b))
    });
    let mut idx = 0usize;
    while rem > 0 {
        f[order[idx % k]] += 1;
        rem -= 1;
        idx += 1;
    }
    f
}

fn starts_of(freqs: &[u32]) -> Result<Vec<u32>> {
    if freqs.contains(&0) {
        return Err(Error::malformed("recoil table has a zero frequency"));
    }
    let mut starts = Vec::with_capacity(freqs.len());
    let mut acc = 0u32;
    for &f in freqs {
        starts.push(acc);
        acc = acc
            .checked_add(f)
            .ok_or_else(|| Error::limit("recoil table overflows"))?;
    }
    if acc != MODEL_TOTAL {
        return Err(Error::malformed("recoil table does not sum to the total"));
    }
    Ok(starts)
}

/// Encode a symbol sequence under a static table.
pub fn encode(symbols: &[usize], freqs: &[u32]) -> Result<Vec<u8>> {
    let starts = starts_of(freqs)?;
    let mut buf = vec![0u8; encode_capacity(symbols.len()).max(16)];
    let written = {
        let mut sink = BackSink::new(&mut buf);
        let mut state = RansState::new();
        let mut ok = true;
        for &s in symbols.iter().rev() {
            let f = *freqs
                .get(s)
                .ok_or_else(|| Error::malformed("recoil symbol out of range"))?;
            if !enc_put(&mut state, &mut sink, starts[s], f, SCALE_BITS) {
                ok = false;
                break;
            }
        }
        if ok && enc_flush(&state, &mut sink) {
            sink.bytes_written()
        } else {
            return Err(Error::limit("recoil sink exhausted"));
        }
    };
    Ok(buf[buf.len() - written..].to_vec())
}

fn symbol_for(starts: &[u32], freqs: &[u32], slot: u32) -> Result<usize> {
    for s in 0..freqs.len() {
        let st = starts[s];
        if slot >= st && slot < st + freqs[s] {
            return Ok(s);
        }
    }
    Err(Error::malformed("recoil slot has no symbol"))
}

/// Decode `n` symbols from the front of a stream.
pub fn decode(bytes: &[u8], n: usize, freqs: &[u32]) -> Result<Vec<usize>> {
    decode_segment(bytes, n, 0, n, freqs, None)
}

/// Decode `count` symbols starting at symbol index `start`, optionally from a
/// checkpoint that begins exactly there.
pub fn decode_segment(
    bytes: &[u8],
    n: usize,
    start: usize,
    count: usize,
    freqs: &[u32],
    from: Option<&RecoilCheckpoint>,
) -> Result<Vec<usize>> {
    let starts = starts_of(freqs)?;
    let end = start
        .checked_add(count)
        .ok_or_else(|| Error::limit("recoil range overflows"))?;
    if end > n {
        return Err(Error::malformed("recoil range exceeds the stream"));
    }
    let (mut reader, mut state) = match from {
        Some(cp) => {
            if cp.symbol_index != start as u64 {
                return Err(Error::malformed("recoil checkpoint index mismatch"));
            }
            let at = cp.rans_byte_pos as usize;
            if at > bytes.len() {
                return Err(Error::malformed(
                    "recoil checkpoint position is past the stream",
                ));
            }
            (FwdReader::new(&bytes[at..]), RansState(cp.state))
        }
        None => {
            if start != 0 {
                return Err(Error::malformed(
                    "recoil mid-stream decode needs a checkpoint",
                ));
            }
            let mut reader = FwdReader::new(bytes);
            let state = dec_init(&mut reader)
                .ok_or_else(|| Error::malformed("recoil stream is truncated"))?;
            (reader, state)
        }
    };
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let slot = dec_slot(&state, SCALE_BITS);
        let s = symbol_for(&starts, freqs, slot)?;
        if !dec_advance(&mut state, &mut reader, starts[s], freqs[s], SCALE_BITS) {
            return Err(Error::malformed("recoil stream is truncated"));
        }
        out.push(s);
    }
    Ok(out)
}

/// Build the checkpoint index at a fixed symbol interval (the first checkpoint
/// is at index 0).
pub fn checkpoints(
    bytes: &[u8],
    n: usize,
    freqs: &[u32],
    interval: usize,
) -> Result<Vec<RecoilCheckpoint>> {
    if interval == 0 {
        return Err(Error::malformed(
            "recoil checkpoint interval must be nonzero",
        ));
    }
    let starts = starts_of(freqs)?;
    let mut reader = FwdReader::new(bytes);
    let mut state =
        dec_init(&mut reader).ok_or_else(|| Error::malformed("recoil stream is truncated"))?;
    let mut out = Vec::new();
    for i in 0..n {
        if i % interval == 0 {
            out.push(RecoilCheckpoint {
                symbol_index: i as u64,
                state: state.0,
                rans_byte_pos: reader.bytes_consumed() as u32,
            });
        }
        let slot = dec_slot(&state, SCALE_BITS);
        let s = symbol_for(&starts, freqs, slot)?;
        if !dec_advance(&mut state, &mut reader, starts[s], freqs[s], SCALE_BITS) {
            return Err(Error::malformed("recoil stream is truncated"));
        }
    }
    Ok(out)
}

/// Serialize a checkpoint index: `count || [varint(index) || u32 state ||
/// varint(byte_pos)]`.
pub fn encode_checkpoints(cps: &[RecoilCheckpoint]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + cps.len() * 10);
    out.extend_from_slice(&(cps.len() as u32).to_le_bytes());
    for cp in cps {
        put_uvarint(&mut out, cp.symbol_index);
        out.extend_from_slice(&cp.state.to_le_bytes());
        put_uvarint(&mut out, u64::from(cp.rans_byte_pos));
    }
    out
}

fn put_uvarint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            break;
        }
        out.push(byte | 0x80);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symbols() -> Vec<usize> {
        (0..5000usize).map(|i| (i * 7 + i / 11) % 9).collect()
    }

    fn table() -> Vec<u32> {
        normalize_histogram(&[40, 30, 20, 15, 10, 8, 5, 3, 2])
    }

    #[test]
    fn histogram_normalizes_to_the_frozen_total_with_min_one() {
        let f = normalize_histogram(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
        assert_eq!(
            f.iter().map(|&x| u64::from(x)).sum::<u64>(),
            u64::from(MODEL_TOTAL)
        );
        assert!(f.iter().all(|&x| x >= 1));
    }

    #[test]
    fn static_stream_round_trips() {
        let f = table();
        let bytes = encode(&symbols(), &f).unwrap();
        assert_eq!(decode(&bytes, 5000, &f).unwrap(), symbols());
    }

    #[test]
    fn checkpoint_segments_reassemble_the_full_decode() {
        let f = table();
        let syms = symbols();
        let bytes = encode(&syms, &f).unwrap();
        let full = decode(&bytes, syms.len(), &f).unwrap();
        let cps = checkpoints(&bytes, syms.len(), &f, 512).unwrap();
        assert_eq!(cps[0].symbol_index, 0);
        assert_eq!(cps.len(), syms.len().div_ceil(512));
        let mut reassembled = Vec::with_capacity(syms.len());
        for (w, cp) in cps.iter().enumerate() {
            let start = w * 512;
            let count = 512.min(syms.len() - start);
            let seg = decode_segment(&bytes, syms.len(), start, count, &f, Some(cp)).unwrap();
            assert_eq!(seg, full[start..start + count], "worker {w}");
            reassembled.extend_from_slice(&seg);
        }
        assert_eq!(reassembled, full);
    }

    #[test]
    fn checkpoint_serialization_is_bounded() {
        let f = table();
        let syms = symbols();
        let bytes = encode(&syms, &f).unwrap();
        let cps = checkpoints(&bytes, syms.len(), &f, 256).unwrap();
        let index = encode_checkpoints(&cps);
        // Ten bytes per checkpoint is the varint+u32 upper envelope for these
        // small indices; the index must stay a small fraction of the stream.
        assert!(index.len() <= 4 + cps.len() * 12);
        assert!(index.len() < bytes.len());
    }
}
