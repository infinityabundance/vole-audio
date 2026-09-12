//! `PageBatchSIMD` — two-state rANS and host-side page batching (Phase 6
//! mechanism 12).
//!
//! A canonical **page** is an independent rANS-coded unit with **two implicit
//! alternating states**: symbol `i` is coded by state `i & 1`, and the two
//! states renormalize into one shared byte stream. Two states recover most of
//! the serial dependency of a single-state stream at no format cost, and — more
//! importantly here — they make the page's decode step short enough to batch.
//!
//! **No ISA width enters the format.** A page is self-contained, so decoding 4,
//! 8 or 16 pages at once is a *host runtime* choice: the same bytes decode
//! identically sequentially or batched, and on a machine with different vector
//! width the format is unchanged. This module implements the batching as a
//! lockstep lane loop (the semantics a SIMD kernel must preserve) so the
//! equivalence is testable on any host.
//!
//! This is a throughput mechanism: it makes no compression-ratio claim.

use crate::entropy::rans::{
    BackSink, FwdReader, MODEL_TOTAL, RansState, SCALE_BITS, dec_advance, dec_init, dec_slot,
    enc_flush, enc_put, encode_capacity,
};
use crate::error::{Error, Result};

/// Maximum lanes a batch may request.
pub const PAGE_LANES_MAX: usize = 16;

/// Lane counts a batch surface may use (`no ISA width in the format`).
pub const PAGE_LANE_LADDER: [usize; 3] = [4, 8, 16];

fn starts_of(freqs: &[u32]) -> Result<Vec<u32>> {
    if freqs.contains(&0) {
        return Err(Error::malformed("page table has a zero frequency"));
    }
    let mut starts = Vec::with_capacity(freqs.len());
    let mut acc = 0u32;
    for &f in freqs {
        starts.push(acc);
        acc = acc
            .checked_add(f)
            .ok_or_else(|| Error::limit("page table overflows"))?;
    }
    if acc != MODEL_TOTAL {
        return Err(Error::malformed("page table does not sum to the total"));
    }
    Ok(starts)
}

fn symbol_for(starts: &[u32], freqs: &[u32], slot: u32) -> Result<usize> {
    for s in 0..freqs.len() {
        let st = starts[s];
        if slot >= st && slot < st + freqs[s] {
            return Ok(s);
        }
    }
    Err(Error::malformed("page slot has no symbol"))
}

/// Encode one canonical page with two alternating rANS states.
pub fn encode_page(symbols: &[usize], freqs: &[u32]) -> Result<Vec<u8>> {
    let starts = starts_of(freqs)?;
    let mut buf = vec![0u8; encode_capacity(symbols.len()).max(16)];
    let written = {
        let mut sink = BackSink::new(&mut buf);
        let mut s0 = RansState::new();
        let mut s1 = RansState::new();
        let mut ok = true;
        for (i, &sym) in symbols.iter().enumerate().rev() {
            let f = *freqs
                .get(sym)
                .ok_or_else(|| Error::malformed("page symbol out of range"))?;
            let state = if i & 1 == 0 { &mut s0 } else { &mut s1 };
            if !enc_put(state, &mut sink, starts[sym], f, SCALE_BITS) {
                ok = false;
                break;
            }
        }
        // BackSink writes backwards, so flushing s1 first leaves `[s0][s1]` at
        // the front, matching the decoder's read order.
        if ok && enc_flush(&s1, &mut sink) && enc_flush(&s0, &mut sink) {
            sink.bytes_written()
        } else {
            return Err(Error::limit("page sink exhausted"));
        }
    };
    Ok(buf[buf.len() - written..].to_vec())
}

/// Decode one canonical page.
pub fn decode_page(bytes: &[u8], n: usize, freqs: &[u32]) -> Result<Vec<usize>> {
    let starts = starts_of(freqs)?;
    let mut reader = FwdReader::new(bytes);
    let mut s0 =
        dec_init(&mut reader).ok_or_else(|| Error::malformed("page stream is truncated"))?;
    let mut s1 =
        dec_init(&mut reader).ok_or_else(|| Error::malformed("page stream is truncated"))?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let state = if i & 1 == 0 { &mut s0 } else { &mut s1 };
        let slot = dec_slot(state, SCALE_BITS);
        let sym = symbol_for(&starts, freqs, slot)?;
        if !dec_advance(state, &mut reader, starts[sym], freqs[sym], SCALE_BITS) {
            return Err(Error::malformed("page stream is truncated"));
        }
        out.push(sym);
    }
    Ok(out)
}

/// Decode pages in batches of `lanes`, advancing each lane in lockstep.
///
/// This is the exact semantics a SIMD page-batch kernel must preserve; the
/// result must equal decoding each page with [`decode_page`].
pub fn decode_batch(
    pages: &[(&[u8], usize)],
    freqs: &[u32],
    lanes: usize,
) -> Result<Vec<Vec<usize>>> {
    if lanes == 0 || lanes > PAGE_LANES_MAX {
        return Err(Error::malformed("page batch lane count out of range"));
    }
    let starts = starts_of(freqs)?;
    let mut out: Vec<Vec<usize>> = Vec::with_capacity(pages.len());
    for chunk in pages.chunks(lanes) {
        let mut state: Vec<(FwdReader<'_>, RansState, RansState, usize)> =
            Vec::with_capacity(chunk.len());
        for &(bytes, n) in chunk {
            let mut reader = FwdReader::new(bytes);
            let s0 = dec_init(&mut reader)
                .ok_or_else(|| Error::malformed("page stream is truncated"))?;
            let s1 = dec_init(&mut reader)
                .ok_or_else(|| Error::malformed("page stream is truncated"))?;
            state.push((reader, s0, s1, n));
        }
        let mut outs: Vec<Vec<usize>> = chunk.iter().map(|&(_, n)| Vec::with_capacity(n)).collect();
        let max_n = chunk.iter().map(|&(_, n)| n).max().unwrap_or(0);
        for i in 0..max_n {
            for (li, lane) in state.iter_mut().enumerate() {
                let (reader, s0, s1, n) = lane;
                if i >= *n {
                    continue;
                }
                let st = if i & 1 == 0 { &mut *s0 } else { &mut *s1 };
                let slot = dec_slot(st, SCALE_BITS);
                let sym = symbol_for(&starts, freqs, slot)?;
                if !dec_advance(st, reader, starts[sym], freqs[sym], SCALE_BITS) {
                    return Err(Error::malformed("page stream is truncated"));
                }
                outs[li].push(sym);
            }
        }
        out.extend(outs);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::recoil::normalize_histogram;

    fn table() -> Vec<u32> {
        normalize_histogram(&[50, 25, 12, 6, 4, 2, 1, 1, 1, 1])
    }

    fn page(seed: usize, n: usize) -> Vec<usize> {
        (0..n).map(|i| (i * 7 + i / 3 + seed) % 10).collect()
    }

    #[test]
    fn two_state_page_round_trips() {
        let f = table();
        for n in [1usize, 2, 3, 511, 512, 513, 2000] {
            let syms = page(n, n);
            let bytes = encode_page(&syms, &f).unwrap();
            assert_eq!(decode_page(&bytes, n, &f).unwrap(), syms, "n={n}");
        }
    }

    #[test]
    fn every_lane_count_equals_sequential_decode() {
        let f = table();
        let pages: Vec<Vec<usize>> = (0..13).map(|k| page(k, 200 + k * 7)).collect();
        let encoded: Vec<Vec<u8>> = pages.iter().map(|p| encode_page(p, &f).unwrap()).collect();
        let sequential: Vec<Vec<usize>> = encoded
            .iter()
            .zip(&pages)
            .map(|(b, p)| decode_page(b, p.len(), &f).unwrap())
            .collect();
        let refs: Vec<(&[u8], usize)> = encoded
            .iter()
            .zip(&pages)
            .map(|(b, p)| (b.as_slice(), p.len()))
            .collect();
        for lanes in PAGE_LANE_LADDER {
            let batched = decode_batch(&refs, &f, lanes).unwrap();
            assert_eq!(batched, sequential, "lanes={lanes}");
        }
    }

    #[test]
    fn lane_count_out_of_range_is_rejected() {
        let f = table();
        let bytes = encode_page(&page(1, 8), &f).unwrap();
        let refs = [(&bytes[..], 8usize)];
        assert!(decode_batch(&refs, &f, 0).is_err());
        assert!(decode_batch(&refs, &f, PAGE_LANES_MAX + 1).is_err());
    }
}
