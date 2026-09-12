//! Phase 2A mechanism 1 — differential predictor-parameter coding and
//! progressive-order restart prediction.
//!
//! ## Differential predictor-parameter coding
//!
//! Learned predictors are fitted per block or per segment. Adjacent blocks of a
//! stationary source produce *correlated* coefficient vectors, so transmitting
//! each vector independently pays for information the previous vector already
//! carried. [`encode_vectors`] stores the first vector absolutely and every
//! later vector as a per-position signed delta (zigzag varint), which is an
//! exact, decoder-visible transform of the parameter sequence.
//!
//! ## Progressive-order restart prediction
//!
//! MPEG-4 ALS avoids transmitting `K` warm-up samples at a restart by using
//! order 1 on sample 2, order 2 on sample 3, and so on. VOLE predictors reset
//! closed-loop with a **zero-initialized** history, so a tap whose history index
//! is negative contributes zero whether or not it is applied. Progressive order
//! is therefore *provably identical* to full order here — the mechanism is
//! structurally inapplicable, and [`block_residual_abs`] makes that testable
//! rather than merely asserted.

use crate::error::{Error, Result};

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

fn uvarint_len(mut v: u64) -> u64 {
    let mut n = 1u64;
    while v >= 0x80 {
        v >>= 7;
        n += 1;
    }
    n
}

fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

fn unzigzag(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

/// Encode a sequence of (possibly different-width) parameter vectors with
/// cross-vector per-position deltas. Exact inverse of [`decode_vectors`].
pub fn encode_vectors(vectors: &[Vec<i64>]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(vectors.len() as u32).to_le_bytes());
    let mut prev: Vec<i64> = Vec::new();
    for v in vectors {
        out.extend_from_slice(&(v.len() as u32).to_le_bytes());
        for (j, &x) in v.iter().enumerate() {
            let p = prev.get(j).copied().unwrap_or(0);
            put_uvarint(&mut out, zigzag(x - p));
        }
        prev = v.clone();
    }
    out
}

/// Decode [`encode_vectors`] output.
pub fn decode_vectors(bytes: &[u8]) -> Result<Vec<Vec<i64>>> {
    let mut pos = 0usize;
    let take = |pos: &mut usize, n: usize| -> Result<&[u8]> {
        let end = pos
            .checked_add(n)
            .ok_or_else(|| Error::limit("param-delta range overflows"))?;
        if end > bytes.len() {
            return Err(Error::malformed("param-delta stream is truncated"));
        }
        let s = &bytes[*pos..end];
        *pos = end;
        Ok(s)
    };
    let count = u32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap()) as usize;
    if count > bytes.len() {
        return Err(Error::limit("param-delta vector count exceeds the stream"));
    }
    let mut prev: Vec<i64> = Vec::new();
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let width = u32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap()) as usize;
        if width > 1 << 20 {
            return Err(Error::limit("param-delta vector width out of range"));
        }
        let mut v = Vec::with_capacity(width);
        for j in 0..width {
            let d = unzigzag(uvarint(bytes, &mut pos)?);
            let p = prev.get(j).copied().unwrap_or(0);
            let x = p
                .checked_add(d)
                .ok_or_else(|| Error::malformed("param-delta value overflows"))?;
            v.push(x);
        }
        prev = v.clone();
        out.push(v);
    }
    if pos != bytes.len() {
        return Err(Error::malformed("param-delta stream has trailing bytes"));
    }
    Ok(out)
}

fn uvarint(bytes: &[u8], pos: &mut usize) -> Result<u64> {
    let mut v = 0u64;
    let mut shift = 0u32;
    loop {
        let b = *bytes
            .get(*pos)
            .ok_or_else(|| Error::malformed("param-delta stream is truncated"))?;
        *pos += 1;
        v |= u64::from(b & 0x7F) << shift;
        if b & 0x80 == 0 {
            return Ok(v);
        }
        shift += 7;
        if shift >= 64 {
            return Err(Error::malformed("param-delta varint is too long"));
        }
    }
}

/// Bytes of the differential encoding.
pub fn delta_bytes(vectors: &[Vec<i64>]) -> u64 {
    encode_vectors(vectors).len() as u64
}

/// Bytes when every vector is stored independently as zigzag varints (the
/// baseline the mechanism is measured against).
pub fn independent_varint_bytes(vectors: &[Vec<i64>]) -> u64 {
    let mut n = 4u64;
    for v in vectors {
        n += 4;
        for &x in v {
            n += uvarint_len(zigzag(x));
        }
    }
    n
}

/// Bytes when every vector is stored as raw little-endian `i16` (the
/// `COEFF_RAW_I16` syntax).
pub fn raw_i16_bytes(vectors: &[Vec<i64>]) -> u64 {
    let mut n = 4u64;
    for v in vectors {
        n += 4 + v.len() as u64 * 2;
    }
    n
}

/// Accumulated absolute residual of a block-local order-`P` predictor over
/// `source`. When `progressive` is set the tap count is limited to the number of
/// in-block samples seen so far (the ALS restart rule).
pub fn block_residual_abs(
    source: &[i32],
    coeffs: &[i32],
    shift: u32,
    block_frames: usize,
    progressive: bool,
) -> i128 {
    if block_frames == 0 || coeffs.is_empty() {
        return 0;
    }
    let p = coeffs.len();
    let mut acc = 0i128;
    let mut from = 0usize;
    while from < source.len() {
        let to = (from + block_frames).min(source.len());
        for t in from..to {
            let local = t - from;
            let taps = if progressive { p.min(local) } else { p };
            let mut s = 0i64;
            for (i, &c) in coeffs.iter().take(taps).enumerate() {
                // Block-local history: samples before `from` are zero.
                if let Some(idx) = t.checked_sub(1 + i)
                    && idx >= from
                {
                    s += i64::from(c) * i64::from(source[idx]);
                }
            }
            let pred = s >> shift;
            acc += i128::from((i64::from(source[t]) - pred).abs());
        }
        from = to;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_deltas_round_trip() {
        let vectors = vec![
            vec![100i64, -50, 3, 0, 7],
            vec![101, -52, 4, 0, 7],
            vec![98, -49],
            vec![],
        ];
        let bytes = encode_vectors(&vectors);
        assert_eq!(decode_vectors(&bytes).unwrap(), vectors);
    }

    #[test]
    fn delta_never_exceeds_independent_for_similar_vectors() {
        let vectors: Vec<Vec<i64>> = (0..16)
            .map(|b| (0..8).map(|k| 1000 + 5 * k - b).collect())
            .collect();
        assert!(delta_bytes(&vectors) < independent_varint_bytes(&vectors));
    }

    #[test]
    fn progressive_order_is_identical_on_a_zero_initialized_restart() {
        let source: Vec<i32> = (0..4096).map(|i| ((i * 89) % 15013) - 7000).collect();
        let coeffs = vec![1900i32, -820, 210, -40];
        for block in [512usize, 1024, 4096] {
            for shift in [10u32, 12, 14] {
                let full = block_residual_abs(&source, &coeffs, shift, block, false);
                let progressive = block_residual_abs(&source, &coeffs, shift, block, true);
                assert_eq!(full, progressive, "block={block} shift={shift}");
            }
        }
    }
}
