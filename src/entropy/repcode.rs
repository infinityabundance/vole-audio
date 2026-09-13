//! Phase 2B — Zstd-style recent-value codes and baseline/extra-bit integer
//! coding for metadata streams.
//!
//! Small metadata sequences (per-segment frame counts, predictor orders,
//! precision/shift selectors, residual-codec ids) are repetitive but currently
//! stored as fixed-width little-endian integers. Two independent Zstd ideas
//! apply directly:
//!
//! * a **recent-value cache** ([`encode_repcode`]) — the three most recently
//!   used values cost two bits to reference, and a fresh value costs a fallback
//!   delta from the cache head;
//! * **baseline + extra bits** ([`encode_baseline`]) — a per-stream minimum
//!   baseline, then unsigned varint deltas, so values clustered near the
//!   baseline cost a few bits each.
//!
//! Both are exact and decoder-visible. Neither is a semantic transform: they are
//! alternative serializations of the same integer sequence.

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

struct BitWriter {
    out: Vec<u8>,
    cur: u8,
    used: u8,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter {
            out: Vec::new(),
            cur: 0,
            used: 0,
        }
    }
    fn bit(&mut self, b: bool) {
        if b {
            self.cur |= 1 << (7 - self.used);
        }
        self.used += 1;
        if self.used == 8 {
            self.out.push(self.cur);
            self.cur = 0;
            self.used = 0;
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.used != 0 {
            self.out.push(self.cur);
        }
        self.out
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    bit: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        BitReader { bytes, bit: 0 }
    }
    fn read_bit(&mut self) -> Result<bool> {
        let byte = self.bit >> 3;
        if byte >= self.bytes.len() {
            return Err(Error::malformed("repcode bit stream is truncated"));
        }
        let b = (self.bytes[byte] >> (7 - (self.bit & 7))) & 1;
        self.bit += 1;
        Ok(b == 1)
    }
}

/// Bit-packed stream: `count` values, each a 2-bit code (`00`/`01`/`10` cache
/// hit, `11` fallback), then the zigzag-varint fallbacks.
fn pack(codes: &[u8], fallbacks: &[u64], count: usize) -> Vec<u8> {
    let mut w = BitWriter::new();
    for &c in codes {
        w.bit(c & 2 != 0);
        w.bit(c & 1 != 0);
    }
    let mut out = w.finish();
    out.extend_from_slice(&(fallbacks.len() as u32).to_le_bytes());
    for &v in fallbacks {
        put_uvarint(&mut out, v);
    }
    let _ = count;
    out
}

/// Encode a value sequence with a three-entry recent-value cache.
///
/// A value equal to cache slot 0/1/2 costs `00`/`01`/`10` (and moves to the
/// front); anything else costs `11` plus a zigzag delta from the cache head and
/// becomes the new head.
pub fn encode_repcode(values: &[u64]) -> Vec<u8> {
    let mut cache: [u64; 3] = [0, 0, 0];
    let mut codes: Vec<u8> = Vec::with_capacity(values.len());
    let mut fallbacks: Vec<u64> = Vec::new();
    for &v in values {
        let hit = cache.iter().position(|&c| c == v);
        match hit {
            Some(0) => codes.push(0),
            Some(1) => {
                cache.swap(0, 1);
                codes.push(1);
            }
            Some(2) => {
                let t = cache[2];
                cache[2] = cache[1];
                cache[1] = cache[0];
                cache[0] = t;
                codes.push(2);
            }
            _ => {
                let d = (v as i64).wrapping_sub(cache[0] as i64);
                fallbacks.push(((d << 1) ^ (d >> 63)) as u64);
                cache[2] = cache[1];
                cache[1] = cache[0];
                cache[0] = v;
                codes.push(3);
            }
        }
    }
    let mut out = Vec::new();
    out.extend_from_slice(&(values.len() as u32).to_le_bytes());
    out.extend_from_slice(&pack(&codes, &fallbacks, values.len()));
    out
}

/// Decode [`encode_repcode`] output.
pub fn decode_repcode(bytes: &[u8]) -> Result<Vec<u64>> {
    if bytes.len() < 8 {
        return Err(Error::malformed("repcode stream is truncated"));
    }
    let count = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let bit_bytes = (count * 2).div_ceil(8);
    if bytes.len() < 4 + bit_bytes + 4 {
        return Err(Error::malformed("repcode stream is truncated"));
    }
    // First pass: read every 2-bit code (the bit region has a fixed length).
    let mut r = BitReader::new(&bytes[4..4 + bit_bytes]);
    let mut codes: Vec<u8> = Vec::with_capacity(count);
    for _ in 0..count {
        let hi = r.read_bit()?;
        let lo = r.read_bit()?;
        codes.push(u8::from(hi) * 2 + u8::from(lo));
    }
    // Second pass: replay the cache in order, pulling fallback varints from the
    // tail as they are reached (the cache state at each position matters).
    let tail = &bytes[4 + bit_bytes..];
    let declared = u32::from_le_bytes(tail[..4].try_into().unwrap()) as usize;
    let fallbacks = codes.iter().filter(|&&c| c == 3).count();
    if declared != fallbacks {
        return Err(Error::malformed("repcode fallback count mismatch"));
    }
    let mut cache: [u64; 3] = [0, 0, 0];
    let mut out: Vec<u64> = Vec::with_capacity(count);
    let mut pos = 4usize;
    for code in codes {
        match code {
            0 => out.push(cache[0]),
            1 => {
                cache.swap(0, 1);
                out.push(cache[0]);
            }
            2 => {
                let t = cache[2];
                cache[2] = cache[1];
                cache[1] = cache[0];
                cache[0] = t;
                out.push(cache[0]);
            }
            _ => {
                let (u, used) = read_uvarint(tail, pos)?;
                pos += used;
                let d = ((u >> 1) as i64) ^ -((u & 1) as i64);
                let v = (cache[0] as i64).wrapping_add(d) as u64;
                cache[2] = cache[1];
                cache[1] = cache[0];
                cache[0] = v;
                out.push(v);
            }
        }
    }
    if pos != tail.len() {
        return Err(Error::malformed("repcode stream has trailing bytes"));
    }
    Ok(out)
}

fn read_uvarint(bytes: &[u8], mut pos: usize) -> Result<(u64, usize)> {
    let start = pos;
    let mut v = 0u64;
    let mut shift = 0u32;
    loop {
        let b = *bytes
            .get(pos)
            .ok_or_else(|| Error::malformed("repcode varint is truncated"))?;
        pos += 1;
        v |= u64::from(b & 0x7F) << shift;
        if b & 0x80 == 0 {
            return Ok((v, pos - start));
        }
        shift += 7;
        if shift >= 64 {
            return Err(Error::malformed("repcode varint is too long"));
        }
    }
}

/// Bytes of the baseline + varint-delta encoding (`baseline = min(values)`).
pub fn encode_baseline(values: &[u64]) -> Vec<u8> {
    let baseline = values.iter().copied().min().unwrap_or(0);
    let mut out = Vec::new();
    out.extend_from_slice(&(values.len() as u32).to_le_bytes());
    out.extend_from_slice(&baseline.to_le_bytes());
    for &v in values {
        put_uvarint(&mut out, v.saturating_sub(baseline));
    }
    out
}

/// Decode [`encode_baseline`] output.
pub fn decode_baseline(bytes: &[u8]) -> Result<Vec<u64>> {
    if bytes.len() < 12 {
        return Err(Error::malformed("baseline stream is truncated"));
    }
    let count = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let baseline = u64::from_le_bytes(bytes[4..12].try_into().unwrap());
    let mut pos = 12usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let (d, used) = read_uvarint(bytes, pos)?;
        pos += used;
        out.push(
            baseline
                .checked_add(d)
                .ok_or_else(|| Error::malformed("baseline value overflows"))?,
        );
    }
    if pos != bytes.len() {
        return Err(Error::malformed("baseline stream has trailing bytes"));
    }
    Ok(out)
}

/// Fixed-width little-endian `u32` byte count (the existing spelling).
pub fn raw_u32_bytes(values: &[u64]) -> u64 {
    4 + values.len() as u64 * 4
}

/// Bytes of the repcode encoding.
pub fn repcode_bytes(values: &[u64]) -> u64 {
    encode_repcode(values).len() as u64
}

/// Bytes of the baseline encoding.
pub fn baseline_bytes(values: &[u64]) -> u64 {
    encode_baseline(values).len() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repcode_round_trips() {
        let values: Vec<u64> = vec![4096, 4096, 4096, 512, 4096, 4096, 512, 512, 7, 4096];
        assert_eq!(decode_repcode(&encode_repcode(&values)).unwrap(), values);
    }

    #[test]
    fn repcode_beats_raw_on_a_repetitive_stream() {
        let values: Vec<u64> = (0..256)
            .map(|i| if i % 5 == 0 { 7 } else { 4096 })
            .collect();
        assert!(repcode_bytes(&values) < raw_u32_bytes(&values));
    }

    #[test]
    fn baseline_is_exact_and_never_larger_than_raw_for_small_values() {
        let values: Vec<u64> = (0..200).map(|i| 4096 + (i % 3)).collect();
        assert_eq!(decode_baseline(&encode_baseline(&values)).unwrap(), values);
        assert!(baseline_bytes(&values) < raw_u32_bytes(&values));
    }
}
