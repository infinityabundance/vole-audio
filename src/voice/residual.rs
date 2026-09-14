//! Compact residual coding for the voice hot path (7C.3).
//!
//! The crate's general residual family (`learned::residual_codec2`) is built for
//! object-scale blocks and every member writes an **8-byte length prefix**. That
//! is correct for a multi-kilobyte block and catastrophic for a 160-sample
//! voice frame: at 8 kbps with 10 ms frames the whole frame allowance is ten
//! bytes, so the prefix alone would consume it.
//!
//! The voice packet already carries an explicit `u16` residual length in its
//! frame record, so the length does not need to be repeated. This module
//! provides the compact codecs the voice path needs, and keeps the general
//! family available: any symbol stream for which the general codec wins on
//! *complete* bytes is still allowed to use it (`id ≥ 128`).
//!
//! ```text
//! id 0    RiceRun   : Elias-gamma zero runs + Golomb-Rice on zigzag values
//! id 1    Rice      : plain Golomb-Rice on zigzag values
//! id 2    Varint    : zigzag varints (wins on very short or very sparse frames)
//! id ≥128 general   : learned::residual_codec2, id = 128 + codec id
//! ```
//!
//! Nothing here is a second entropy subsystem: it is the compact framing the
//! streaming profile requires, and the general family remains reachable.

use crate::error::{Error, Kind, Result};
use crate::learned::residual_codec2 as rc2;

/// Zero-run + Golomb-Rice.
pub const RICE_RUN: u8 = 0;
/// Plain Golomb-Rice.
pub const RICE: u8 = 1;
/// Zigzag varints.
pub const VARINT: u8 = 2;
/// Offset added to a `learned::residual_codec2` id.
pub const V2_BASE: u8 = 128;
/// Rice parameters searched (log2 of the geometric parameter).
pub const K_MAX: u8 = 24;

// ---------------------------------------------------------------------------
// Bit I/O (MSB first, byte aligned at the end)
// ---------------------------------------------------------------------------

struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    bits: u32,
}

impl BitWriter {
    fn new() -> BitWriter {
        BitWriter {
            out: Vec::new(),
            acc: 0,
            bits: 0,
        }
    }
    fn bit(&mut self, b: bool) {
        self.acc = (self.acc << 1) | u32::from(b);
        self.bits += 1;
        if self.bits == 8 {
            self.out.push(self.acc as u8);
            self.acc = 0;
            self.bits = 0;
        }
    }
    fn bits(&mut self, v: u64, n: u32) {
        for i in (0..n).rev() {
            self.bit((v >> i) & 1 == 1);
        }
    }
    fn gamma(&mut self, x: u64) {
        debug_assert!(x >= 1);
        let l = 63 - x.leading_zeros();
        for _ in 0..l {
            self.bit(false);
        }
        self.bits(x, l + 1);
    }
    fn rice(&mut self, v: u64, k: u8) {
        let q = v >> k;
        for _ in 0..q {
            self.bit(true);
        }
        self.bit(false);
        if k > 0 {
            self.bits(v & ((1u64 << k) - 1), u32::from(k));
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.bits > 0 {
            self.acc <<= 8 - self.bits;
            self.out.push(self.acc as u8);
        }
        self.out
    }
}

struct BitReader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(b: &'a [u8]) -> BitReader<'a> {
        BitReader { b, pos: 0 }
    }
    fn bit(&mut self) -> Result<bool> {
        let byte = self.pos >> 3;
        if byte >= self.b.len() {
            return Err(Error::malformed("voice residual bits exhausted"));
        }
        let bit = (self.b[byte] >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Ok(bit == 1)
    }
    fn bits(&mut self, n: u32) -> Result<u64> {
        let mut v = 0u64;
        for _ in 0..n {
            v = (v << 1) | u64::from(self.bit()?);
        }
        Ok(v)
    }
    fn gamma(&mut self) -> Result<u64> {
        let mut l = 0u32;
        while !self.bit()? {
            l += 1;
            if l > 63 {
                return Err(Error::malformed("voice residual gamma code is unbounded"));
            }
        }
        if l == 0 {
            return Ok(1);
        }
        Ok((1u64 << l) | self.bits(l)?)
    }
    fn rice(&mut self, k: u8) -> Result<u64> {
        let mut q = 0u64;
        while self.bit()? {
            q += 1;
            if q > (1u64 << 40) {
                return Err(Error::malformed("voice residual unary code is unbounded"));
            }
        }
        let r = if k > 0 { self.bits(u32::from(k))? } else { 0 };
        Ok((q << k) | r)
    }
}

/// Zigzag map (sign interleave).
#[inline]
fn zigzag(v: i32) -> u64 {
    u64::from(((v << 1) ^ (v >> 31)) as u32)
}

/// Inverse of [`zigzag`].
#[inline]
fn unzigzag(u: u64) -> i32 {
    let x = u as u32;
    ((x >> 1) as i32) ^ -((x & 1) as i32)
}

// ---------------------------------------------------------------------------
// Encoders
// ---------------------------------------------------------------------------

/// The `k` minimising the plain Golomb-Rice cost of a mapped vector.
fn best_k(mapped: &[u64]) -> u8 {
    // Cost(k) = Σ (m >> k) + (k + 1)·n, so the optimum is near log2(mean).
    let n = mapped.len() as u64;
    if n == 0 {
        return 0;
    }
    let sum: u64 = mapped.iter().sum();
    let mean = (sum / n).max(1);
    let centre = 63 - u64::from(mean.leading_zeros());
    // The exact cost is not convex in the presence of large outliers, so
    // compare the neighbours.
    let cost = |k: u64| -> u64 { mapped.iter().map(|&m| (m >> k) + k + 1).sum::<u64>() };
    let lo = centre.saturating_sub(2);
    let hi = (centre + 2).min(u64::from(K_MAX));
    let lo = lo.min(hi);
    let mut k = lo;
    let mut best = cost(lo);
    for cand in lo + 1..=hi {
        let c = cost(cand);
        if c < best {
            best = c;
            k = cand;
        }
    }
    k as u8
}

fn encode_rice(symbols: &[i32]) -> Vec<u8> {
    let mapped: Vec<u64> = symbols.iter().map(|&v| zigzag(v)).collect();
    let k = best_k(&mapped);
    let mut w = BitWriter::new();
    w.bits(u64::from(k), 5);
    for &m in &mapped {
        w.rice(m, k);
    }
    w.finish()
}

fn encode_rice_run(symbols: &[i32]) -> Vec<u8> {
    let mapped: Vec<u64> = symbols.iter().map(|&v| zigzag(v)).collect();
    // The Rice parameter governs the nonzero values only, since the zero runs
    // are carried by the gamma code.
    let nonzero: Vec<u64> = mapped.iter().copied().filter(|&m| m > 0).collect();
    let k = best_k(&nonzero);
    let mut w = BitWriter::new();
    w.bits(u64::from(k), 5);
    let n = mapped.len();
    let mut i = 0usize;
    while i < n {
        let mut z = 0usize;
        while i + z < n && mapped[i + z] == 0 {
            z += 1;
        }
        w.gamma(z as u64 + 1);
        i += z;
        if i >= n {
            break;
        }
        w.rice(mapped[i] - 1, k);
        i += 1;
    }
    w.finish()
}

fn encode_varint(symbols: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(symbols.len());
    for &v in symbols {
        let mut u = zigzag(v);
        loop {
            let byte = (u & 0x7f) as u8;
            u >>= 7;
            if u == 0 {
                out.push(byte);
                break;
            }
            out.push(byte | 0x80);
        }
    }
    out
}

/// Encode a residual with the compact codecs and, when it wins on complete
/// bytes, with the crate's general family.
pub fn encode_best(symbols: &[i32], general: &[rc2::ResidualCodecV2]) -> Vec<u8> {
    let mut best: Option<Vec<u8>> = None;
    let mut consider = |id: u8, payload: Vec<u8>| {
        let mut bytes = Vec::with_capacity(payload.len() + 1);
        bytes.push(id);
        bytes.extend_from_slice(&payload);
        match &best {
            Some(b) if b.len() <= bytes.len() => {}
            _ => best = Some(bytes),
        }
    };
    consider(RICE_RUN, encode_rice_run(symbols));
    consider(RICE, encode_rice(symbols));
    consider(VARINT, encode_varint(symbols));
    // The general family carries an 8-byte length prefix, so it only ever wins
    // when a frame is large enough to amortise it.
    if symbols.len() >= 1024 {
        for &c in general {
            let payload = c.encode(symbols);
            if let Some(id) = general_offset(c) {
                consider(id, payload);
            }
        }
    }
    best.unwrap_or_else(|| vec![VARINT])
}

fn general_offset(c: rc2::ResidualCodecV2) -> Option<u8> {
    let id = c.id();
    (id < 128).then_some(id + V2_BASE)
}

/// Decode a residual of exactly `n` symbols.
pub fn decode(bytes: &[u8], n: usize) -> Result<Vec<i32>> {
    let (&id, payload) = bytes
        .split_first()
        .ok_or_else(|| Error::malformed("empty voice residual"))?;
    if id >= V2_BASE {
        rc2::ResidualCodecV2::from_id(id - V2_BASE)
            .ok_or_else(|| Error::new(Kind::Unsupported, format!("unknown residual codec {id}")))?;
        let mut raw = Vec::with_capacity(payload.len() + 1);
        raw.push(id - V2_BASE);
        raw.extend_from_slice(payload);
        return rc2::decode_encoding_v3(&raw, n)?.decode(n);
    }
    let mut r = BitReader::new(payload);
    match id {
        RICE => {
            let k = r.bits(5)? as u8;
            if k > K_MAX {
                return Err(Error::malformed("voice Rice parameter out of range"));
            }
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                out.push(unzigzag(r.rice(k)?));
            }
            Ok(out)
        }
        RICE_RUN => {
            let k = r.bits(5)? as u8;
            if k > K_MAX {
                return Err(Error::malformed("voice Rice parameter out of range"));
            }
            let mut out = Vec::with_capacity(n);
            while out.len() < n {
                let z = r.gamma()? - 1;
                if z > (n - out.len()) as u64 {
                    return Err(Error::malformed("voice zero run overruns the frame"));
                }
                out.extend(std::iter::repeat_n(0i32, z as usize));
                if out.len() >= n {
                    break;
                }
                out.push(unzigzag(r.rice(k)? + 1));
            }
            Ok(out)
        }
        VARINT => {
            let mut out = Vec::with_capacity(n);
            let mut i = 0usize;
            let mut shift = 0u32;
            let mut acc = 0u64;
            while out.len() < n {
                if i >= payload.len() {
                    return Err(Error::malformed("voice varint residual is truncated"));
                }
                let byte = payload[i];
                i += 1;
                if shift > 35 {
                    return Err(Error::malformed("voice varint is unbounded"));
                }
                acc |= u64::from(byte & 0x7f) << shift;
                if byte & 0x80 == 0 {
                    out.push(unzigzag(acc));
                    acc = 0;
                    shift = 0;
                } else {
                    shift += 7;
                }
            }
            if i != payload.len() {
                return Err(Error::malformed("voice varint residual has trailing bytes"));
            }
            Ok(out)
        }
        other => Err(Error::new(
            Kind::Unsupported,
            format!("unknown voice residual codec {other}"),
        )),
    }
}

/// Cheap complete-byte estimate for the encoder's rate search. Exact, not
/// modelled: it runs the encoders and measures.
pub fn estimated_bytes(symbols: &[i32]) -> usize {
    let a = encode_rice_run(symbols).len() + 1;
    let b = encode_rice(symbols).len() + 1;
    let c = encode_varint(symbols).len() + 1;
    a.min(b).min(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(v: &[i32]) {
        let enc = encode_best(v, &[]);
        let back = decode(&enc, v.len()).expect("decodes");
        assert_eq!(back, v, "round trip failed for {} symbols", v.len());
    }

    #[test]
    fn compact_codecs_round_trip() {
        roundtrip(&[]);
        roundtrip(&[0]);
        roundtrip(&[1]);
        roundtrip(&[-1]);
        roundtrip(&[0, 0, 0, 0, 0]);
        roundtrip(&[7, -3, 0, 0, 12, -1, 0]);
        roundtrip(&(0..200).map(|i| i - 100).collect::<Vec<_>>());
        roundtrip(&(0..300).map(|i| ((i * 37) % 11) - 5).collect::<Vec<_>>());
        roundtrip(&(0..160).map(|_| 0i32).collect::<Vec<_>>());
    }

    #[test]
    fn extreme_values_round_trip() {
        roundtrip(&[i32::MIN, i32::MAX, -1, 0, 1]);
        roundtrip(&[i32::MIN; 5]);
        roundtrip(&[i32::MAX; 5]);
    }

    #[test]
    fn zero_heavy_streams_cost_far_less_than_their_raw_width() {
        // 160 zeros must not cost 160 bits — the whole point of the run code.
        let zeros = vec![0i32; 160];
        let bytes = encode_best(&zeros, &[]);
        assert!(
            bytes.len() <= 4,
            "160 zeros cost {} bytes; the compact coder is not compact",
            bytes.len()
        );
        // And a realistic speech-like residual must beat two bytes per symbol.
        let residual: Vec<i32> = (0..160)
            .map(|i| if i % 5 == 0 { (i % 7) - 3 } else { 0 })
            .collect();
        let bytes = encode_best(&residual, &[]);
        assert!(
            bytes.len() < 40,
            "sparse residual cost {} bytes",
            bytes.len()
        );
    }

    #[test]
    fn the_estimate_is_exact_for_the_compact_codecs() {
        for v in [
            vec![0i32; 160],
            (0..160).map(|i| i - 80).collect::<Vec<_>>(),
            (0..160).map(|i| i % 3).collect::<Vec<_>>(),
        ] {
            let enc = encode_best(&v, &[]);
            assert_eq!(estimated_bytes(&v), enc.len(), "estimate must be exact");
        }
    }

    #[test]
    fn all_zeros_and_all_ones_are_small() {
        assert!(encode_best(&vec![1i32; 160], &[]).len() < 160);
        assert!(encode_best(&vec![-1i32; 160], &[]).len() < 160);
    }

    #[test]
    fn truncated_payloads_are_rejected() {
        let v: Vec<i32> = (0..64).map(|i| i - 32).collect();
        let enc = encode_best(&v, &[]);
        for cut in 1..enc.len() {
            let r = decode(&enc[..cut], v.len());
            assert!(r.is_err(), "truncation to {cut} must not decode");
        }
    }

    #[test]
    fn the_general_family_is_reachable_for_large_frames() {
        // A 4096-symbol frame is large enough to amortise the general prefix;
        // whatever wins must still round trip through the same dispatcher.
        let v: Vec<i32> = (0..4096)
            .map(|i| if i % 17 == 0 { (i % 251) - 125 } else { 0 })
            .collect();
        let enc = encode_best(&v, &rc2::SEARCH_CODECS);
        let back = decode(&enc, v.len()).unwrap();
        assert_eq!(back, v);
    }
}
