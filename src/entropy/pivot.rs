//! `PivotSideStreams` — level-transposed prefix coding for tiny side streams
//! (Phase 6 mechanism 15).
//!
//! Small high-volume side streams (model ids, expert selectors, scale classes,
//! MRU tokens) are commonly coded with a Huffman table. A conventional decoder
//! walks the tree one symbol at a time, which is a serial pointer chase.
//! PivCo-style coding stores the **same prefix code** but reorders its bits **by
//! tree level**: first every symbol's first bit, then every still-active
//! symbol's second bit, and so on. Each level is then a flat pass over the active
//! symbols that a vector unit can partition, with no per-symbol branch.
//!
//! The layout carries no ISA width: it is a deterministic permutation of the
//! code bits, so the same bytes decode identically with a scalar level pass or a
//! wide vector pass. This module provides both layouts plus a serial decoder as
//! the reference.

use crate::error::{Error, Result};

/// A canonical prefix codebook: `(code, length)` per symbol, MSB-first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PivotCodebook {
    /// Per-symbol `(code, length_in_bits)`.
    pub codes: Vec<(u32, u8)>,
    /// Longest code length in bits.
    pub max_len: u8,
}

impl PivotCodebook {
    /// Per-symbol code lengths (the canonical transmitted side information).
    pub fn lengths(&self) -> Vec<u8> {
        self.codes.iter().map(|&(_, l)| l).collect()
    }
}

/// Build a canonical prefix code for a frequency table.
///
/// Huffman code lengths are computed with a deterministic tie-break (heavier
/// weight first, then the smaller symbol index), and codes are assigned
/// canonically by `(length, symbol)`. A one-symbol table gets length 1.
pub fn huffman_codebook(freqs: &[u64]) -> PivotCodebook {
    let k = freqs.len();
    if k == 0 {
        return PivotCodebook {
            codes: Vec::new(),
            max_len: 0,
        };
    }
    // Only symbols with nonzero frequency participate; zero-frequency symbols
    // receive length 0 and are never coded.
    let mut weight: Vec<u64> = Vec::new();
    let mut min_sym: Vec<usize> = Vec::new();
    let mut children: Vec<Option<(usize, usize)>> = Vec::new();
    let mut leaf: Vec<Option<usize>> = Vec::new();
    let mut active: Vec<usize> = Vec::new();
    for (s, &f) in freqs.iter().enumerate() {
        if f > 0 {
            weight.push(f);
            min_sym.push(s);
            children.push(None);
            leaf.push(Some(s));
            active.push(weight.len() - 1);
        }
    }
    if active.is_empty() {
        return PivotCodebook {
            codes: vec![(0u32, 0u8); k],
            max_len: 0,
        };
    }
    if active.len() == 1 {
        let mut codes = vec![(0u32, 0u8); k];
        codes[min_sym[active[0]]] = (0, 1);
        return PivotCodebook { codes, max_len: 1 };
    }
    while active.len() > 1 {
        active.sort_by(|&a, &b| {
            weight[a]
                .cmp(&weight[b])
                .then(min_sym[a].cmp(&min_sym[b]))
                .then(a.cmp(&b))
        });
        let a = active.remove(0);
        let b = active.remove(0);
        weight.push(weight[a].saturating_add(weight[b]));
        min_sym.push(min_sym[a].min(min_sym[b]));
        children.push(Some((a, b)));
        leaf.push(None);
        active.push(weight.len() - 1);
    }
    let root = active[0];
    let mut lengths = vec![0u8; k];
    let mut stack = vec![(root, 0u32)];
    while let Some((id, d)) = stack.pop() {
        match children[id] {
            Some((l, r)) => {
                stack.push((l, d + 1));
                stack.push((r, d + 1));
            }
            None => {
                if let Some(s) = leaf[id] {
                    lengths[s] = d.clamp(1, 255) as u8;
                }
            }
        }
    }
    canonical_from_lengths(&lengths)
}

/// Assign canonical codes from lengths (sorted by `(length, symbol)`).
fn canonical_from_lengths(lengths: &[u8]) -> PivotCodebook {
    let max_len = lengths.iter().copied().max().unwrap_or(0);
    let mut order: Vec<usize> = (0..lengths.len()).filter(|&s| lengths[s] > 0).collect();
    order.sort_by(|&a, &b| lengths[a].cmp(&lengths[b]).then(a.cmp(&b)));
    let mut codes = vec![(0u32, 0u8); lengths.len()];
    let mut code: u32 = 0;
    let mut prev_len: u8 = 0;
    for &s in &order {
        let l = lengths[s];
        code <<= u32::from(l - prev_len);
        codes[s] = (code, l);
        code += 1;
        prev_len = l;
    }
    PivotCodebook { codes, max_len }
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
            return Err(Error::malformed("pivot bit stream is truncated"));
        }
        let b = (self.bytes[byte] >> (7 - (self.bit & 7))) & 1;
        self.bit += 1;
        Ok(b == 1)
    }
}

fn code_bit(code: u32, len: u8, level: u8) -> bool {
    (code >> (u32::from(len) - 1 - u32::from(level))) & 1 == 1
}

/// Conventional layout: each symbol's code bits back to back.
pub fn encode_serial(symbols: &[usize], cb: &PivotCodebook) -> Result<Vec<u8>> {
    let mut w = BitWriter::new();
    for &s in symbols {
        let (code, len) = *cb
            .codes
            .get(s)
            .ok_or_else(|| Error::malformed("pivot symbol out of range"))?;
        if len == 0 {
            return Err(Error::malformed("pivot symbol has no code"));
        }
        for level in 0..len {
            w.bit(code_bit(code, len, level));
        }
    }
    Ok(w.finish())
}

/// PivCo layout: all first bits, then all second bits of still-active symbols.
pub fn encode_pivot(symbols: &[usize], cb: &PivotCodebook) -> Result<Vec<u8>> {
    let mut w = BitWriter::new();
    for level in 0..cb.max_len {
        for &s in symbols {
            let (code, len) = *cb
                .codes
                .get(s)
                .ok_or_else(|| Error::malformed("pivot symbol out of range"))?;
            if len == 0 {
                return Err(Error::malformed("pivot symbol has no code"));
            }
            if len > level {
                w.bit(code_bit(code, len, level));
            }
        }
    }
    Ok(w.finish())
}

/// Reference serial decode: a prefix walk per symbol.
pub fn decode_serial(bytes: &[u8], n: usize, cb: &PivotCodebook) -> Result<Vec<usize>> {
    // Map (code, len) to symbol.
    let mut reader = BitReader::new(bytes);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut code = 0u32;
        let mut len = 0u8;
        loop {
            if len >= cb.max_len {
                return Err(Error::malformed("pivot code exceeds the maximum length"));
            }
            code = (code << 1) | u32::from(reader.read_bit()?);
            len += 1;
            if let Some(s) = cb.codes.iter().position(|&(c, l)| l == len && c == code) {
                out.push(s);
                break;
            }
        }
    }
    Ok(out)
}

/// Level-transposed decode: one flat pass per level, reassembling each active
/// symbol's code without a per-symbol tree walk.
pub fn decode_pivot(bytes: &[u8], n: usize, cb: &PivotCodebook) -> Result<Vec<usize>> {
    let mut reader = BitReader::new(bytes);
    let mut paths = vec![(0u32, 0u8); n];
    let mut resolved: Vec<Option<usize>> = vec![None; n];
    let mut remaining = n;
    for _level in 0..cb.max_len {
        if remaining == 0 {
            break;
        }
        for i in 0..n {
            if resolved[i].is_some() {
                continue;
            }
            let (code, len) = paths[i];
            let bit = reader.read_bit()?;
            let ncode = (code << 1) | u32::from(bit);
            let nlen = len + 1;
            if let Some(s) = cb.codes.iter().position(|&(c, l)| l == nlen && c == ncode) {
                resolved[i] = Some(s);
                remaining -= 1;
            } else {
                paths[i] = (ncode, nlen);
            }
        }
    }
    if remaining != 0 {
        return Err(Error::malformed(
            "pivot stream ended before all codes resolved",
        ));
    }
    resolved
        .into_iter()
        .collect::<Option<Vec<usize>>>()
        .ok_or_else(|| Error::malformed("pivot stream left an unresolved code"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codebook() -> PivotCodebook {
        huffman_codebook(&[50, 20, 10, 6, 4, 3, 2, 1])
    }

    fn stream() -> Vec<usize> {
        (0..3000usize).map(|i| (i * i + i / 3) % 8).collect()
    }

    #[test]
    fn serial_round_trips() {
        let cb = codebook();
        let s = stream();
        let bytes = encode_serial(&s, &cb).unwrap();
        assert_eq!(decode_serial(&bytes, s.len(), &cb).unwrap(), s);
    }

    #[test]
    fn pivot_round_trips_and_agrees_with_serial() {
        let cb = codebook();
        let s = stream();
        let serial = encode_serial(&s, &cb).unwrap();
        let pivot = encode_pivot(&s, &cb).unwrap();
        assert_eq!(decode_serial(&serial, s.len(), &cb).unwrap(), s);
        assert_eq!(decode_pivot(&pivot, s.len(), &cb).unwrap(), s);
        // Same number of code bits, so at most one byte of padding difference.
        let diff = (serial.len() as i64 - pivot.len() as i64).abs();
        assert!(diff <= 1, "serial {} pivot {}", serial.len(), pivot.len());
    }

    #[test]
    fn single_symbol_alphabet_is_handled() {
        let cb = huffman_codebook(&[7]);
        assert_eq!(cb.max_len, 1);
        let s = vec![0usize; 100];
        let bytes = encode_pivot(&s, &cb).unwrap();
        assert_eq!(decode_pivot(&bytes, s.len(), &cb).unwrap(), s);
    }
}
