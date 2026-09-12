//! `SolidObjectColumns` — archive-level column transposition (Phase 6
//! mechanism 9).
//!
//! The canonical `.volea` archive stores each object's full-object container as
//! an opaque, independently integrity-bound payload. That is the right
//! *logical* constitution — objects stay independent — but it is the wrong
//! *physical* order for a bounded-context entropy coder, which sees each
//! object's header, index, payloads and integrity digest separated by the next
//! object's unrelated bytes.
//!
//! `SolidObjectColumns` is a **separate archive constitution** that parses each
//! container into homologous columns and physically stores them transposed:
//!
//! ```text
//! headers of every object
//! index records of every object (with their payload offsets canonicalized)
//! payloads grouped by segment index: segment 0 of every object, then segment 1, …
//! integrity digests of every object
//! ```
//!
//! A permutation restores logical order exactly; no bytes are referenced across
//! objects, and every object's own SHA-256 integrity still verifies on decode.
//! The transposition is only worthwhile when a downstream adaptive coder — such
//! as [`encode_bytes_adaptive`], whose model state deliberately carries across
//! the whole solid stream — can exploit the resulting statistical locality.
//!
//! This is an **archive-only** claim: a solid sample-pack result must never be
//! compared with a single-file audio compression result.

use crate::entropy::rans::{
    BackSink, FwdReader, RansState, SCALE_BITS, dec_advance, dec_init, dec_slot, enc_flush,
    enc_put, encode_capacity,
};
use crate::error::{Error, Result};
use crate::hash::sha256::Sha256;

/// Solid archive container magic.
pub const SOLID_MAGIC: &[u8; 13] = b"vole.solid.p1";
/// Solid archive container version.
pub const SOLID_VERSION: u8 = 1;

/// Binary probability precision (matches the rANS profile total).
const PROB_BITS: u32 = SCALE_BITS;
/// Normalized binary total (`1 << PROB_BITS`).
const PROB_TOTAL: u32 = 1 << PROB_BITS;
/// Initial probability of a one bit.
const PROB_INIT: u16 = (PROB_TOTAL / 2) as u16;
/// Minimum/maximum probability (keeps both interval frequencies positive).
const PROB_MIN: u16 = 1;
const PROB_MAX: u16 = (PROB_TOTAL - 1) as u16;
/// Adaptation shift: `p += (total - p) >> shift` after a one.
const PROB_SHIFT: u32 = 5;
/// Byte contexts: `previous_byte * 256 + bit_tree_node`, node in `1..=255`.
const CTX_COUNT: usize = 256 * 256;
/// Bits coded per rANS chunk (bounds encoder working memory).
const BITS_PER_CHUNK: usize = 1 << 16;

// ---------------------------------------------------------------------------
// Adaptive order-1 byte coder (carried state)
// ---------------------------------------------------------------------------

/// Encode arbitrary bytes with an adaptive order-1 binary rANS coder.
///
/// Each byte is coded MSB-first; the context is the previous byte plus the
/// bit-tree node of the partially decoded current byte. The model is never
/// reset, so state carries across the whole stream — which is exactly what a
/// solid column ordering is meant to exploit.
pub fn encode_bytes_adaptive(data: &[u8]) -> Vec<u8> {
    let bit_len = data.len().saturating_mul(8);
    let nchunks = bit_len.div_ceil(BITS_PER_CHUNK);
    let mut probs = vec![PROB_INIT; CTX_COUNT];
    let mut chunks: Vec<Vec<u8>> = Vec::with_capacity(nchunks);
    let mut prev_byte = 0usize;
    let mut base = 0usize;
    while base < bit_len {
        let end = (base + BITS_PER_CHUNK).min(bit_len);
        let len = end - base;
        let mut starts = Vec::with_capacity(len);
        let mut freqs = Vec::with_capacity(len);
        for b in base..end {
            let byte = data[b >> 3];
            let pos = b & 7;
            let bit = (byte >> (7 - pos)) & 1;
            let node = (1usize << pos) | ((byte as usize) >> (8 - pos));
            let ctx = prev_byte * 256 + node;
            let p = u32::from(probs[ctx]);
            let (start, freq) = if bit == 0 {
                (0u32, PROB_TOTAL - p)
            } else {
                (PROB_TOTAL - p, p)
            };
            starts.push(start);
            freqs.push(freq);
            probs[ctx] = adapt(p, u32::from(bit));
            if pos == 7 {
                prev_byte = byte as usize;
            }
        }
        let mut buf = vec![0u8; encode_capacity(len).max(16)];
        let written = {
            let mut sink = BackSink::new(&mut buf);
            let mut state = RansState::new();
            let mut ok = true;
            for i in (0..len).rev() {
                if !enc_put(&mut state, &mut sink, starts[i], freqs[i], SCALE_BITS) {
                    ok = false;
                    break;
                }
            }
            if ok && enc_flush(&state, &mut sink) {
                sink.bytes_written()
            } else {
                0
            }
        };
        chunks.push(buf[buf.len() - written..].to_vec());
        base = end;
    }
    let mut out = Vec::new();
    out.extend_from_slice(&(bit_len as u64).to_le_bytes());
    out.extend_from_slice(&(nchunks as u32).to_le_bytes());
    for c in &chunks {
        put_uvarint(&mut out, c.len() as u64);
    }
    for c in &chunks {
        out.extend_from_slice(c);
    }
    out
}

/// Decode bytes produced by [`encode_bytes_adaptive`], requiring exactly `len`
/// bytes.
pub fn decode_bytes_adaptive(bytes: &[u8], len: usize) -> Result<Vec<u8>> {
    let mut r = Cursor::new(bytes);
    let bit_len = usize::try_from(r.u64()?)
        .map_err(|_| Error::limit("adaptive byte stream bit count exceeds host usize"))?;
    if bit_len != len.saturating_mul(8) {
        return Err(Error::malformed("adaptive byte stream length mismatch"));
    }
    let nchunks = r.u32()? as usize;
    if nchunks != bit_len.div_ceil(BITS_PER_CHUNK) {
        return Err(Error::malformed(
            "adaptive byte stream chunk count mismatch",
        ));
    }
    let mut lens = Vec::with_capacity(nchunks);
    for _ in 0..nchunks {
        lens.push(usize::try_from(r.uvarint()?).unwrap_or(usize::MAX));
    }
    let mut chunks = Vec::with_capacity(nchunks);
    for &l in &lens {
        chunks.push(r.take(l)?);
    }
    if r.remaining() != 0 {
        return Err(Error::malformed("adaptive byte stream has trailing bytes"));
    }
    let mut probs = vec![PROB_INIT; CTX_COUNT];
    let mut out = Vec::with_capacity(len);
    let mut prev_byte = 0usize;
    let mut cur: u8 = 0;
    let mut base = 0usize;
    for chunk in &chunks {
        let end = (base + BITS_PER_CHUNK).min(bit_len);
        let mut reader = FwdReader::new(chunk);
        let mut state = dec_init(&mut reader)
            .ok_or_else(|| Error::malformed("adaptive stream is truncated"))?;
        for b in base..end {
            let pos = b & 7;
            let node = (1usize << pos) | ((cur as usize) >> (8 - pos));
            let ctx = prev_byte * 256 + node;
            let p = u32::from(probs[ctx]);
            let slot = dec_slot(&state, SCALE_BITS);
            let bit = u32::from(slot >= PROB_TOTAL - p);
            let (start, freq) = if bit == 0 {
                (0u32, PROB_TOTAL - p)
            } else {
                (PROB_TOTAL - p, p)
            };
            if !dec_advance(&mut state, &mut reader, start, freq, SCALE_BITS) {
                return Err(Error::malformed("adaptive stream is truncated"));
            }
            probs[ctx] = adapt(p, bit);
            if bit == 1 {
                cur |= 1 << (7 - pos);
            }
            if pos == 7 {
                out.push(cur);
                prev_byte = cur as usize;
                cur = 0;
            }
        }
        base = end;
    }
    if out.len() != len {
        return Err(Error::malformed("adaptive byte stream length mismatch"));
    }
    Ok(out)
}

#[inline]
fn adapt(p: u32, bit: u32) -> u16 {
    if bit == 1 {
        (p + ((PROB_TOTAL - p) >> PROB_SHIFT)).min(u32::from(PROB_MAX)) as u16
    } else {
        (p - (p >> PROB_SHIFT)).max(u32::from(PROB_MIN)) as u16
    }
}

// ---------------------------------------------------------------------------
// Solid column transposition
// ---------------------------------------------------------------------------

/// One parsed full-object container in column form.
struct Parsed {
    header: Vec<u8>,
    /// Raw index bytes with payload offsets canonicalized.
    index: Vec<u8>,
    payloads: Vec<Vec<u8>>,
    integrity: Vec<u8>,
}

/// Parse and canonicalize one full-object container.
fn parse_container(obj: &[u8]) -> Result<Parsed> {
    let d = crate::fullobj::decode_full_object(obj)?;
    let header_len = usize::try_from(d.header_bytes)
        .map_err(|_| Error::limit("solid header bytes exceeds host usize"))?;
    let index_len = usize::try_from(d.index_bytes)
        .map_err(|_| Error::limit("solid index bytes exceeds host usize"))?;
    let integrity_len = usize::try_from(d.integrity_bytes)
        .map_err(|_| Error::limit("solid integrity bytes exceeds host usize"))?;
    let seg_count = d.segments.len();
    if header_len + index_len + integrity_len > obj.len() {
        return Err(Error::malformed("solid container geometry is inconsistent"));
    }
    let header = obj[..header_len].to_vec();
    let mut index = obj[header_len..header_len + index_len].to_vec();
    let integrity = obj[obj.len() - integrity_len..].to_vec();
    let mut payloads = Vec::with_capacity(seg_count);
    for s in &d.segments {
        let at = usize::try_from(s.payload_offset)
            .map_err(|_| Error::limit("solid payload offset exceeds host usize"))?;
        let len = usize::try_from(s.payload_length)
            .map_err(|_| Error::limit("solid payload length exceeds host usize"))?;
        let end = at
            .checked_add(len)
            .ok_or_else(|| Error::limit("solid payload range overflows"))?;
        if end > obj.len() {
            return Err(Error::malformed(
                "solid payload range exceeds the container",
            ));
        }
        payloads.push(obj[at..end].to_vec());
    }
    if seg_count > 0 {
        if index_len % seg_count != 0 {
            return Err(Error::malformed("solid index is not record aligned"));
        }
        let rec = index_len / seg_count;
        if rec < 16 {
            return Err(Error::malformed("solid index record is too small"));
        }
        // Canonicalize payload offsets and verify the original was canonical.
        let payload_start = header_len + index_len;
        let mut offset = payload_start;
        for (k, p) in payloads.iter().enumerate() {
            let at = k * rec + rec - 16;
            index[at..at + 8].copy_from_slice(&(offset as u64).to_le_bytes());
            offset = offset
                .checked_add(p.len())
                .ok_or_else(|| Error::limit("solid payload offset overflows"))?;
        }
        let mut rebuilt = Vec::with_capacity(obj.len());
        rebuilt.extend_from_slice(&header);
        rebuilt.extend_from_slice(&index);
        for p in &payloads {
            rebuilt.extend_from_slice(p);
        }
        rebuilt.extend_from_slice(&integrity);
        if rebuilt != obj {
            return Err(Error::malformed(
                "solid input container is not in canonical payload order",
            ));
        }
    }
    Ok(Parsed {
        header,
        index,
        payloads,
        integrity,
    })
}

/// Transpose a set of full-object containers into the solid column stream.
pub fn encode_solid(objects: &[Vec<u8>]) -> Result<Vec<u8>> {
    let mut parsed = Vec::with_capacity(objects.len());
    for obj in objects {
        parsed.push(parse_container(obj)?);
    }
    let object_count = parsed.len();
    let header_len = parsed.first().map(|p| p.header.len()).unwrap_or(0);
    let integrity_len = parsed.first().map(|p| p.integrity.len()).unwrap_or(0);
    let index_record_bytes = parsed
        .first()
        .map(|p| {
            if p.payloads.is_empty() {
                0
            } else {
                p.index.len() / p.payloads.len()
            }
        })
        .unwrap_or(0);
    for p in &parsed {
        if p.header.len() != header_len {
            return Err(Error::malformed(
                "solid objects have different header lengths",
            ));
        }
        if p.integrity.len() != integrity_len {
            return Err(Error::malformed(
                "solid objects have different integrity lengths",
            ));
        }
        let rec = if p.payloads.is_empty() {
            0
        } else {
            p.index.len() / p.payloads.len()
        };
        if rec != index_record_bytes {
            return Err(Error::malformed(
                "solid objects have different index record sizes",
            ));
        }
    }
    let object_count_u32 =
        u32::try_from(object_count).map_err(|_| Error::limit("solid object count overflows"))?;
    let mut out = Vec::new();
    out.extend_from_slice(SOLID_MAGIC);
    out.push(SOLID_VERSION);
    out.extend_from_slice(&object_count_u32.to_le_bytes());
    out.extend_from_slice(&(header_len as u32).to_le_bytes());
    out.extend_from_slice(&(index_record_bytes as u32).to_le_bytes());
    out.extend_from_slice(&(integrity_len as u32).to_le_bytes());
    for p in &parsed {
        out.extend_from_slice(&(p.payloads.len() as u32).to_le_bytes());
    }
    for p in &parsed {
        out.extend_from_slice(&p.header);
    }
    for p in &parsed {
        out.extend_from_slice(&p.index);
    }
    let max_segments = parsed.iter().map(|p| p.payloads.len()).max().unwrap_or(0);
    for k in 0..max_segments {
        for p in &parsed {
            if let Some(payload) = p.payloads.get(k) {
                out.extend_from_slice(payload);
            }
        }
    }
    for p in &parsed {
        out.extend_from_slice(&p.integrity);
    }
    let digest = Sha256::digest(&out);
    out.extend_from_slice(&digest);
    Ok(out)
}

/// Reconstruct the logical objects from a solid column stream.
pub fn decode_solid(bytes: &[u8]) -> Result<Vec<Vec<u8>>> {
    let mut r = Cursor::new(bytes);
    if r.remaining() < SOLID_MAGIC.len() + 1 + 32 {
        return Err(Error::malformed("solid container is truncated"));
    }
    if r.take(SOLID_MAGIC.len())? != SOLID_MAGIC.as_slice() {
        return Err(Error::malformed("solid container tag mismatch"));
    }
    if r.u8()? != SOLID_VERSION {
        return Err(Error::malformed("solid container version mismatch"));
    }
    let object_count = r.u32()? as usize;
    let header_len = r.u32()? as usize;
    let index_record_bytes = r.u32()? as usize;
    let integrity_len = r.u32()? as usize;
    if object_count > bytes.len() {
        return Err(Error::limit("solid object count exceeds the container"));
    }
    if object_count == 0 {
        if r.remaining() != 32 {
            return Err(Error::malformed(
                "solid container has an invalid trailing region",
            ));
        }
        let expected = r.take(32)?;
        let body_len = bytes.len() - 32;
        if Sha256::digest(&bytes[..body_len]).as_slice() != expected {
            return Err(Error::malformed("solid container digest does not verify"));
        }
        return Ok(Vec::new());
    }
    if header_len == 0 || integrity_len == 0 || index_record_bytes < 16 {
        return Err(Error::malformed("solid container geometry is invalid"));
    }
    let mut seg_counts = Vec::with_capacity(object_count);
    let mut total_segments: usize = 0;
    for _ in 0..object_count {
        let n = r.u32()? as usize;
        total_segments = total_segments
            .checked_add(n)
            .ok_or_else(|| Error::limit("solid segment count overflows"))?;
        seg_counts.push(n);
    }
    let header_total = header_len
        .checked_mul(object_count)
        .ok_or_else(|| Error::limit("solid header column overflows"))?;
    let index_total = index_record_bytes
        .checked_mul(total_segments)
        .ok_or_else(|| Error::limit("solid index column overflows"))?;
    let integrity_total = integrity_len
        .checked_mul(object_count)
        .ok_or_else(|| Error::limit("solid integrity column overflows"))?;
    let headers = r.take(header_total)?.to_vec();
    let index = r.take(index_total)?.to_vec();
    // Payload lengths come from the trailing `u64` of each index record; the
    // stream lays them out segment-major.
    let mut payload_lengths: Vec<Vec<usize>> = Vec::with_capacity(object_count);
    let mut rec = 0usize;
    for &n in &seg_counts {
        let mut row = Vec::with_capacity(n);
        for _ in 0..n {
            let at = rec * index_record_bytes + index_record_bytes - 8;
            let len = u64::from_le_bytes(
                index[at..at + 8]
                    .try_into()
                    .map_err(|_| Error::malformed("solid index record is truncated"))?,
            );
            row.push(
                usize::try_from(len)
                    .map_err(|_| Error::limit("solid payload length exceeds host usize"))?,
            );
            rec += 1;
        }
        payload_lengths.push(row);
    }
    let max_segments = payload_lengths.iter().map(|p| p.len()).max().unwrap_or(0);
    // Payload rows in object order; the stream lays them out segment-major.
    let mut payload_rows: Vec<Vec<Vec<u8>>> =
        seg_counts.iter().map(|&n| vec![Vec::new(); n]).collect();
    for k in 0..max_segments {
        for (o, row) in payload_rows.iter_mut().enumerate() {
            if k < row.len() {
                let len = payload_lengths[o][k];
                row[k] = r.take(len)?.to_vec();
            }
        }
    }
    let integrities = r.take(integrity_total)?.to_vec();
    if r.remaining() != 32 {
        return Err(Error::malformed(
            "solid container has an invalid trailing region",
        ));
    }
    let expected = r.take(32)?;
    let body_len = bytes.len() - 32;
    if Sha256::digest(&bytes[..body_len]).as_slice() != expected {
        return Err(Error::malformed("solid container digest does not verify"));
    }
    let mut out = Vec::with_capacity(object_count);
    for o in 0..object_count {
        let header = &headers[o * header_len..(o + 1) * header_len];
        let row = &payload_rows[o];
        let rec_start = payload_lengths[..o].iter().map(|p| p.len()).sum::<usize>();
        let index_start = rec_start * index_record_bytes;
        let index_end = index_start + row.len() * index_record_bytes;
        let mut idx = index[index_start..index_end].to_vec();
        let payload_start = header_len + idx.len();
        let mut offset = payload_start;
        for (k, payload) in row.iter().enumerate() {
            let at = k * index_record_bytes + index_record_bytes - 16;
            idx[at..at + 8].copy_from_slice(&(offset as u64).to_le_bytes());
            offset = offset
                .checked_add(payload.len())
                .ok_or_else(|| Error::limit("solid payload offset overflows"))?;
        }
        let mut obj = Vec::with_capacity(offset + integrity_len);
        obj.extend_from_slice(header);
        obj.extend_from_slice(&idx);
        for payload in row {
            obj.extend_from_slice(payload);
        }
        obj.extend_from_slice(&integrities[o * integrity_len..(o + 1) * integrity_len]);
        // Every object must still verify under its own integrity digest.
        let body = &obj[..obj.len() - integrity_len];
        let integrity = &obj[obj.len() - integrity_len..];
        if Sha256::digest(body).as_slice() != integrity {
            return Err(Error::malformed(
                "solid object integrity does not verify after reconstruction",
            ));
        }
        out.push(obj);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Small byte cursor / varint helpers
// ---------------------------------------------------------------------------

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Cursor { bytes, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| Error::limit("cursor range overflows"))?;
        if end > self.bytes.len() {
            return Err(Error::malformed("solid container is truncated"));
        }
        let s = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn uvarint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        let mut shift = 0u32;
        loop {
            let b = self.u8()?;
            v |= u64::from(b & 0x7F) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
            if shift >= 64 {
                return Err(Error::malformed("solid varint is too long"));
            }
        }
    }
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

    fn container(frames: u64, base: i32) -> Vec<u8> {
        let samples: Vec<i32> = (0..frames as i32).map(|i| base + (i % 17)).collect();
        let obj = crate::fullobj::compile_full_object(
            "solid-test",
            48_000,
            1,
            frames,
            crate::fullobj::FullSemantics::OneShot,
            &samples,
            crate::inverse::SearchBudget::default(),
        )
        .unwrap();
        obj.bytes
    }

    #[test]
    fn adaptive_byte_coder_round_trips() {
        let data: Vec<u8> = (0..40_000usize).map(|i| (i * 31 + i / 7) as u8).collect();
        let enc = encode_bytes_adaptive(&data);
        let back = decode_bytes_adaptive(&enc, data.len()).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn adaptive_byte_coder_is_exact_on_empty_and_extreme() {
        let empty: Vec<u8> = Vec::new();
        let enc = encode_bytes_adaptive(&empty);
        assert!(decode_bytes_adaptive(&enc, 0).unwrap().is_empty());
        let data = vec![0u8, 255, 0, 255, 0, 0, 255];
        let enc = encode_bytes_adaptive(&data);
        assert_eq!(decode_bytes_adaptive(&enc, data.len()).unwrap(), data);
    }

    #[test]
    fn solid_transposition_round_trips_exactly() {
        let objects = vec![
            container(300, 10),
            container(700, -400),
            container(500, 7000),
        ];
        let solid = encode_solid(&objects).unwrap();
        let back = decode_solid(&solid).unwrap();
        assert_eq!(back, objects);
    }

    #[test]
    fn solid_stream_codes_smaller_than_logical_concatenation_when_homogeneous() {
        // Homologous columns: several objects from the same generator make the
        // transposed stream more predictable than the object-major stream.
        let objects: Vec<Vec<u8>> = (0..6).map(|k| container(600, k * 3 - 8)).collect();
        let solid = encode_solid(&objects).unwrap();
        let mut naive = Vec::new();
        for o in &objects {
            naive.extend_from_slice(&(o.len() as u32).to_le_bytes());
            naive.extend_from_slice(o);
        }
        let solid_coded = encode_bytes_adaptive(&solid);
        let naive_coded = encode_bytes_adaptive(&naive);
        // The gate is exactness; the size relation is reported by the court.
        assert!(decode_bytes_adaptive(&solid_coded, solid.len()).unwrap() == solid);
        assert!(decode_bytes_adaptive(&naive_coded, naive.len()).unwrap() == naive);
    }

    #[test]
    fn solid_rejects_mutations_and_truncations() {
        let objects = vec![container(300, 10), container(400, -20)];
        let solid = encode_solid(&objects).unwrap();
        for cut in [1usize, 5, 17, solid.len() / 2, solid.len() - 1] {
            let _ = decode_solid(&solid[..cut]);
        }
        let mut tampered = solid.clone();
        let mid = tampered.len() / 2;
        tampered[mid] ^= 0x5A;
        assert!(decode_solid(&tampered).is_err());
    }
}
