//! Exact residual codec family v2 (Exp2, priorities `1`, `3.1`–`3.6`).
//!
//! Exp1's six codecs (`super::residual_codec`) are **frozen**: their bytes,
//! their selection order and their `encode_all` row set never change. This
//! module adds a *superset* family beside them. Because every Exp1 codec stays
//! selectable and v2 selection is a strict minimum with ascending-id ties,
//! Exp2 is structurally non-regressing:
//!
//! ```text
//! best_v2(residual) ≤ best_v1(residual)     for every residual
//! ```
//!
//! New codecs (ids 6..=11):
//!
//! ```text
//! id 6  PartitionRice    partitioned Rice over a frozen ladder (per-partition k)
//! id 7  CoreTailRice     two-regime Rice with an Exp-Golomb escape (heavy tails)
//! id 8  RunLengthRice    adaptive run-length/Rice over zero runs (RLGR family)
//! id 9  ZeroMaskRice     zero mask (bitmap/RLE/sparse) + magnitude stream
//! id 10 BytePlane        byte-length plane + significance byte planes + sign
//! id 11 ContextRans      context-conditioned rANS over magnitude bit-length buckets
//! ```
//!
//! Every decoder is length-checked, allocation-bounded and hostile-safe. Every
//! codec is an exact inverse on the canonical dense residual domain `Vec<i32>`.
//! No codec has semantic authority: the selected member is always the measured
//! minimum complete cost.

use crate::entropy::rans::{
    BackSink, FwdReader, RansState, dec_advance, dec_init, enc_flush, enc_put, encode_capacity,
};
use crate::error::{Error, Kind, Result};

/// Frozen partition ladder for `PartitionRice`.
pub const PARTITION_LADDER: [usize; 7] = [16, 32, 64, 128, 256, 512, 1024];

/// Alignment unit of the `PartitionRice` segmentation.
const PARTITION_UNIT: usize = 16;

/// Maximum Rice parameter considered by any v2 codec.
const MAX_RICE_K2: u32 = 40;

/// Maximum admissible unary run length (hostile-input bound).
const MAX_RICE_UNARY2: u64 = 1 << 20;

/// Maximum Exp-Golomb bit length accepted while decoding (hostile bound).
const MAX_EG0_BITS: u32 = 48;

/// Maximum zero-run chunk (keeps the adaptive run coder bounded).
const MAX_RUN_CHUNK: u64 = 1 << 24;

/// rANS alphabet: magnitude bit-length buckets `0..=33` (0 = value is zero).
const CTX_ALPHABET: usize = 34;

/// Context count of the `ContextRans` causal model (`O.2`-style tiny gate).
const CTX_CONTEXTS: usize = 4;

/// rANS scale bits used by `ContextRans` (matches the frozen audio profile).
const CTX_SCALE_BITS: u32 = crate::entropy::rans::SCALE_BITS;

/// rANS normalized total used by `ContextRans`.
const CTX_MODEL_TOTAL: u32 = crate::entropy::rans::MODEL_TOTAL;

/// The complete residual codec family: the frozen Exp1 members and the Exp2
/// additions, in canonical ascending-id order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ResidualCodecV2 {
    // --- frozen Exp1 members (delegated byte-for-byte) ---
    DenseI32 = 0,
    SparseDelta = 1,
    ZigZagVarint = 2,
    BlockRice = 3,
    PredictiveRice = 4,
    LiteralResidual = 5,
    // --- Exp2 additions ---
    PartitionRice = 6,
    CoreTailRice = 7,
    RunLengthRice = 8,
    ZeroMaskRice = 9,
    BytePlane = 10,
    ContextRans = 11,
    // --- Exp3 additions (Seal S4) ---
    /// Partitioned general Golomb coding (arbitrary, non-power-of-two `M`).
    Golomb = 12,
    /// Centered partitioned general Golomb (a stored residual center).
    CenteredGolomb = 13,
    /// Strip a common power-of-two factor, then encode the quotients.
    FactorShift = 14,
}

impl ResidualCodecV2 {
    /// The frozen Exp1 members, in canonical tie order.
    pub const ALL_V1: [ResidualCodecV2; 6] = [
        ResidualCodecV2::DenseI32,
        ResidualCodecV2::SparseDelta,
        ResidualCodecV2::ZigZagVarint,
        ResidualCodecV2::BlockRice,
        ResidualCodecV2::PredictiveRice,
        ResidualCodecV2::LiteralResidual,
    ];

    /// The Exp2 additions, in canonical tie order.
    pub const ALL_V2: [ResidualCodecV2; 6] = [
        ResidualCodecV2::PartitionRice,
        ResidualCodecV2::CoreTailRice,
        ResidualCodecV2::RunLengthRice,
        ResidualCodecV2::ZeroMaskRice,
        ResidualCodecV2::BytePlane,
        ResidualCodecV2::ContextRans,
    ];

    /// The whole family, ascending id.
    pub const ALL: [ResidualCodecV2; 12] = [
        ResidualCodecV2::DenseI32,
        ResidualCodecV2::SparseDelta,
        ResidualCodecV2::ZigZagVarint,
        ResidualCodecV2::BlockRice,
        ResidualCodecV2::PredictiveRice,
        ResidualCodecV2::LiteralResidual,
        ResidualCodecV2::PartitionRice,
        ResidualCodecV2::CoreTailRice,
        ResidualCodecV2::RunLengthRice,
        ResidualCodecV2::ZeroMaskRice,
        ResidualCodecV2::BytePlane,
        ResidualCodecV2::ContextRans,
    ];

    /// The Exp3 additions (Seal S4), in canonical tie order.
    pub const ALL_V3: [ResidualCodecV2; 15] = [
        ResidualCodecV2::DenseI32,
        ResidualCodecV2::SparseDelta,
        ResidualCodecV2::ZigZagVarint,
        ResidualCodecV2::BlockRice,
        ResidualCodecV2::PredictiveRice,
        ResidualCodecV2::LiteralResidual,
        ResidualCodecV2::PartitionRice,
        ResidualCodecV2::CoreTailRice,
        ResidualCodecV2::RunLengthRice,
        ResidualCodecV2::ZeroMaskRice,
        ResidualCodecV2::BytePlane,
        ResidualCodecV2::ContextRans,
        ResidualCodecV2::Golomb,
        ResidualCodecV2::CenteredGolomb,
        ResidualCodecV2::FactorShift,
    ];

    /// Canonical codec identifier byte.
    pub const fn id(self) -> u8 {
        self as u8
    }

    /// Stable evidence label.
    pub const fn name(self) -> &'static str {
        match self {
            ResidualCodecV2::DenseI32 => "dense_i32",
            ResidualCodecV2::SparseDelta => "sparse_delta",
            ResidualCodecV2::ZigZagVarint => "zigzag_varint",
            ResidualCodecV2::BlockRice => "block_rice",
            ResidualCodecV2::PredictiveRice => "predictive_rice",
            ResidualCodecV2::LiteralResidual => "literal_residual",
            ResidualCodecV2::PartitionRice => "partition_rice",
            ResidualCodecV2::CoreTailRice => "core_tail_rice",
            ResidualCodecV2::RunLengthRice => "run_length_rice",
            ResidualCodecV2::ZeroMaskRice => "zero_mask_rice",
            ResidualCodecV2::BytePlane => "byte_plane",
            ResidualCodecV2::ContextRans => "context_rans",
            ResidualCodecV2::Golomb => "golomb",
            ResidualCodecV2::CenteredGolomb => "centered_golomb",
            ResidualCodecV2::FactorShift => "factor_shift",
        }
    }

    /// Resolve a codec from its identifier byte (`0..=11`).
    pub const fn from_id(id: u8) -> Option<ResidualCodecV2> {
        match id {
            0 => Some(ResidualCodecV2::DenseI32),
            1 => Some(ResidualCodecV2::SparseDelta),
            2 => Some(ResidualCodecV2::ZigZagVarint),
            3 => Some(ResidualCodecV2::BlockRice),
            4 => Some(ResidualCodecV2::PredictiveRice),
            5 => Some(ResidualCodecV2::LiteralResidual),
            6 => Some(ResidualCodecV2::PartitionRice),
            7 => Some(ResidualCodecV2::CoreTailRice),
            8 => Some(ResidualCodecV2::RunLengthRice),
            9 => Some(ResidualCodecV2::ZeroMaskRice),
            10 => Some(ResidualCodecV2::BytePlane),
            11 => Some(ResidualCodecV2::ContextRans),
            12 => Some(ResidualCodecV2::Golomb),
            13 => Some(ResidualCodecV2::CenteredGolomb),
            14 => Some(ResidualCodecV2::FactorShift),
            _ => None,
        }
    }

    /// The frozen Exp1 member this maps to, if any.
    pub const fn as_v1(self) -> Option<crate::learned::residual_codec::ResidualCodec> {
        match self {
            ResidualCodecV2::DenseI32 => {
                Some(crate::learned::residual_codec::ResidualCodec::DenseI32)
            }
            ResidualCodecV2::SparseDelta => {
                Some(crate::learned::residual_codec::ResidualCodec::SparseDelta)
            }
            ResidualCodecV2::ZigZagVarint => {
                Some(crate::learned::residual_codec::ResidualCodec::ZigZagVarint)
            }
            ResidualCodecV2::BlockRice => {
                Some(crate::learned::residual_codec::ResidualCodec::BlockRice)
            }
            ResidualCodecV2::PredictiveRice => {
                Some(crate::learned::residual_codec::ResidualCodec::PredictiveRice)
            }
            ResidualCodecV2::LiteralResidual => {
                Some(crate::learned::residual_codec::ResidualCodec::LiteralResidual)
            }
            _ => None,
        }
    }

    /// True when this codec belongs to the frozen Exp1 family.
    pub const fn is_v1(self) -> bool {
        (self as u8) < 6
    }

    /// Encode a dense residual.
    pub fn encode(self, residual: &[i32]) -> Vec<u8> {
        match self {
            ResidualCodecV2::DenseI32
            | ResidualCodecV2::SparseDelta
            | ResidualCodecV2::ZigZagVarint
            | ResidualCodecV2::BlockRice
            | ResidualCodecV2::PredictiveRice
            | ResidualCodecV2::LiteralResidual => self.as_v1().expect("v1 codec").encode(residual),
            ResidualCodecV2::PartitionRice => encode_partition_rice(residual),
            ResidualCodecV2::CoreTailRice => encode_core_tail_rice(residual),
            ResidualCodecV2::RunLengthRice => encode_run_length_rice(residual),
            ResidualCodecV2::ZeroMaskRice => encode_zero_mask_rice(residual),
            ResidualCodecV2::BytePlane => encode_byte_plane(residual),
            ResidualCodecV2::ContextRans => encode_context_rans(residual),
            ResidualCodecV2::Golomb => encode_golomb(residual),
            ResidualCodecV2::CenteredGolomb => encode_centered_golomb(residual),
            ResidualCodecV2::FactorShift => encode_factor_shift(residual),
        }
    }

    /// Decode into a dense residual of exactly `len` values.
    pub fn decode(self, bytes: &[u8], len: usize) -> Result<Vec<i32>> {
        if len as u64 * 4 > crate::limits::MAX_LEARNED_RESIDUAL_BYTES {
            return Err(Error::limit("residual length exceeds the byte bound"));
        }
        match self {
            ResidualCodecV2::DenseI32
            | ResidualCodecV2::SparseDelta
            | ResidualCodecV2::ZigZagVarint
            | ResidualCodecV2::BlockRice
            | ResidualCodecV2::PredictiveRice
            | ResidualCodecV2::LiteralResidual => {
                self.as_v1().expect("v1 codec").decode(bytes, len)
            }
            ResidualCodecV2::PartitionRice => decode_partition_rice(bytes, len),
            ResidualCodecV2::CoreTailRice => decode_core_tail_rice(bytes, len),
            ResidualCodecV2::RunLengthRice => decode_run_length_rice(bytes, len),
            ResidualCodecV2::ZeroMaskRice => decode_zero_mask_rice(bytes, len),
            ResidualCodecV2::BytePlane => decode_byte_plane(bytes, len),
            ResidualCodecV2::ContextRans => decode_context_rans(bytes, len),
            ResidualCodecV2::Golomb => decode_golomb(bytes, len),
            ResidualCodecV2::CenteredGolomb => decode_centered_golomb(bytes, len),
            ResidualCodecV2::FactorShift => decode_factor_shift(bytes, len),
        }
    }
}

impl From<crate::learned::residual_codec::ResidualCodec> for ResidualCodecV2 {
    fn from(c: crate::learned::residual_codec::ResidualCodec) -> ResidualCodecV2 {
        match c {
            crate::learned::residual_codec::ResidualCodec::DenseI32 => ResidualCodecV2::DenseI32,
            crate::learned::residual_codec::ResidualCodec::SparseDelta => {
                ResidualCodecV2::SparseDelta
            }
            crate::learned::residual_codec::ResidualCodec::ZigZagVarint => {
                ResidualCodecV2::ZigZagVarint
            }
            crate::learned::residual_codec::ResidualCodec::BlockRice => ResidualCodecV2::BlockRice,
            crate::learned::residual_codec::ResidualCodec::PredictiveRice => {
                ResidualCodecV2::PredictiveRice
            }
            crate::learned::residual_codec::ResidualCodec::LiteralResidual => {
                ResidualCodecV2::LiteralResidual
            }
        }
    }
}

/// One encoded v2 residual candidate: canonical bytes include the codec id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualEncodingV2 {
    pub codec: ResidualCodecV2,
    /// `[codec_id] || payload`.
    pub bytes: Vec<u8>,
}

impl ResidualEncodingV2 {
    /// The complete stored residual cost (codec id byte + payload).
    pub fn complete_bytes(&self) -> u64 {
        self.bytes.len() as u64
    }

    /// Decode this encoding back to the dense residual.
    pub fn decode(&self, len: usize) -> Result<Vec<i32>> {
        self.codec.decode(&self.bytes[1..], len)
    }
}

/// Encode `residual` with every v3 codec (the whole v2 family plus the Seal S4
/// additions) and return the smallest canonical encoding; ties break by
/// ascending id.
pub fn encode_best_v3(residual: &[i32]) -> ResidualEncodingV2 {
    let mut best: Option<ResidualEncodingV2> = None;
    for codec in ResidualCodecV2::ALL_V3 {
        let payload = codec.encode(residual);
        let mut bytes = Vec::with_capacity(payload.len() + 1);
        bytes.push(codec.id());
        bytes.extend_from_slice(&payload);
        let candidate = ResidualEncodingV2 { codec, bytes };
        match &best {
            None => best = Some(candidate),
            Some(b) if candidate.complete_bytes() < b.complete_bytes() => best = Some(candidate),
            Some(_) => {}
        }
    }
    best.expect("the codec family is non-empty")
}

/// Measure every v3 codec's cost for a residual.
pub fn encode_all_v3(residual: &[i32]) -> Vec<ResidualEncodingV2> {
    ResidualCodecV2::ALL_V3
        .iter()
        .map(|&codec| {
            let payload = codec.encode(residual);
            let mut bytes = Vec::with_capacity(payload.len() + 1);
            bytes.push(codec.id());
            bytes.extend_from_slice(&payload);
            ResidualEncodingV2 { codec, bytes }
        })
        .collect()
}

/// Decode `[codec_id] || payload`, validating the declared length (v3 accepts
/// every v2 id and the Seal S4 additions).
pub fn decode_encoding_v3(bytes: &[u8], len: usize) -> Result<ResidualEncodingV2> {
    decode_encoding_v2(bytes, len)
}

/// The frozen Exp1 minimum (`best_v1`), preserved exactly.
pub fn encode_best_v1(residual: &[i32]) -> ResidualEncodingV2 {
    wrap(crate::learned::residual_codec::encode_best(residual))
}

fn wrap(e: crate::learned::residual_codec::ResidualEncoding) -> ResidualEncodingV2 {
    ResidualEncodingV2 {
        codec: ResidualCodecV2::from(e.codec),
        bytes: e.bytes,
    }
}

/// Encode `residual` with every v2 codec (including the frozen v1 members) and
/// return the smallest canonical encoding; ties break by ascending id.
pub fn encode_best_v2(residual: &[i32]) -> ResidualEncodingV2 {
    let mut best: Option<ResidualEncodingV2> = None;
    for codec in ResidualCodecV2::ALL {
        let payload = codec.encode(residual);
        let mut bytes = Vec::with_capacity(payload.len() + 1);
        bytes.push(codec.id());
        bytes.extend_from_slice(&payload);
        let candidate = ResidualEncodingV2 { codec, bytes };
        match &best {
            None => best = Some(candidate),
            Some(b) if candidate.complete_bytes() < b.complete_bytes() => best = Some(candidate),
            Some(_) => {}
        }
    }
    best.expect("the codec family is non-empty")
}

/// Measure every v2 codec's cost for a residual.
pub fn encode_all_v2(residual: &[i32]) -> Vec<ResidualEncodingV2> {
    ResidualCodecV2::ALL
        .iter()
        .map(|&codec| {
            let payload = codec.encode(residual);
            let mut bytes = Vec::with_capacity(payload.len() + 1);
            bytes.push(codec.id());
            bytes.extend_from_slice(&payload);
            ResidualEncodingV2 { codec, bytes }
        })
        .collect()
}

/// Decode `[codec_id] || payload`, validating the declared length.
pub fn decode_encoding_v2(bytes: &[u8], len: usize) -> Result<ResidualEncodingV2> {
    let (&id, _) = bytes
        .split_first()
        .ok_or_else(|| Error::malformed("empty residual encoding"))?;
    let codec = ResidualCodecV2::from_id(id)
        .ok_or_else(|| Error::new(Kind::Unsupported, format!("unknown residual codec {id}")))?;
    let encoding = ResidualEncodingV2 {
        codec,
        bytes: bytes.to_vec(),
    };
    let dense = encoding.decode(len)?;
    if dense.len() != len {
        return Err(Error::malformed("residual encoding length mismatch"));
    }
    Ok(encoding)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

#[inline]
fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

#[inline]
fn unzigzag(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

fn zigzag_map(residual: &[i32]) -> Vec<u64> {
    residual.iter().map(|&v| zigzag(i64::from(v))).collect()
}

fn to_i32(u: u64) -> Result<i32> {
    let v = unzigzag(u);
    if v < i64::from(i32::MIN) || v > i64::from(i32::MAX) {
        return Err(Error::malformed("residual out of i32 domain"));
    }
    Ok(v as i32)
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

fn uvarint_len(mut v: u64) -> u64 {
    let mut n = 1u64;
    while v >= 0x80 {
        v >>= 7;
        n += 1;
    }
    n
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }
    fn is_empty(&self) -> bool {
        self.remaining() == 0
    }
    fn u8(&mut self) -> Result<u8> {
        let b = *self
            .bytes
            .get(self.pos)
            .ok_or_else(|| Error::malformed("residual stream is truncated"))?;
        self.pos += 1;
        Ok(b)
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| Error::limit("residual read overflows"))?;
        let s = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| Error::malformed("residual stream is truncated"))?;
        self.pos = end;
        Ok(s)
    }
    fn u64le(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn u32le(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32le(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn rest(&self) -> &'a [u8] {
        &self.bytes[self.pos.min(self.bytes.len())..]
    }
    fn uvarint(&mut self) -> Result<u64> {
        let mut v: u64 = 0;
        let mut shift = 0u32;
        loop {
            let b = self.u8()?;
            if shift >= 64 {
                return Err(Error::malformed("residual varint overflows u64"));
            }
            v |= u64::from(b & 0x7F) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
        }
    }
    fn finish(&self) -> Result<()> {
        if self.is_empty() {
            Ok(())
        } else {
            Err(Error::malformed("residual stream has trailing bytes"))
        }
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
    fn bits(&mut self, value: u64, n: u32) {
        for i in (0..n).rev() {
            self.bit((value >> i) & 1 == 1);
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
            return Err(Error::malformed("bit stream is truncated"));
        }
        let b = (self.bytes[byte] >> (7 - (self.bit & 7))) & 1;
        self.bit += 1;
        Ok(b == 1)
    }
    fn read_bits(&mut self, n: u32) -> Result<u64> {
        if n > 64 {
            return Err(Error::malformed("bit read exceeds u64"));
        }
        let mut v = 0u64;
        for _ in 0..n {
            v = (v << 1) | u64::from(self.read_bit()?);
        }
        Ok(v)
    }
}

/// Choose the Rice parameter minimizing the bit cost and guaranteeing that the
/// encoder and decoder agree on the decodability floor.
fn best_rice_k(values: &[u64]) -> (u32, u64) {
    if values.is_empty() {
        return (0, 0);
    }
    let max_u = values.iter().copied().max().unwrap_or(0);
    let mut floor_k = 0u32;
    while floor_k < MAX_RICE_K2 && (max_u >> floor_k) > MAX_RICE_UNARY2 {
        floor_k += 1;
    }
    let mut best_k = floor_k;
    let mut best_bits = u64::MAX;
    for k in floor_k..=MAX_RICE_K2 {
        let bits: u64 = values
            .iter()
            .map(|&u| (u >> k).saturating_add(1).saturating_add(u64::from(k)))
            .sum();
        if bits < best_bits {
            best_bits = bits;
            best_k = k;
        }
    }
    (best_k, best_bits)
}

fn write_rice(w: &mut BitWriter, values: &[u64], k: u32) {
    for &u in values {
        let q = u >> k;
        for _ in 0..q {
            w.bit(false);
        }
        w.bit(true);
        if k > 0 {
            w.bits(u & ((1u64 << k) - 1), k);
        }
    }
}

fn read_rice(r: &mut BitReader<'_>, n: usize, k: u32) -> Result<Vec<u64>> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut q: u64 = 0;
        while !r.read_bit()? {
            q += 1;
            if q > MAX_RICE_UNARY2 {
                return Err(Error::limit("rice unary run exceeds the bound"));
            }
        }
        let rem = if k > 0 { r.read_bits(k)? } else { 0 };
        let u = q
            .checked_shl(k)
            .ok_or_else(|| Error::malformed("rice value overflows"))?
            | rem;
        out.push(u);
    }
    Ok(out)
}

/// Exp-Golomb order-0: `m` zero bits, a one bit, then `m` value bits of `x+1`.
fn eg0_bits(x: u64) -> u64 {
    let m = 63 - (x + 1).leading_zeros();
    u64::from(m) + 1 + u64::from(m)
}

fn write_eg0(w: &mut BitWriter, x: u64) {
    let m = 63 - (x + 1).leading_zeros();
    for _ in 0..m {
        w.bit(false);
    }
    w.bit(true);
    if m > 0 {
        w.bits(x + 1, m);
    }
}

fn read_eg0(r: &mut BitReader<'_>) -> Result<u64> {
    let mut m = 0u32;
    while !r.read_bit()? {
        m += 1;
        if m > MAX_EG0_BITS {
            return Err(Error::limit("exp-golomb run exceeds the bound"));
        }
    }
    if m == 0 {
        return Ok(0);
    }
    let v = r.read_bits(m)?;
    // `x + 1` has exactly `m+1` bits, so the top bit is implicit.
    let x = (v | (1u64 << m)) - 1;
    Ok(x)
}

// ---------------------------------------------------------------------------
// id 6: PartitionRice
// ---------------------------------------------------------------------------

/// Partitioned block-adaptive Rice over a frozen length ladder.
///
/// Segmentation is a deterministic shortest path over aligned boundaries,
/// minimizing the actual stored size: each partition pays its length varint, its
/// Rice parameter byte and its Rice bits. Ties prefer fewer partitions, then
/// shorter partitions.
fn encode_partition_rice(residual: &[i32]) -> Vec<u8> {
    let n = residual.len();
    let mapped = zigzag_map(residual);
    let mut out = Vec::new();
    out.extend_from_slice(&(n as u64).to_le_bytes());
    if n == 0 {
        out.extend_from_slice(&0u32.to_le_bytes());
        return out;
    }
    let unit = PARTITION_UNIT;
    // Node i (0..=nodes-1) is position i*unit; node `nodes` is position n.
    let nodes = n.div_ceil(unit);
    // dp[i] = (total bit+header cost, partition count, chosen length)
    let mut dp_cost = vec![u64::MAX; nodes + 1];
    let mut dp_parts = vec![u64::MAX; nodes + 1];
    let mut choice = vec![0usize; nodes];
    dp_cost[nodes] = 0;
    dp_parts[nodes] = 0;
    for i in (0..nodes).rev() {
        let p = i * unit;
        if p >= n {
            continue;
        }
        // Incremental nested scan: extend by `unit` for each ladder rung.
        let mut sums = vec![0u64; (MAX_RICE_K2 + 1) as usize];
        let mut count = 0u64;
        let mut max_u = 0u64;
        let mut best_cost = u64::MAX;
        let mut best_parts = u64::MAX;
        let mut best_len = 0usize;
        // Candidate lengths: the ladder rungs that fit, plus the exact tail.
        for rung in PARTITION_LADDER {
            let end = p + rung;
            if end > n {
                break;
            }
            // Extend the accumulator from `p + count` to `end`.
            while (p as u64 + count) < end as u64 {
                let u = mapped[p + count as usize];
                max_u = max_u.max(u);
                for (k, s) in sums.iter_mut().enumerate() {
                    *s += u >> k;
                }
                count += 1;
            }
            let (cost, parts) = partition_edge_cost(&sums, count, max_u);
            let next = if end == n { nodes } else { end / unit };
            if dp_cost[next] == u64::MAX {
                continue;
            }
            let total = cost
                .saturating_add(uvarint_len(rung as u64))
                .saturating_add(1)
                .saturating_add(dp_cost[next]);
            let total_parts = parts.saturating_add(dp_parts[next]).saturating_add(1);
            if total < best_cost || (total == best_cost && total_parts < best_parts) {
                best_cost = total;
                best_parts = total_parts;
                best_len = rung;
            }
        }
        // Exact tail partition (covers `n` not aligned to the unit).
        let tail = n - p;
        if tail <= *PARTITION_LADDER.last().unwrap() && !tail.is_multiple_of(unit) {
            // Recompute the tail accumulator from scratch (it was not scanned).
            let mut sums_t = vec![0u64; (MAX_RICE_K2 + 1) as usize];
            let mut max_ut = 0u64;
            for &u in &mapped[p..n] {
                max_ut = max_ut.max(u);
                for (k, s) in sums_t.iter_mut().enumerate() {
                    *s += u >> k;
                }
            }
            let (cost, parts) = partition_edge_cost(&sums_t, tail as u64, max_ut);
            let total = cost
                .saturating_add(uvarint_len(tail as u64))
                .saturating_add(1)
                .saturating_add(dp_cost[nodes]);
            let total_parts = parts.saturating_add(1);
            if total < best_cost || (total == best_cost && total_parts < best_parts) {
                best_cost = total;
                best_parts = total_parts;
                best_len = tail;
            }
        }
        dp_cost[i] = best_cost;
        dp_parts[i] = best_parts;
        choice[i] = best_len;
    }
    // Reconstruct partitions.
    let mut partitions: Vec<(usize, usize)> = Vec::new();
    let mut i = 0usize;
    while i < nodes {
        let p = i * unit;
        let l = choice[i];
        if l == 0 {
            // Should not happen; fall back to the whole tail.
            partitions.push((p, n));
            break;
        }
        partitions.push((p, (p + l).min(n)));
        if p + l >= n {
            break;
        }
        i = (p + l) / unit;
    }
    out.extend_from_slice(&(partitions.len() as u32).to_le_bytes());
    let mut w = BitWriter::new();
    for &(a, b) in &partitions {
        put_uvarint(&mut out, (b - a) as u64);
        let (k, _) = best_rice_k(&mapped[a..b]);
        out.push(k as u8);
        write_rice(&mut w, &mapped[a..b], k);
    }
    out.extend_from_slice(&w.finish());
    out
}

/// Rice bit cost for a scanned block given per-`k` prefix sums.
fn partition_edge_cost(sums: &[u64], count: u64, max_u: u64) -> (u64, u64) {
    let mut floor_k = 0u32;
    while floor_k < MAX_RICE_K2 && (max_u >> floor_k) > MAX_RICE_UNARY2 {
        floor_k += 1;
    }
    let mut best = u64::MAX;
    for k in floor_k..=MAX_RICE_K2 {
        let bits = sums[k as usize]
            .saturating_add(count)
            .saturating_add(count.saturating_mul(u64::from(k)));
        if bits < best {
            best = bits;
        }
    }
    (best, 1)
}

fn decode_partition_rice(bytes: &[u8], len: usize) -> Result<Vec<i32>> {
    let mut r = Reader::new(bytes);
    let total = r.u64le()? as usize;
    if total != len {
        return Err(Error::malformed("partition rice length mismatch"));
    }
    let part_count = r.take(4)?;
    let part_count = u32::from_le_bytes(part_count.try_into().unwrap()) as usize;
    if part_count > len.max(1) {
        return Err(Error::limit("partition count exceeds the residual length"));
    }
    let mut parts: Vec<(usize, u32)> = Vec::with_capacity(part_count);
    let mut sum = 0usize;
    for _ in 0..part_count {
        let l = r.uvarint()? as usize;
        let k = r.u8()? as u32;
        if k > MAX_RICE_K2 {
            return Err(Error::malformed("partition rice parameter out of range"));
        }
        sum = sum
            .checked_add(l)
            .ok_or_else(|| Error::limit("partition lengths overflow"))?;
        if sum > len {
            return Err(Error::malformed("partition lengths exceed the residual"));
        }
        parts.push((l, k));
    }
    if sum != len {
        return Err(Error::malformed(
            "partition lengths do not sum to the residual",
        ));
    }
    let body = r.take(r.remaining())?;
    let mut br = BitReader::new(body);
    let mut out = Vec::with_capacity(len);
    for (l, k) in parts {
        let vals = read_rice(&mut br, l, k)?;
        for u in vals {
            out.push(to_i32(u)?);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// id 7: CoreTailRice
// ---------------------------------------------------------------------------

/// Two-regime Rice: a dense core with an Exp-Golomb escape for the tail.
fn core_tail_bits(values: &[u64], k: u32, e: u32) -> u64 {
    let esc = 1u64 << e;
    let mut bits = 0u64;
    for &u in values {
        let q = u >> k;
        if q < esc {
            bits = bits.saturating_add(q).saturating_add(1);
        } else {
            bits = bits.saturating_add(esc).saturating_add(1);
            bits = bits.saturating_add(eg0_bits(q - esc));
        }
        bits = bits.saturating_add(u64::from(k));
    }
    bits
}

fn encode_core_tail_rice(residual: &[i32]) -> Vec<u8> {
    let n = residual.len();
    let mapped = zigzag_map(residual);
    let mut out = Vec::new();
    out.extend_from_slice(&(n as u64).to_le_bytes());
    if n == 0 {
        out.push(0);
        out.push(0);
        return out;
    }
    let mut best = (0u32, 0u32, u64::MAX);
    for e in 0..=8u32 {
        for k in 0..=MAX_RICE_K2 {
            let bits = core_tail_bits(&mapped, k, e);
            if bits < best.2 {
                best = (k, e, bits);
            }
        }
    }
    let (k, e, _) = best;
    out.push(k as u8);
    out.push(e as u8);
    let esc = 1u64 << e;
    let mut w = BitWriter::new();
    for &u in &mapped {
        let q = u >> k;
        if q < esc {
            for _ in 0..q {
                w.bit(false);
            }
            w.bit(true);
        } else {
            for _ in 0..esc {
                w.bit(false);
            }
            w.bit(true);
            write_eg0(&mut w, q - esc);
        }
        if k > 0 {
            w.bits(u & ((1u64 << k) - 1), k);
        }
    }
    out.extend_from_slice(&w.finish());
    out
}

fn decode_core_tail_rice(bytes: &[u8], len: usize) -> Result<Vec<i32>> {
    let mut r = Reader::new(bytes);
    let total = r.u64le()? as usize;
    if total != len {
        return Err(Error::malformed("core/tail rice length mismatch"));
    }
    let k = r.u8()? as u32;
    let e = r.u8()? as u32;
    if k > MAX_RICE_K2 || e > 8 {
        return Err(Error::malformed("core/tail parameters out of range"));
    }
    let body = r.take(r.remaining())?;
    let mut br = BitReader::new(body);
    let esc = 1u64 << e;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let mut q: u64 = 0;
        while !br.read_bit()? {
            q += 1;
            if q > esc {
                return Err(Error::limit("core/tail unary run exceeds the escape"));
            }
        }
        let q = if q == esc {
            esc + read_eg0(&mut br)?
        } else {
            q
        };
        let rem = if k > 0 { br.read_bits(k)? } else { 0 };
        let u = q
            .checked_shl(k)
            .ok_or_else(|| Error::malformed("core/tail value overflows"))?
            | rem;
        out.push(to_i32(u)?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// id 8: RunLengthRice (adaptive, decoder-visible state only)
// ---------------------------------------------------------------------------

const RL_INIT_K: u32 = 2;

/// Escape threshold of the adaptive run-length value code.
const RL_ESC: u64 = 1 << 16;

fn rl_adapt_k(k: u32, m: u64) -> u32 {
    if m < (1u64 << k.min(40)) {
        k.saturating_sub(1)
    } else {
        (k + 1).min(MAX_RICE_K2)
    }
}

/// Adaptive run-length/Rice over zero runs (`O.10`, RLGR family).
///
/// Zero runs are coded with Exp-Golomb; nonzero magnitudes with a Rice parameter
/// adapted from previously decoded values only. No probability model is
/// transmitted and no hidden state exists.
fn encode_run_length_rice(residual: &[i32]) -> Vec<u8> {
    let n = residual.len();
    let mut out = Vec::new();
    out.extend_from_slice(&(n as u64).to_le_bytes());
    let mut w = BitWriter::new();
    let mut k = RL_INIT_K;
    let mut i = 0usize;
    while i < n {
        let mut d = 0u64;
        while i + (d as usize) < n && residual[i + d as usize] == 0 {
            d += 1;
            if d > MAX_RUN_CHUNK {
                break;
            }
        }
        write_eg0(&mut w, d);
        i += d as usize;
        if i >= n {
            break;
        }
        let v = residual[i];
        let m = v.unsigned_abs() as u64;
        let q = m >> k;
        if q < RL_ESC {
            for _ in 0..q {
                w.bit(false);
            }
            w.bit(true);
        } else {
            for _ in 0..RL_ESC {
                w.bit(false);
            }
            w.bit(true);
            write_eg0(&mut w, q - RL_ESC);
        }
        if k > 0 {
            w.bits(m & ((1u64 << k) - 1), k);
        }
        w.bit(v < 0);
        k = rl_adapt_k(k, m);
        i += 1;
    }
    out.extend_from_slice(&w.finish());
    out
}

fn decode_run_length_rice(bytes: &[u8], len: usize) -> Result<Vec<i32>> {
    let mut r = Reader::new(bytes);
    let total = r.u64le()? as usize;
    if total != len {
        return Err(Error::malformed("run-length rice length mismatch"));
    }
    let body = r.take(r.remaining())?;
    let mut br = BitReader::new(body);
    let mut out = Vec::with_capacity(len);
    let mut k = RL_INIT_K;
    while out.len() < len {
        let d = read_eg0(&mut br)? as usize;
        if d > MAX_RUN_CHUNK as usize || out.len() + d > len {
            return Err(Error::limit("run-length exceeds the residual"));
        }
        out.extend(std::iter::repeat_n(0i32, d));
        if out.len() >= len {
            break;
        }
        let mut q: u64 = 0;
        while !br.read_bit()? {
            q += 1;
            if q > RL_ESC {
                return Err(Error::limit("run-length rice unary exceeds the bound"));
            }
        }
        let q = if q == RL_ESC {
            RL_ESC + read_eg0(&mut br)?
        } else {
            q
        };
        let rem = if k > 0 { br.read_bits(k)? } else { 0 };
        let m = q
            .checked_shl(k)
            .ok_or_else(|| Error::malformed("run-length magnitude overflows"))?
            | rem;
        if m == 0 {
            return Err(Error::malformed("run-length nonzero magnitude is zero"));
        }
        if m > i64::from(i32::MAX) as u64 + 1 {
            return Err(Error::malformed("run-length magnitude out of i32 domain"));
        }
        let neg = br.read_bit()?;
        let v = if neg { -(m as i64) } else { m as i64 };
        if v < i64::from(i32::MIN) || v > i64::from(i32::MAX) {
            return Err(Error::malformed("run-length value out of i32 domain"));
        }
        out.push(v as i32);
        k = rl_adapt_k(k, m);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// id 9: ZeroMaskRice
// ---------------------------------------------------------------------------

const MASK_RAW: u8 = 0;
const MASK_RLE: u8 = 1;
const MASK_SPARSE: u8 = 2;
const MAG_RICE: u8 = 0;
const MAG_VARINT: u8 = 1;

fn encode_mask_raw(flags: &[bool]) -> Vec<u8> {
    let mut w = BitWriter::new();
    for &f in flags {
        w.bit(f);
    }
    w.finish()
}

fn decode_mask_raw(bytes: &[u8], len: usize) -> Result<Vec<bool>> {
    let mut br = BitReader::new(bytes);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(br.read_bit()?);
    }
    Ok(out)
}

fn encode_mask_rle(flags: &[bool]) -> Vec<u8> {
    // Alternating run lengths starting with a zero-run.
    let mut w = BitWriter::new();
    let mut i = 0usize;
    let n = flags.len();
    let mut expect_zero = true;
    while i < n {
        let mut d = 0u64;
        while i + (d as usize) < n && flags[i + d as usize] == !expect_zero {
            d += 1;
        }
        write_eg0(&mut w, d);
        i += d as usize;
        expect_zero = !expect_zero;
    }
    w.finish()
}

fn decode_mask_rle(bytes: &[u8], len: usize) -> Result<Vec<bool>> {
    let mut br = BitReader::new(bytes);
    let mut out = Vec::with_capacity(len);
    let mut expect_zero = true;
    while out.len() < len {
        let d = read_eg0(&mut br)? as usize;
        if out.len() + d > len {
            return Err(Error::limit("mask run exceeds the residual"));
        }
        out.extend(std::iter::repeat_n(!expect_zero, d));
        expect_zero = !expect_zero;
    }
    Ok(out)
}

fn encode_mask_sparse(flags: &[bool]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut prev: Option<u64> = None;
    for (i, &f) in flags.iter().enumerate() {
        if f {
            let idx = i as u64;
            let delta = match prev {
                None => idx + 1,
                Some(p) => idx - p,
            };
            put_uvarint(&mut out, delta);
            prev = Some(idx);
        }
    }
    out
}

fn decode_mask_sparse(bytes: &[u8], len: usize) -> Result<Vec<bool>> {
    let mut r = Reader::new(bytes);
    let mut out = vec![false; len];
    let mut prev: Option<u64> = None;
    while !r.is_empty() {
        let delta = r.uvarint()?;
        if delta == 0 {
            return Err(Error::malformed("sparse mask indices must ascend"));
        }
        let idx = match prev {
            None => delta - 1,
            Some(p) => p
                .checked_add(delta)
                .ok_or_else(|| Error::malformed("sparse mask index overflows"))?,
        };
        if idx >= len as u64 {
            return Err(Error::malformed("sparse mask index out of range"));
        }
        out[idx as usize] = true;
        prev = Some(idx);
    }
    Ok(out)
}

fn encode_zero_mask_rice(residual: &[i32]) -> Vec<u8> {
    let n = residual.len();
    let mut out = Vec::new();
    out.extend_from_slice(&(n as u64).to_le_bytes());
    let flags: Vec<bool> = residual.iter().map(|&v| v != 0).collect();
    let candidates: [(u8, Vec<u8>); 3] = [
        (MASK_RAW, encode_mask_raw(&flags)),
        (MASK_RLE, encode_mask_rle(&flags)),
        (MASK_SPARSE, encode_mask_sparse(&flags)),
    ];
    let (mask_mode, mask_bytes) = candidates
        .iter()
        .min_by_key(|(m, b)| (b.len() as u64, *m))
        .map(|(m, b)| (*m, b.clone()))
        .expect("non-empty candidate set");
    out.push(mask_mode);
    put_uvarint(&mut out, mask_bytes.len() as u64);
    out.extend_from_slice(&mask_bytes);

    let mags: Vec<u64> = residual
        .iter()
        .filter(|&&v| v != 0)
        .map(|&v| zigzag(i64::from(v)))
        .collect();
    let (k, rice_bits) = best_rice_k(&mags);
    let mut rice_w = BitWriter::new();
    write_rice(&mut rice_w, &mags, k);
    let rice_payload = rice_w.finish();
    // Header: k byte + bit length.
    let rice_len = 1u64 + rice_bits.div_ceil(8);
    let mut varint = Vec::new();
    for &m in &mags {
        put_uvarint(&mut varint, m);
    }
    let varint_len = varint.len() as u64;
    if rice_len <= varint_len {
        out.push(MAG_RICE);
        put_uvarint(&mut out, rice_len);
        out.push(k as u8);
        out.extend_from_slice(&rice_payload);
    } else {
        out.push(MAG_VARINT);
        put_uvarint(&mut out, varint_len);
        out.extend_from_slice(&varint);
    }
    out
}

fn decode_zero_mask_rice(bytes: &[u8], len: usize) -> Result<Vec<i32>> {
    let mut r = Reader::new(bytes);
    let total = r.u64le()? as usize;
    if total != len {
        return Err(Error::malformed("zero-mask rice length mismatch"));
    }
    let mask_mode = r.u8()?;
    let mask_len = r.uvarint()?;
    let mask_len = usize::try_from(mask_len)
        .map_err(|_| Error::limit("zero-mask length exceeds host usize"))?;
    if mask_len > len.div_ceil(8) + 16 + len {
        return Err(Error::limit("zero-mask payload exceeds the residual bound"));
    }
    let mask_bytes = r.take(mask_len)?;
    let flags = match mask_mode {
        MASK_RAW => decode_mask_raw(mask_bytes, len)?,
        MASK_RLE => decode_mask_rle(mask_bytes, len)?,
        MASK_SPARSE => decode_mask_sparse(mask_bytes, len)?,
        _ => return Err(Error::malformed("unknown zero-mask mode")),
    };
    if flags.len() != len {
        return Err(Error::malformed("zero-mask length mismatch"));
    }
    let nonzero = flags.iter().filter(|&&f| f).count();
    let mag_mode = r.u8()?;
    let mag_len = r.uvarint()?;
    let mag_len = usize::try_from(mag_len)
        .map_err(|_| Error::limit("zero-mask magnitude length exceeds host usize"))?;
    let mag_bytes = r.take(mag_len)?;
    r.finish()?;
    let mags: Vec<u64> = match mag_mode {
        MAG_RICE => {
            if mag_bytes.is_empty() {
                return Err(Error::malformed("zero-mask rice payload is empty"));
            }
            let k = mag_bytes[0] as u32;
            if k > MAX_RICE_K2 {
                return Err(Error::malformed("zero-mask rice parameter out of range"));
            }
            let mut br = BitReader::new(&mag_bytes[1..]);
            read_rice(&mut br, nonzero, k)?
        }
        MAG_VARINT => {
            let mut rr = Reader::new(mag_bytes);
            let mut v = Vec::with_capacity(nonzero);
            for _ in 0..nonzero {
                v.push(rr.uvarint()?);
            }
            rr.finish()?;
            v
        }
        _ => return Err(Error::malformed("unknown zero-mask magnitude mode")),
    };
    if mags.len() != nonzero {
        return Err(Error::malformed("zero-mask magnitude count mismatch"));
    }
    let mut out = Vec::with_capacity(len);
    let mut it = mags.into_iter();
    for &f in &flags {
        if f {
            let u = it.next().expect("count checked");
            out.push(to_i32(u)?);
        } else {
            out.push(0);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// id 10: BytePlane
// ---------------------------------------------------------------------------

/// Byte-length plane + significance byte planes + sign stream.
fn byte_count(u: u64) -> u8 {
    let bits = 64 - u.leading_zeros();
    bits.div_ceil(8) as u8
}

fn encode_byte_plane(residual: &[i32]) -> Vec<u8> {
    let n = residual.len();
    // Magnitudes and signs are stored in separate planes (zigzag is *not*
    // applied here, or the sign would be counted twice).
    let mags: Vec<u64> = residual
        .iter()
        .map(|&v| u64::from(v.unsigned_abs()))
        .collect();
    let mut out = Vec::new();
    out.extend_from_slice(&(n as u64).to_le_bytes());
    let mut lens = BitWriter::new();
    for &m in &mags {
        let b = byte_count(m);
        lens.bits(u64::from(b), 4);
    }
    out.extend_from_slice(&lens.finish());
    // Sign stream: packed bits for every nonzero value.
    let signs: Vec<bool> = residual.iter().map(|&v| v < 0).collect();
    let sign_bits: Vec<u8> = {
        let mut w = BitWriter::new();
        for &s in &signs {
            w.bit(s);
        }
        w.finish()
    };
    out.extend_from_slice(&sign_bits);
    for plane in 0..8u32 {
        let mut w = BitWriter::new();
        for &m in &mags {
            let b = u32::from(byte_count(m));
            if b > plane {
                let byte = ((m >> (8 * plane)) & 0xFF) as u8;
                w.bits(u64::from(byte), 8);
            }
        }
        out.extend_from_slice(&w.finish());
    }
    out
}

fn decode_byte_plane(bytes: &[u8], len: usize) -> Result<Vec<i32>> {
    let mut r = Reader::new(bytes);
    let total = r.u64le()? as usize;
    if total != len {
        return Err(Error::malformed("byte-plane length mismatch"));
    }
    let lens_len = len.div_ceil(2);
    let lens_bytes = r.take(lens_len)?;
    let mut lr = BitReader::new(lens_bytes);
    let mut lens = Vec::with_capacity(len);
    for _ in 0..len {
        lens.push(lr.read_bits(4)? as u8);
    }
    let sign_len = len.div_ceil(8);
    let sign_bytes = r.take(sign_len)?.to_vec();
    let mut sr = BitReader::new(&sign_bytes);
    let mut signs = Vec::with_capacity(len);
    for _ in 0..len {
        signs.push(sr.read_bit()?);
    }
    let mut values = vec![0u64; len];
    for plane in 0..8u32 {
        let count = lens.iter().filter(|&&b| u32::from(b) > plane).count();
        let bytes_needed = count;
        let plane_bytes = r.take(bytes_needed)?;
        let mut idx = 0usize;
        for i in 0..len {
            if u32::from(lens[i]) > plane {
                let byte = plane_bytes[idx];
                idx += 1;
                values[i] |= u64::from(byte) << (8 * plane);
            }
        }
    }
    r.finish()?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        if lens[i] == 0 {
            out.push(0);
            continue;
        }
        let m = values[i];
        let v = if signs[i] { -(m as i64) } else { m as i64 };
        if v < i64::from(i32::MIN) || v > i64::from(i32::MAX) {
            return Err(Error::malformed("byte-plane value out of i32 domain"));
        }
        out.push(v as i32);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// id 11: ContextRans
// ---------------------------------------------------------------------------

fn bucket_of(u: u64) -> u8 {
    if u == 0 {
        0
    } else {
        (64 - u.leading_zeros()).min(CTX_ALPHABET as u32 - 1) as u8
    }
}

fn context_of(prev: u8) -> usize {
    match prev {
        0 => 0,
        1..=4 => 1,
        5..=12 => 2,
        _ => 3,
    }
}

/// Normalize counts into frequencies summing exactly to `total`, with a
/// non-zero frequency for every observed symbol.
fn normalize_counts(counts: &[u64], total: u32) -> Vec<u16> {
    let k = counts.len();
    let c_total: u64 = counts.iter().sum();
    if c_total == 0 {
        let mut f = vec![0u16; k];
        f[0] = total as u16;
        return f;
    }
    let mut f = vec![0u16; k];
    let mut frac = vec![0f64; k];
    for s in 0..k {
        if counts[s] == 0 {
            continue;
        }
        let ideal = counts[s] as f64 * f64::from(total) / c_total as f64;
        let fl = ideal.floor().max(1.0);
        f[s] = fl as u16;
        frac[s] = ideal - fl;
    }
    let mut sum: i64 = f.iter().map(|&x| i64::from(x)).sum();
    let target = i64::from(total);
    // Adjust deterministically: add to the largest fractional parts, remove
    // from the smallest among symbols with room.
    while sum < target {
        let mut best = usize::MAX;
        let mut best_f = f64::NEG_INFINITY;
        for s in 0..k {
            if counts[s] == 0 {
                continue;
            }
            if frac[s] > best_f {
                best_f = frac[s];
                best = s;
            }
        }
        if best == usize::MAX {
            break;
        }
        f[best] += 1;
        frac[best] -= 1.0;
        sum += 1;
    }
    while sum > target {
        let mut best = usize::MAX;
        let mut best_f = f64::INFINITY;
        for s in 0..k {
            if counts[s] == 0 || f[s] <= 1 {
                continue;
            }
            if frac[s] < best_f {
                best_f = frac[s];
                best = s;
            }
        }
        if best == usize::MAX {
            break;
        }
        f[best] -= 1;
        frac[best] += 1.0;
        sum -= 1;
    }
    f
}

fn encode_context_rans(residual: &[i32]) -> Vec<u8> {
    let n = residual.len();
    let mapped = zigzag_map(residual);
    let buckets: Vec<u8> = mapped.iter().map(|&u| bucket_of(u)).collect();
    let mut out = Vec::new();
    out.extend_from_slice(&(n as u64).to_le_bytes());
    out.push(CTX_CONTEXTS as u8);
    if n == 0 {
        for _ in 0..CTX_CONTEXTS {
            for s in 0..CTX_ALPHABET {
                let f: u16 = if s == 0 { CTX_MODEL_TOTAL as u16 } else { 0 };
                out.extend_from_slice(&f.to_le_bytes());
            }
        }
        put_uvarint(&mut out, 0);
        return out;
    }
    // Context-conditioned counts.
    let mut counts = vec![vec![0u64; CTX_ALPHABET]; CTX_CONTEXTS];
    let mut prev = 0u8;
    for &b in &buckets {
        counts[context_of(prev)][b as usize] += 1;
        prev = b;
    }
    let mut tables = Vec::with_capacity(CTX_CONTEXTS);
    for row in &counts {
        tables.push(normalize_counts(row, CTX_MODEL_TOTAL));
    }
    for t in &tables {
        for &f in t {
            out.extend_from_slice(&f.to_le_bytes());
        }
    }
    // Cumulative starts.
    let starts: Vec<Vec<u32>> = tables
        .iter()
        .map(|t| {
            let mut v = vec![0u32; CTX_ALPHABET];
            let mut acc = 0u32;
            for s in 0..CTX_ALPHABET {
                v[s] = acc;
                acc += u32::from(t[s]);
            }
            v
        })
        .collect();
    let mut buf = vec![0u8; encode_capacity(n)];
    // Contexts are computed in forward order (each depends on the previous
    // symbol); rANS encodes in reverse of the decode order, and the decoder
    // reads forward, so the encoder iterates the symbols in reverse.
    let mut ctxs = Vec::with_capacity(n);
    let mut prev = 0u8;
    for &b in &buckets {
        ctxs.push(context_of(prev));
        prev = b;
    }
    let written = {
        let mut sink = BackSink::new(&mut buf);
        let mut state = RansState::new();
        for i in (0..n).rev() {
            let ctx = ctxs[i];
            let b = buckets[i];
            let start = starts[ctx][b as usize];
            let freq = u32::from(tables[ctx][b as usize]);
            if freq == 0 || !enc_put(&mut state, &mut sink, start, freq, CTX_SCALE_BITS) {
                // Should be unreachable when normalization is correct; emit a
                // canonical empty stream and let `encode_best_v2` ignore it.
                out.push(0xFF);
                put_uvarint(&mut out, 0);
                return out;
            }
        }
        if !enc_flush(&state, &mut sink) {
            out.push(0xFF);
            put_uvarint(&mut out, 0);
            return out;
        }
        sink.bytes_written()
    };
    let pos = buf.len() - written;
    let stream = &buf[pos..];
    put_uvarint(&mut out, stream.len() as u64);
    out.extend_from_slice(stream);
    // Low bits: for every nonzero bucket, the `bucket - 1` low bits of `u`.
    let mut w = BitWriter::new();
    for (&b, &u) in buckets.iter().zip(mapped.iter()) {
        if b > 0 {
            w.bits(u & ((1u64 << (b - 1)) - 1), u32::from(b - 1));
        }
    }
    out.extend_from_slice(&w.finish());
    out
}

fn decode_context_rans(bytes: &[u8], len: usize) -> Result<Vec<i32>> {
    let mut r = Reader::new(bytes);
    let total = r.u64le()? as usize;
    if total != len {
        return Err(Error::malformed("context-rans length mismatch"));
    }
    if len == 0 {
        return Ok(Vec::new());
    }
    let nctx = r.u8()? as usize;
    if nctx == 0 || nctx > 16 {
        return Err(Error::malformed("context-rans context count out of range"));
    }
    let mut tables = vec![vec![0u16; CTX_ALPHABET]; nctx];
    for row in tables.iter_mut() {
        let mut sum = 0u32;
        for slot in row.iter_mut() {
            *slot = r.u16()?;
            sum += u32::from(*slot);
        }
        if sum != CTX_MODEL_TOTAL {
            return Err(Error::malformed(
                "context-rans model does not sum to the total",
            ));
        }
    }
    let stream_len = r.uvarint()?;
    let stream_len = usize::try_from(stream_len)
        .map_err(|_| Error::limit("context-rans stream length exceeds host usize"))?;
    if stream_len > encode_capacity(len) {
        return Err(Error::limit("context-rans stream exceeds the bound"));
    }
    let stream = r.take(stream_len)?;
    let low_bytes = r.take(r.remaining())?;
    let starts: Vec<Vec<u32>> = tables
        .iter()
        .map(|t| {
            let mut v = vec![0u32; CTX_ALPHABET];
            let mut acc = 0u32;
            for s in 0..CTX_ALPHABET {
                v[s] = acc;
                acc += u32::from(t[s]);
            }
            v
        })
        .collect();
    let mut reader = FwdReader::new(stream);
    let mut state = dec_init(&mut reader)
        .ok_or_else(|| Error::malformed("context-rans stream has no valid initial state"))?;
    let mut buckets = Vec::with_capacity(len);
    let mut prev = 0u8;
    for _ in 0..len {
        let ctx = context_of(prev);
        let slot = crate::entropy::rans::dec_slot(&state, CTX_SCALE_BITS);
        let t = &tables[ctx];
        let mut sym = usize::MAX;
        for s in 0..CTX_ALPHABET {
            let f = u32::from(t[s]);
            if f == 0 {
                continue;
            }
            let st = starts[ctx][s];
            if slot >= st && slot < st + f {
                sym = s;
                break;
            }
        }
        if sym == usize::MAX {
            return Err(Error::malformed("context-rans slot has no symbol"));
        }
        if !dec_advance(
            &mut state,
            &mut reader,
            starts[ctx][sym],
            u32::from(t[sym]),
            CTX_SCALE_BITS,
        ) {
            return Err(Error::malformed("context-rans stream is truncated"));
        }
        buckets.push(sym as u8);
        prev = sym as u8;
    }
    // Low bits.
    let mut br = BitReader::new(low_bytes);
    let mut out = Vec::with_capacity(len);
    for &b in &buckets {
        if b == 0 {
            out.push(0);
            continue;
        }
        let low = br.read_bits(u32::from(b - 1))?;
        let u = (1u64 << (b - 1)) | low;
        out.push(to_i32(u)?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// id 12: Golomb (partitioned, arbitrary non-power-of-two M)
// id 13: CenteredGolomb (partitioned, with a stored residual center)
// ---------------------------------------------------------------------------

/// Frozen partition size of the Golomb codecs (samples per partition).
const GOLOMB_CHUNK: usize = 512;

/// Upper bound on a Golomb divisor considered by the encoder search.
const MAX_GOLOMB_M: u64 = 1 << 20;

#[inline]
fn zigzag_i64(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

#[inline]
fn unzigzag_i64(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

/// Bit cost of coding `values` with divisor `m` (unary quotient + truncated
/// binary remainder).
fn golomb_cost_bits(values: &[u64], m: u64) -> u64 {
    let m = m.max(1);
    let b = 63 - m.leading_zeros() as u64; // floor(log2 m)
    let t = (1u64 << (b + 1)) - m;
    let mut bits = 0u64;
    for &u in values {
        let q = u / m;
        let r = u % m;
        let extra = if r < t { b } else { b + 1 };
        bits = bits
            .saturating_add(q)
            .saturating_add(1)
            .saturating_add(extra);
    }
    bits
}

/// Choose the Golomb divisor minimizing the bit cost, bounded so no unary run is
/// pathological.
fn best_golomb_m(values: &[u64]) -> u64 {
    if values.is_empty() {
        return 1;
    }
    let n = values.len() as u64;
    let sum: u64 = values.iter().sum();
    let mean = (sum / n).max(1);
    let max_u = *values.iter().max().unwrap_or(&0);
    let lo = (max_u / MAX_RICE_UNARY2).max(1);
    let hi = mean.saturating_mul(3).max(lo).min(MAX_GOLOMB_M);
    let mut best_m = lo;
    let mut best = u64::MAX;
    let mut m = lo;
    while m <= hi {
        let c = golomb_cost_bits(values, m);
        if c < best {
            best = c;
            best_m = m;
        }
        m += 1;
    }
    best_m
}

fn write_golomb_group(w: &mut BitWriter, values: &[u64], m: u64) -> Result<()> {
    let b = 63 - m.leading_zeros();
    let t = (1u64 << (b + 1)) - m;
    for &u in values {
        let q = u / m;
        let r = u % m;
        if q > MAX_RICE_UNARY2 {
            return Err(Error::internal("golomb unary run exceeds the bound"));
        }
        for _ in 0..q {
            w.bit(false);
        }
        w.bit(true);
        if r < t {
            if b > 0 {
                w.bits(r, b);
            }
        } else {
            w.bits(r + t, b + 1);
        }
    }
    Ok(())
}

fn read_golomb_group(r: &mut BitReader<'_>, count: usize, m: u64) -> Result<Vec<u64>> {
    let b = 63 - m.leading_zeros();
    let t = (1u64 << (b + 1)) - m;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut q = 0u64;
        loop {
            if r.read_bit()? {
                break;
            }
            q += 1;
            if q > MAX_RICE_UNARY2 {
                return Err(Error::malformed("golomb unary run exceeds the bound"));
            }
        }
        // Truncated binary: read `b` bits; if they are below the cutoff read one
        // more and subtract the cutoff.
        let rem = if b == 0 {
            0u64
        } else {
            let y = r.read_bits(b)?;
            if y < t {
                y
            } else {
                let z = u64::from(r.read_bit()?);
                ((y << 1) | z) - t
            }
        };
        out.push(q * m + rem);
    }
    Ok(out)
}

fn encode_golomb(residual: &[i32]) -> Vec<u8> {
    let mapped = zigzag_map(residual);
    encode_golomb_mapped(&mapped)
}

fn encode_golomb_mapped(mapped: &[u64]) -> Vec<u8> {
    let n = mapped.len();
    let mut out = Vec::new();
    out.extend_from_slice(&(n as u64).to_le_bytes());
    out.extend_from_slice(&(GOLOMB_CHUNK as u32).to_le_bytes());
    let nchunks = n.div_ceil(GOLOMB_CHUNK);
    out.extend_from_slice(&(nchunks as u32).to_le_bytes());
    let mut params = Vec::with_capacity(nchunks);
    for c in 0..nchunks {
        let from = c * GOLOMB_CHUNK;
        let to = (from + GOLOMB_CHUNK).min(n);
        let m = best_golomb_m(&mapped[from..to]);
        params.push(m);
    }
    for &m in &params {
        put_uvarint(&mut out, m);
    }
    let mut w = BitWriter::new();
    for (c, &m) in params.iter().enumerate() {
        let from = c * GOLOMB_CHUNK;
        let to = (from + GOLOMB_CHUNK).min(n);
        if write_golomb_group(&mut w, &mapped[from..to], m).is_err() {
            // Bounded unary run: fall back to a divisor guaranteeing it.
            let max_u = mapped[from..to].iter().copied().max().unwrap_or(0);
            let mm = (max_u / MAX_RICE_UNARY2).max(1);
            let _ = write_golomb_group(&mut w, &mapped[from..to], mm);
        }
    }
    out.extend_from_slice(&w.finish());
    out
}

fn decode_golomb(bytes: &[u8], len: usize) -> Result<Vec<i32>> {
    let (mapped, _) = decode_golomb_mapped(bytes, len)?;
    mapped.iter().map(|&u| to_i32(u)).collect()
}

fn decode_golomb_mapped(bytes: &[u8], len: usize) -> Result<(Vec<u64>, usize)> {
    let mut r = Reader::new(bytes);
    let total = r.u64le()? as usize;
    if total != len {
        return Err(Error::malformed("golomb length mismatch"));
    }
    if len == 0 {
        return Ok((Vec::new(), 0));
    }
    let chunk = r.u32le()? as usize;
    if chunk == 0 || chunk > (1 << 24) {
        return Err(Error::malformed("golomb partition size out of range"));
    }
    let nchunks = r.u32le()? as usize;
    if nchunks != len.div_ceil(chunk) {
        return Err(Error::malformed("golomb partition count mismatch"));
    }
    let mut params = Vec::with_capacity(nchunks);
    for _ in 0..nchunks {
        let m = r.uvarint()?;
        if m == 0 || m > MAX_GOLOMB_M {
            return Err(Error::malformed("golomb divisor out of range"));
        }
        params.push(m);
    }
    let rest = r.rest();
    let mut br = BitReader::new(rest);
    let mut out = Vec::with_capacity(len);
    for (c, &m) in params.iter().enumerate() {
        let from = c * chunk;
        let count = (from + chunk).min(len) - from;
        let vals = read_golomb_group(&mut br, count, m)?;
        out.extend_from_slice(&vals);
    }
    Ok((out, len))
}

fn encode_centered_golomb(residual: &[i32]) -> Vec<u8> {
    let n = residual.len();
    let mut out = Vec::new();
    out.extend_from_slice(&(n as u64).to_le_bytes());
    if n == 0 {
        out.extend_from_slice(&0i32.to_le_bytes());
        return out;
    }
    let center = median_i32(residual);
    out.extend_from_slice(&center.to_le_bytes());
    let mapped: Vec<u64> = residual
        .iter()
        .map(|&v| zigzag_i64(i64::from(v) - i64::from(center)))
        .collect();
    // The Golomb body carries its own length prefix; decode reads it back.
    let body = encode_golomb_mapped(&mapped);
    out.extend_from_slice(&body);
    out
}

fn decode_centered_golomb(bytes: &[u8], len: usize) -> Result<Vec<i32>> {
    let mut r = Reader::new(bytes);
    let total = r.u64le()? as usize;
    if total != len {
        return Err(Error::malformed("centered-golomb length mismatch"));
    }
    if len == 0 {
        return Ok(Vec::new());
    }
    let center = r.i32le()?;
    let rest = r.rest();
    let (mapped, _) = decode_golomb_mapped(rest, len)?;
    let mut out = Vec::with_capacity(len);
    for &u in &mapped {
        let e = unzigzag_i64(u);
        let v = i64::from(center)
            .checked_add(e)
            .ok_or_else(|| Error::malformed("centered-golomb value overflows"))?;
        if v < i64::from(i32::MIN) || v > i64::from(i32::MAX) {
            return Err(Error::malformed("centered-golomb value out of i32 domain"));
        }
        out.push(v as i32);
    }
    Ok(out)
}

fn median_i32(values: &[i32]) -> i32 {
    let mut v: Vec<i32> = values.to_vec();
    v.sort_unstable();
    v[v.len() / 2]
}

// ---------------------------------------------------------------------------
// id 14: FactorShift (strip a common power-of-two factor)
// ---------------------------------------------------------------------------

/// Encode the residual quotients after dividing out their common power-of-two
/// factor. The factor is stored as a shift count.
fn encode_factor_shift(residual: &[i32]) -> Vec<u8> {
    let n = residual.len();
    let mut out = Vec::new();
    out.extend_from_slice(&(n as u64).to_le_bytes());
    if n == 0 {
        out.push(0);
        return out;
    }
    let mut t = 32u32;
    for &v in residual {
        if v != 0 {
            t = t.min(v.trailing_zeros());
        }
    }
    if t == 32 {
        t = 0; // all zero
    }
    t = t.min(31);
    out.push(t as u8);
    let q: Vec<i32> = residual.iter().map(|&v| v >> t).collect();
    let inner = encode_best_v2(&q);
    out.extend_from_slice(&inner.bytes);
    out
}

fn decode_factor_shift(bytes: &[u8], len: usize) -> Result<Vec<i32>> {
    let mut r = Reader::new(bytes);
    let total = r.u64le()? as usize;
    if total != len {
        return Err(Error::malformed("factor-shift length mismatch"));
    }
    if len == 0 {
        return Ok(Vec::new());
    }
    let t = r.u8()? as u32;
    if t > 31 {
        return Err(Error::malformed("factor-shift factor out of range"));
    }
    let rest = r.rest();
    let inner = decode_encoding_v2(rest, len)?;
    let q = inner.decode(len)?;
    Ok(q.into_iter().map(|v| v << t).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_residuals() -> Vec<Vec<i32>> {
        let mut v = Vec::new();
        v.push(vec![0i32; 300]);
        let mut a = vec![0i32; 300];
        a[10] = 50_000;
        a[42] = -70_000;
        v.push(a);
        v.push((0..300).map(|i| ((i * 37) % 9) - 4).collect());
        v.push((0..300).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect());
        let mut s = 0x1234_5678u64;
        v.push(
            (0..300)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (s as i32).wrapping_mul(2_654_435_761u32 as i32)
                })
                .collect(),
        );
        v.push(
            (0..64)
                .map(|i| if i % 3 == 0 { i32::MIN } else { i32::MAX })
                .collect(),
        );
        // Heavy tail: mostly small, rare very large.
        v.push(
            (0..400)
                .map(|i| if i % 97 == 0 { 1_400_000 } else { i % 3 - 1 })
                .collect(),
        );
        // Long zero runs with isolated large spikes.
        v.push(
            (0..600)
                .map(|i| if i % 120 == 7 { -900_000 } else { 0 })
                .collect(),
        );
        // Empty and single.
        v.push(Vec::new());
        v.push(vec![7]);
        v
    }

    #[test]
    fn every_v2_codec_round_trips_exactly() {
        for (n, residual) in sample_residuals().into_iter().enumerate() {
            for codec in ResidualCodecV2::ALL {
                let bytes = codec.encode(&residual);
                let back = codec.decode(&bytes, residual.len()).unwrap();
                assert_eq!(back, residual, "codec {} case {n}", codec.name());
            }
        }
    }

    #[test]
    fn golomb_codecs_round_trip_exactly() {
        for residual in sample_residuals() {
            for codec in [ResidualCodecV2::Golomb, ResidualCodecV2::CenteredGolomb] {
                let payload = codec.encode(&residual);
                let back = codec
                    .decode(&payload, residual.len())
                    .unwrap_or_else(|e| panic!("{:?}: {e}", codec));
                assert_eq!(back, residual, "{:?}", codec);
            }
        }
    }

    #[test]
    fn best_v3_is_never_larger_than_best_v2() {
        for residual in sample_residuals() {
            let v2 = encode_best_v2(&residual);
            let v3 = encode_best_v3(&residual);
            assert!(v3.complete_bytes() <= v2.complete_bytes());
            assert_eq!(v3.decode(residual.len()).unwrap(), residual);
        }
    }

    #[test]
    fn best_v2_is_never_larger_than_best_v1() {
        for residual in sample_residuals() {
            let v1 = encode_best_v1(&residual);
            let v2 = encode_best_v2(&residual);
            assert!(
                v2.complete_bytes() <= v1.complete_bytes(),
                "v2 {} > v1 {}",
                v2.complete_bytes(),
                v1.complete_bytes()
            );
            assert_eq!(v2.decode(residual.len()).unwrap(), residual);
        }
    }

    #[test]
    fn best_v2_is_minimal_and_deterministic() {
        for residual in sample_residuals() {
            let best = encode_best_v2(&residual);
            let all = encode_all_v2(&residual);
            let min = all.iter().map(|e| e.complete_bytes()).min().unwrap();
            assert_eq!(best.complete_bytes(), min);
            let first = all.iter().find(|e| e.complete_bytes() == min).unwrap();
            assert_eq!(best.codec, first.codec);
            assert_eq!(encode_best_v2(&residual), best);
        }
    }

    #[test]
    fn v1_members_are_byte_identical_to_the_frozen_family() {
        for residual in sample_residuals() {
            let frozen = crate::learned::residual_codec::encode_best(&residual);
            let wrapped = encode_best_v1(&residual);
            assert_eq!(wrapped.codec.as_v1(), Some(frozen.codec));
            assert_eq!(wrapped.bytes, frozen.bytes);
        }
    }

    #[test]
    fn hostile_inputs_never_panic() {
        for codec in ResidualCodecV2::ALL {
            for len in [0usize, 1, 4, 16, 300] {
                for a in 0u16..=255 {
                    let _ = codec.decode(&[a as u8], len);
                }
                let _ = codec.decode(&[0xFF; 32], len);
                let _ = codec.decode(&[0x00; 32], len);
                let _ = codec.decode(&[0x80; 64], len);
            }
        }
    }

    #[test]
    fn truncation_is_rejected_or_never_panics() {
        for residual in sample_residuals() {
            for codec in ResidualCodecV2::ALL {
                let bytes = codec.encode(&residual);
                for cut in 1..bytes.len().min(64) {
                    let short = &bytes[..bytes.len() - cut];
                    let _ = codec.decode(short, residual.len());
                }
            }
        }
    }

    #[test]
    fn codec_family_has_stable_ids_and_names() {
        for (i, codec) in ResidualCodecV2::ALL.iter().enumerate() {
            assert_eq!(codec.id() as usize, i);
            assert_eq!(ResidualCodecV2::from_id(codec.id()), Some(*codec));
        }
        assert_eq!(ResidualCodecV2::ALL_V1.len(), 6);
        assert_eq!(ResidualCodecV2::ALL_V2.len(), 6);
        assert!(ResidualCodecV2::ALL_V1.iter().all(|c| c.is_v1()));
        assert!(ResidualCodecV2::ALL_V2.iter().all(|c| !c.is_v1()));
    }

    #[test]
    fn partition_rice_beats_fixed_rice_on_a_changepoint() {
        // A regime switch: quiet first half, large second half.
        let mut r = vec![0i32; 512];
        for (i, slot) in r.iter_mut().enumerate() {
            *slot = if i < 256 {
                (i % 3) as i32 - 1
            } else {
                (i as i32) * 997
            };
        }
        let p = ResidualCodecV2::PartitionRice.complete_len(&r);
        let fixed = ResidualCodecV2::BlockRice.complete_len(&r);
        assert!(p <= fixed, "partition {p} vs fixed {fixed}");
    }

    impl ResidualCodecV2 {
        fn complete_len(self, r: &[i32]) -> u64 {
            self.encode(r).len() as u64 + 1
        }
    }
}
