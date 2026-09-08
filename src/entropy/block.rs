//! Canonical self-describing entropy blocks (H.2.2/H.2.5/H.2.6) — host.
//!
//! One [`Block`] codes exactly one symbol stream (a `Vec<u8>` of symbol
//! values). Page assembly (frames x channels x streams), page indexes, shared
//! model pools, and RAW-vs-rANS selection live in `represent.rs`; this module
//! owns the `vole.entropy.p1` container bytes and their hostile-input-safe
//! parser.
//!
//! Byte layout (self-delimiting; every field endian-defined):
//!
//! ```text
//! tag "vole.entropy.p1"      15 bytes ASCII
//! version                    1 byte  (= 1)
//! payload_kind               1 byte  (0 = RANS, 1 = RAW)
//! symbolization              1 byte  (Symbolization code)
//! channel_scope              1 byte  (object channels this stream belongs to)
//! model_mode                 1 byte  (0 = inline model, 1 = shared pool ref)
//! symbol_count               u64 LE  (number of coded symbols)
//! encoded_len                u32 LE  (payload length that follows)
//! model                      inline: canonical model bytes
//!                            shared: pool index u32 LE
//! payload                    encoded_len bytes (rANS stream or raw symbols)
//! integrity                  flag u8; if 1, 32 bytes SHA-256 over the bytes
//!                            preceding the flag
//! ```
//!
//! RAW fallback (H.2.5): a block with `payload_kind = RAW` carries its symbol
//! bytes verbatim. Selection between RANS and RAW is made on **complete
//! bytes** (container + model + payload), never on body size alone.

use crate::entropy::model::SymbolModel;
use crate::entropy::rans;
use crate::entropy::symbol::Symbolization;
use crate::error::{Error, Result};
use crate::hash::sha256::Sha256;

/// Container profile tag (frozen; see docs/RANS.md).
pub const FORMAT_TAG: &[u8; 15] = b"vole.entropy.p1";
/// Container version (frozen for this profile).
pub const FORMAT_VERSION: u8 = 1;

/// Payload kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PayloadKind {
    Rans = 0,
    Raw = 1,
}

impl PayloadKind {
    pub const fn code(self) -> u8 {
        self as u8
    }
    pub const fn from_code(c: u8) -> Option<PayloadKind> {
        match c {
            0 => Some(PayloadKind::Rans),
            1 => Some(PayloadKind::Raw),
            _ => None,
        }
    }
}

/// Model reference within a block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelRef {
    /// Model bytes carried inside the block.
    Inline(SymbolModel),
    /// Index into the object's shared model pool (content-addressed at the
    /// object/store level).
    Shared(u32),
}

/// One coded symbol stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    /// Symbolization of the samples this stream belongs to (identity for RAW
    /// sample bytes).
    pub symbolization: Symbolization,
    /// Object channel scope (informational; validated 1..=MAX_CHANNELS).
    pub channel_scope: u8,
    pub payload: BlockPayload,
    /// Optional SHA-256 over the canonical block bytes before the integrity
    /// flag.
    pub integrity: Option<[u8; 32]>,
}

/// Coded payload of one symbol stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockPayload {
    Rans {
        /// The normalized model the stream was coded with.
        model: ModelRef,
        /// Number of coded symbols (the decoded stream length).
        symbol_count: u64,
        /// Canonical rANS stream (`state || renorm bytes`).
        bytes: Vec<u8>,
    },
    /// Raw symbol bytes (or raw LE sample bytes for identity pages).
    Raw { bytes: Vec<u8> },
}

impl BlockPayload {
    pub const fn kind(&self) -> PayloadKind {
        match self {
            BlockPayload::Rans { .. } => PayloadKind::Rans,
            BlockPayload::Raw { .. } => PayloadKind::Raw,
        }
    }

    /// Symbol count of the payload (decoded length for RANS, byte length for
    /// RAW).
    pub fn symbol_count(&self) -> u64 {
        match self {
            BlockPayload::Rans { symbol_count, .. } => *symbol_count,
            BlockPayload::Raw { bytes } => bytes.len() as u64,
        }
    }
}

/// Fixed header length before the model/payload region.
const HEADER_LEN: usize = 15 + 1 + 1 + 1 + 1 + 1 + 8 + 4;

/// Build the fixed header of a block.
fn header_bytes(
    payload_kind: PayloadKind,
    block: &Block,
    symbol_count: u64,
    encoded_len: u32,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN);
    out.extend_from_slice(FORMAT_TAG);
    out.push(FORMAT_VERSION);
    out.push(payload_kind.code());
    out.push(block.symbolization.code());
    out.push(block.channel_scope);
    match &block.payload {
        BlockPayload::Rans { model, .. } => match model {
            ModelRef::Inline(_) => out.push(0),
            ModelRef::Shared(_) => out.push(1),
        },
        BlockPayload::Raw { .. } => out.push(0),
    }
    out.extend_from_slice(&symbol_count.to_le_bytes());
    out.extend_from_slice(&encoded_len.to_le_bytes());
    out
}

/// Serialize a block to canonical container bytes.
pub fn block_bytes(block: &Block) -> Result<Vec<u8>> {
    match &block.payload {
        BlockPayload::Rans {
            model,
            symbol_count,
            bytes,
        } => {
            let model_bytes = match model {
                ModelRef::Inline(m) => m.canonical_bytes(),
                ModelRef::Shared(idx) => (*idx).to_le_bytes().to_vec(),
            };
            if bytes.len() > u32::MAX as usize {
                return Err(Error::limit("block payload exceeds u32 length ceiling"));
            }
            if *symbol_count > u64::from(crate::limits::MAX_ENTROPY_PAGE_DECODED_BYTES) {
                return Err(Error::limit("symbol_count above decode ceiling"));
            }
            let mut out = header_bytes(PayloadKind::Rans, block, *symbol_count, bytes.len() as u32);
            out.extend_from_slice(&model_bytes);
            out.extend_from_slice(bytes);
            append_integrity(block, &mut out)?;
            Ok(out)
        }
        BlockPayload::Raw { bytes } => {
            if bytes.len() > u32::MAX as usize {
                return Err(Error::limit("RAW payload exceeds u32 length ceiling"));
            }
            let mut out = header_bytes(
                PayloadKind::Raw,
                block,
                bytes.len() as u64,
                bytes.len() as u32,
            );
            out.extend_from_slice(bytes);
            append_integrity(block, &mut out)?;
            Ok(out)
        }
    }
}

/// Append the integrity flag + digest over the preceding bytes when the block
/// requests integrity. The digest covers the canonical bytes up to (but not
/// including) the integrity flag.
fn append_integrity(block: &Block, out: &mut Vec<u8>) -> Result<()> {
    match block.integrity {
        Some(digest) => {
            // Digest over everything already written (header+model+payload).
            let computed = Sha256::digest(out);
            if computed != digest {
                return Err(Error::integrity(
                    "block integrity digest does not match its bytes",
                ));
            }
            out.push(1);
            out.extend_from_slice(&digest);
            Ok(())
        }
        None => {
            out.push(0);
            Ok(())
        }
    }
}

/// Parse one canonical block from `bytes` (consumes exactly one block; the
/// remainder is returned). Hostile inputs produce typed errors.
pub fn parse_block(bytes: &[u8]) -> Result<(Block, &[u8])> {
    if bytes.len() < HEADER_LEN + 1 {
        return Err(Error::malformed("block shorter than fixed header"));
    }
    if &bytes[..15] != FORMAT_TAG {
        return Err(Error::malformed("bad entropy container tag"));
    }
    if bytes[15] != FORMAT_VERSION {
        return Err(Error::malformed("unsupported entropy container version"));
    }
    let payload_kind = PayloadKind::from_code(bytes[16])
        .ok_or_else(|| Error::malformed("invalid payload kind"))?;
    let symbolization = Symbolization::from_code(bytes[17])
        .ok_or_else(|| Error::malformed("invalid symbolization id"))?;
    let channel_scope = bytes[18];
    if channel_scope == 0 || u32::from(channel_scope) > crate::limits::MAX_CHANNELS {
        return Err(Error::malformed("channel scope out of domain"));
    }
    let model_mode = bytes[19];
    if model_mode > 1 {
        return Err(Error::malformed("invalid model mode"));
    }
    let symbol_count = u64::from_le_bytes(bytes[20..28].try_into().unwrap());
    if symbol_count > u64::from(crate::limits::MAX_ENTROPY_PAGE_DECODED_BYTES) {
        return Err(Error::limit("symbol_count above decode ceiling"));
    }
    let encoded_len = u32::from_le_bytes(bytes[28..32].try_into().unwrap()) as usize;
    let mut pos = 32usize;

    // Model region. RAW blocks carry no model; RANS blocks carry an inline
    // model or a shared pool reference.
    let model = match payload_kind {
        PayloadKind::Raw => {
            if model_mode != 0 {
                return Err(Error::malformed(
                    "RAW block must not declare a model (mode 0 only)",
                ));
            }
            None
        }
        PayloadKind::Rans => match model_mode {
            0 => {
                // Inline canonical model bytes: u16 count + (u16 value, u32 freq).
                if bytes.len() < pos + 2 {
                    return Err(Error::malformed("truncated inline model"));
                }
                let count = u16::from_le_bytes(bytes[pos..pos + 2].try_into().unwrap()) as usize;
                if count == 0 || count > crate::limits::MAX_MODEL_ALPHABET {
                    return Err(Error::malformed("model alphabet out of domain"));
                }
                let model_len = 2 + count * 6;
                if bytes.len() < pos + model_len {
                    return Err(Error::malformed("truncated inline model body"));
                }
                if model_len > crate::limits::MAX_ENTROPY_INLINE_MODEL_BYTES as usize {
                    return Err(Error::limit("inline model above byte ceiling"));
                }
                let parsed = SymbolModel::parse_canonical(&bytes[pos..pos + model_len])
                    .ok_or_else(|| Error::malformed("invalid inline model bytes"))?;
                pos += model_len;
                Some(ModelRef::Inline(parsed))
            }
            1 => {
                if bytes.len() < pos + 4 {
                    return Err(Error::malformed("truncated shared model ref"));
                }
                let idx = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
                pos += 4;
                Some(ModelRef::Shared(idx))
            }
            _ => unreachable!("model_mode validated above"),
        },
    };

    // Payload region.
    if encoded_len > crate::limits::MAX_ENTROPY_PAGE_BODY_BYTES as usize {
        return Err(Error::limit("payload above body ceiling"));
    }
    if bytes.len() < pos + encoded_len {
        return Err(Error::malformed("truncated payload"));
    }
    let payload_bytes = &bytes[pos..pos + encoded_len];
    pos += encoded_len;

    // Integrity flag + digest.
    if bytes.len() < pos + 1 {
        return Err(Error::malformed("missing integrity flag"));
    }
    let integrity_flag = bytes[pos];
    let integrity = if integrity_flag == 1 {
        if bytes.len() < pos + 1 + 32 {
            return Err(Error::malformed("truncated integrity digest"));
        }
        let mut digest = [0u8; 32];
        digest.copy_from_slice(&bytes[pos + 1..pos + 1 + 32]);
        // Verify over everything preceding the flag byte.
        let mut hasher = Sha256::new();
        hasher.update(&bytes[..pos]);
        let computed = hasher.finalize();
        if computed != digest {
            return Err(Error::integrity("block integrity digest mismatch"));
        }
        pos += 1 + 32;
        Some(digest)
    } else if integrity_flag == 0 {
        pos += 1;
        None
    } else {
        return Err(Error::malformed("invalid integrity flag"));
    };

    let payload = if payload_kind == PayloadKind::Rans {
        let model = model.expect("RANS payload carries a model");
        BlockPayload::Rans {
            model,
            symbol_count,
            bytes: payload_bytes.to_vec(),
        }
    } else {
        if symbol_count as usize != payload_bytes.len() {
            return Err(Error::malformed("RAW symbol_count != payload length"));
        }
        BlockPayload::Raw {
            bytes: payload_bytes.to_vec(),
        }
    };
    let block = Block {
        symbolization,
        channel_scope,
        payload,
        integrity,
    };
    Ok((block, &bytes[pos..]))
}

/// rANS-encode one symbol stream with a given model. Every symbol value must
/// be present in the model (enforced with a typed error; a model built from
/// the same stream satisfies this by construction).
pub fn encode_rans_stream(model: &SymbolModel, symbols: &[u8], scale_bits: u32) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; rans::encode_capacity(symbols.len())];
    let mut sink = rans::BackSink::new(&mut buf);
    let mut state = rans::RansState::new();
    for &sym in symbols {
        let idx = model
            .index_of(u16::from(sym))
            .ok_or_else(|| Error::internal("symbol absent from model during encode"))?;
        if !rans::enc_put(
            &mut state,
            &mut sink,
            model.start[idx],
            model.freq[idx],
            scale_bits,
        ) {
            return Err(Error::internal("rANS encode capacity exceeded"));
        }
    }
    if !rans::enc_flush(&state, &mut sink) {
        return Err(Error::internal("rANS flush capacity exceeded"));
    }
    Ok(sink.encoded().to_vec())
}

/// rANS-decode a stream given the model; returns the symbol bytes in
/// **encode order**. Bounded by `symbol_count` and the encoded length.
///
/// Canonical validation (RANS.md "State machine", decode step 3): a valid
/// stream must decode to the terminal state `x == RANS_STATE_L` after the
/// final symbol's renorm **and** leave the byte cursor exactly at the end of
/// the encoded payload. Streams with trailing garbage, altered initial or
/// terminal state, or truncated renorm bytes are rejected — never silently
/// accepted as if they were canonical.
pub fn decode_rans_stream(
    model: &SymbolModel,
    encoded: &[u8],
    symbol_count: u64,
    scale_bits: u32,
) -> Result<Vec<u8>> {
    if !model.validate() {
        return Err(Error::malformed("model invariants violated"));
    }
    let n = usize::try_from(symbol_count)
        .map_err(|_| Error::limit("symbol_count exceeds host usize"))?;
    if n > crate::limits::MAX_ENTROPY_PAGE_DECODED_BYTES as usize {
        return Err(Error::limit("symbol_count above decode ceiling"));
    }
    let mut reader = rans::FwdReader::new(encoded);
    let Some(mut state) = rans::dec_init(&mut reader) else {
        return Err(Error::malformed("truncated or below-range rANS state"));
    };
    let mut out = vec![0u8; n];
    for i in (0..n).rev() {
        let slot = rans::dec_slot(&state, scale_bits);
        let idx = model.slot_index(slot);
        // Interval containment is guaranteed by model.validate(); assert in
        // debug and reject defensively in release (slot - start underflow
        // protection).
        if slot < model.start[idx] {
            return Err(Error::malformed("slot outside model interval"));
        }
        let start = model.start[idx];
        let freq = model.freq[idx];
        if slot >= start + freq {
            return Err(Error::malformed("slot outside model interval"));
        }
        if !rans::dec_advance(&mut state, &mut reader, start, freq, scale_bits) {
            return Err(Error::malformed("truncated rANS renorm bytes"));
        }
        out[i] = model.symbols[idx] as u8;
    }
    // Canonical terminal condition: the state must return exactly to
    // RANS_STATE_L and every payload byte must have been consumed.
    if state.0 != rans::STATE_L {
        return Err(Error::malformed(
            "noncanonical terminal rANS state (state != RANS_STATE_L)",
        ));
    }
    if reader.bytes_consumed() != encoded.len() {
        return Err(Error::malformed(
            "noncanonical rANS stream length (trailing or unread bytes)",
        ));
    }
    Ok(out)
}

/// Decode a block's symbols given a shared-model pool (used when the block
/// references `Shared(idx)`). Returns symbol bytes in encode order.
pub fn decode_block_symbols(
    block: &Block,
    pool: &[SymbolModel],
    scale_bits: u32,
) -> Result<Vec<u8>> {
    match &block.payload {
        BlockPayload::Rans {
            model,
            symbol_count,
            bytes,
        } => {
            let model = match model {
                ModelRef::Inline(m) => m,
                ModelRef::Shared(idx) => pool.get(*idx as usize).ok_or_else(|| {
                    Error::dependency(format!("shared model pool index {idx} out of range"))
                })?,
            };
            decode_rans_stream(model, bytes, *symbol_count, scale_bits)
        }
        BlockPayload::Raw { bytes } => Ok(bytes.clone()),
    }
}

/// Convenience builder for a RAW block carrying raw symbol bytes.
pub fn raw_block(symbolization: Symbolization, channel_scope: u8, bytes: Vec<u8>) -> Block {
    Block {
        symbolization,
        channel_scope,
        payload: BlockPayload::Raw { bytes },
        integrity: None,
    }
}

/// Convenience builder for a RANS block with an inline model.
pub fn rans_block(
    symbolization: Symbolization,
    channel_scope: u8,
    model: SymbolModel,
    symbol_count: u64,
    bytes: Vec<u8>,
) -> Block {
    Block {
        symbolization,
        channel_scope,
        payload: BlockPayload::Rans {
            model: ModelRef::Inline(model),
            symbol_count,
            bytes,
        },
        integrity: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::rans::SCALE_BITS;
    use crate::error::Kind;

    fn byte_model(stream: &[u8]) -> SymbolModel {
        let mut counts = [0u64; 256];
        for &b in stream {
            counts[b as usize] += 1;
        }
        SymbolModel::from_byte_counts(&counts).unwrap()
    }

    #[test]
    fn block_roundtrip_rans_and_raw() {
        let stream: Vec<u8> = (0..2000u32).map(|i| ((i * 7 + 3) % 251) as u8).collect();
        let model = byte_model(&stream);
        let encoded = encode_rans_stream(&model, &stream, SCALE_BITS).unwrap();
        assert!(encoded.len() < stream.len(), "structured stream compresses");
        let block = rans_block(
            Symbolization::Identity,
            1,
            model,
            stream.len() as u64,
            encoded,
        );
        let bytes = block_bytes(&block).unwrap();
        let (parsed, rest) = parse_block(&bytes).unwrap();
        assert!(rest.is_empty());
        let decoded = decode_block_symbols(&parsed, &[], SCALE_BITS).unwrap();
        assert_eq!(decoded, stream);
    }

    #[test]
    fn raw_block_roundtrip_and_self_delimiting() {
        let raw = vec![0xabu8; 100];
        let block = raw_block(Symbolization::Identity, 2, raw.clone());
        let bytes = block_bytes(&block).unwrap();
        let (parsed, rest) = parse_block(&bytes).unwrap();
        assert!(rest.is_empty());
        assert_eq!(decode_block_symbols(&parsed, &[], SCALE_BITS).unwrap(), raw);
        // Two concatenated blocks parse independently.
        let b2 = raw_block(Symbolization::Identity, 1, vec![1, 2, 3]);
        let mut both = bytes.clone();
        both.extend_from_slice(&block_bytes(&b2).unwrap());
        let (p1, rest) = parse_block(&both).unwrap();
        let (p2, rest2) = parse_block(rest).unwrap();
        assert!(rest2.is_empty());
        assert_eq!(decode_block_symbols(&p1, &[], SCALE_BITS).unwrap(), raw);
        assert_eq!(
            decode_block_symbols(&p2, &[], SCALE_BITS).unwrap(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn block_with_integrity_verifies() {
        let stream = vec![7u8, 8, 9, 10, 42, 43, 44, 45];
        let model = byte_model(&stream);
        let encoded = encode_rans_stream(&model, &stream, SCALE_BITS).unwrap();
        let mut block = rans_block(
            Symbolization::Identity,
            1,
            model,
            stream.len() as u64,
            encoded,
        );
        // Digest over the canonical bytes up to (not including) the trailing
        // integrity flag byte of the no-integrity serialization.
        let plain = block_bytes(&block).unwrap();
        let digest = Sha256::digest(&plain[..plain.len() - 1]);
        block.integrity = Some(digest);
        let bytes = block_bytes(&block).unwrap();
        let (parsed, rest) = parse_block(&bytes).unwrap();
        assert!(rest.is_empty());
        assert_eq!(parsed.integrity, Some(digest));
        assert_eq!(
            decode_block_symbols(&parsed, &[], SCALE_BITS).unwrap(),
            stream
        );

        // Tampering anywhere is detected (flip bytes in tag, header, and
        // payload regions).
        for idx in [0usize, 17, 21, bytes.len() - 10] {
            let mut tampered = bytes.clone();
            tampered[idx] ^= 0x40;
            assert!(parse_block(&tampered).is_err(), "tamper at {idx} detected");
        }
    }

    #[test]
    fn canonical_decoder_rejects_trailing_altered_and_truncated_streams() {
        // RANS.md decode step 3 freeze: a valid decode ends with
        // x == RANS_STATE_L and the cursor exactly at the payload end. The
        // decoder must therefore reject appended bytes (any length, any
        // value), tail truncations, and a below-range initial state. A byte
        // mutation anywhere else must fail typed or decode to a *different*
        // symbol sequence — encoding is a bijection for a fixed model and
        // length, so silent acceptance of the original symbols is impossible.
        let streams: Vec<Vec<u8>> = vec![
            // Structured (compressible) stream.
            (0..2000u32).map(|i| ((i * 7 + 3) % 251) as u8).collect(),
            // Near-uniform stream (larger model, more renorm traffic).
            (0..3000u32)
                .map(|i| {
                    let mut x = i.wrapping_mul(2654435761).wrapping_add(97);
                    x ^= x >> 13;
                    (x.wrapping_mul(0x9e3779b1) as u8) ^ ((i % 7) as u8)
                })
                .collect(),
            // Tiny stream (single symbol, minimal state traffic).
            vec![42u8; 3],
        ];
        for stream in &streams {
            let model = byte_model(stream);
            let encoded = encode_rans_stream(&model, stream, SCALE_BITS).unwrap();
            // Sanity: canonical decode reproduces the stream.
            assert_eq!(
                decode_rans_stream(&model, &encoded, stream.len() as u64, SCALE_BITS).unwrap(),
                *stream
            );
            // Every one-byte append (0..=255) is rejected (cursor must end
            // exactly at the payload end).
            for b in 0..=255u8 {
                let mut tail = encoded.clone();
                tail.push(b);
                assert!(
                    matches!(
                        decode_rans_stream(&model, &tail, stream.len() as u64, SCALE_BITS),
                        Err(e) if e.kind() == Kind::Malformed
                    ),
                    "append 0x{b:02x} accepted on {}-byte stream",
                    stream.len()
                );
            }
            // Multi-byte appends are rejected.
            for trail in 1..=16usize {
                let mut tail = encoded.clone();
                tail.extend(std::iter::repeat_n(0u8, trail));
                assert!(matches!(
                    decode_rans_stream(&model, &tail, stream.len() as u64, SCALE_BITS),
                    Err(e) if e.kind() == Kind::Malformed
                ));
            }
            // Tail truncations are rejected (the terminal renorm needs every
            // byte the canonical stream carries).
            for cut in 1..=16usize.min(encoded.len()) {
                let short = &encoded[..encoded.len() - cut];
                assert!(
                    decode_rans_stream(&model, short, stream.len() as u64, SCALE_BITS).is_err(),
                    "truncation by {cut} accepted"
                );
            }
            // A below-range initial state is rejected at dec_init.
            let mut low = encoded.clone();
            low[..4].copy_from_slice(&(crate::entropy::rans::STATE_L - 1).to_le_bytes());
            assert!(matches!(
                decode_rans_stream(&model, &low, stream.len() as u64, SCALE_BITS),
                Err(e) if e.kind() == Kind::Malformed
            ));
            // Byte mutations (state bytes, renorm body, terminal byte): fail
            // typed or decode to a different sequence — never the original.
            for at in 0..encoded.len() {
                for flip in [0x01u8, 0x40, 0x80, 0xff] {
                    let mut mutv = encoded.clone();
                    mutv[at] ^= flip;
                    match decode_rans_stream(&model, &mutv, stream.len() as u64, SCALE_BITS) {
                        Err(e) => assert_eq!(e.kind(), Kind::Malformed),
                        Ok(seq) => assert_ne!(seq, *stream, "mutation at {at} undetected"),
                    }
                }
            }
        }
    }
    #[test]
    fn hostile_blocks_fail_typed() {
        let raw = raw_block(Symbolization::Identity, 1, vec![0u8; 8]);
        let bytes = block_bytes(&raw).unwrap();
        // Bad tag.
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(matches!(parse_block(&bad), Err(e) if e.kind() == Kind::Malformed));
        // Bad version.
        let mut bad = bytes.clone();
        bad[15] = 99;
        assert!(matches!(parse_block(&bad), Err(e) if e.kind() == Kind::Malformed));
        // Truncation at every boundary is a typed error, never a panic.
        for cut in 0..bytes.len() {
            let _ = parse_block(&bytes[..cut]);
        }
        // Invalid symbolization / channel scope.
        let mut bad = bytes.clone();
        bad[17] = 0;
        assert!(matches!(parse_block(&bad), Err(e) if e.kind() == Kind::Malformed));
        let mut bad = bytes.clone();
        bad[18] = 0;
        assert!(matches!(parse_block(&bad), Err(e) if e.kind() == Kind::Malformed));
        // Symbol count bomb.
        let mut bad = bytes.clone();
        bad[20..28].copy_from_slice(&(u64::MAX).to_le_bytes());
        assert!(matches!(parse_block(&bad), Err(e) if e.kind() == Kind::LimitExceeded));
        // RAW with shared model ref rejected.
        let mut bad = bytes.clone();
        bad[19] = 1;
        assert!(parse_block(&bad).is_err());
    }

    #[test]
    fn high_entropy_stream_grows_or_breaks_even() {
        // Random bytes should not rANS-compress below RAW.
        let stream: Vec<u8> = (0..4096u32)
            .map(|i| {
                let mut x = i.wrapping_mul(2654435761).wrapping_add(97);
                x ^= x >> 13;
                x.wrapping_mul(0x9e3779b1) as u8
            })
            .collect();
        let model = byte_model(&stream);
        let encoded = encode_rans_stream(&model, &stream, SCALE_BITS).unwrap();
        let rans_total = block_bytes(&rans_block(
            Symbolization::Identity,
            1,
            model,
            stream.len() as u64,
            encoded,
        ))
        .unwrap()
        .len();
        let raw_total = block_bytes(&raw_block(Symbolization::Identity, 1, stream.clone()))
            .unwrap()
            .len();
        assert!(
            rans_total >= raw_total,
            "incompressible material must not be forced into rANS (H.2.5)"
        );
    }
}
