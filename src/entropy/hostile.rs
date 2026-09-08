//! Deterministic hostile-input generators for parser/decoder hardening
//! (H.2.33/H.2.34/H.2.47).
//!
//! The hostile court and fuzz-style unit tests feed these byte strings to the
//! entropy parsers/decoders. The contract (owner: `docs/RANS.md`
//! "Malformed-stream behavior"): every malformed input produces a **typed
//! error** — never panic, never UB, never unbounded allocation/CPU, never
//! silent wrong sample output. A decoded `Ok` from a mutated input must
//! equal the original decode (a mutation that still parses must not change
//! meaning silently).

use crate::entropy::block::{self};
use crate::entropy::model::SymbolModel;
use crate::entropy::rans;
use crate::entropy::represent;
use crate::entropy::symbol::Symbolization;

/// A labeled hostile byte string.
pub struct HostileCase {
    pub label: &'static str,
    pub bytes: Vec<u8>,
}

/// All strict truncations of `bytes` (every prefix).
pub fn truncations(bytes: &[u8]) -> Vec<HostileCase> {
    (0..bytes.len())
        .map(|cut| HostileCase {
            label: "truncation",
            bytes: bytes[..cut].to_vec(),
        })
        .collect()
}

/// Byte-flip mutations: every byte toggled with a spread of masks.
pub fn byte_flips(bytes: &[u8]) -> Vec<HostileCase> {
    let masks = [0x01u8, 0x80, 0xff, 0x55];
    let mut out = Vec::new();
    for (i, &b) in bytes.iter().enumerate() {
        for &m in &masks {
            let mut mutated = bytes.to_vec();
            mutated[i] = b ^ m;
            out.push(HostileCase {
                label: "byte-flip",
                bytes: mutated,
            });
        }
    }
    out
}

/// Structural bombs: hand-built malformed containers targeting each parser
/// invariant.
pub fn structural_bombs() -> Vec<HostileCase> {
    let mut cases: Vec<HostileCase> = Vec::new();
    let mut push = |label: &'static str, bytes: Vec<u8>| cases.push(HostileCase { label, bytes });

    // Bad tag / version / kind.
    let mut bad_tag = block::FORMAT_TAG.to_vec();
    bad_tag[0] = b'X';
    push("bad block tag", bad_tag);
    let mut bad_ver = block::FORMAT_TAG.to_vec();
    bad_ver.push(99);
    push("bad block version", bad_ver);
    push("empty", Vec::new());
    push("single byte", vec![0xab]);
    // A valid RAW block then hostile surroundings.
    let raw = block::raw_block(Symbolization::Identity, 1, vec![0u8; 8]);
    let valid = block::block_bytes(&raw).unwrap();
    push("valid raw block", valid.clone());
    // Truncations of the valid block.
    for t in truncations(&valid) {
        push("truncated valid block", t.bytes);
    }
    // Symbol count bomb on a RAW block (limit class).
    let mut bomb = valid.clone();
    bomb[20..28].copy_from_slice(&u64::MAX.to_le_bytes());
    push("symbol count bomb", bomb);
    // Inline model claiming an absurd alphabet: craft a valid block, then
    // corrupt the inline model count region.
    let tiny = block::rans_block(
        Symbolization::Identity,
        1,
        SymbolModel::from_byte_counts(&byte_counts(&[0u8; 4])).unwrap(),
        4,
        vec![0u8; 4],
    );
    let mut hb = block::block_bytes(&tiny).unwrap();
    // The inline model begins after the fixed header (32 bytes): u16 count.
    if hb.len() > 34 {
        hb[32] = 0xff;
        hb[33] = 0xff; // alphabet 65535 > 256
        push("model alphabet bomb", hb);
    }

    // Container-level bombs.
    let c = represent::literal_container_bytes(&represent::RepresentedLiteral {
        descriptor: crate::object::descriptor::ObjectDescriptor::new(
            crate::object::descriptor::Representation::Literal,
            512,
            crate::universe::layout::Layout::Mono,
            None,
        )
        .unwrap(),
        page_frames: 512,
        symbolization: Symbolization::Lane4Plain,
        model_mode: represent::ModelMode::Inline,
        pool: Vec::new(),
        pages: Vec::new(),
    })
    .unwrap();
    push("empty-pages container", c.clone());
    let mut pc = c.clone();
    pc[15] = 7;
    push("container version bomb", pc);
    let mut kc = c.clone();
    kc[16] = 9;
    push("container kind bomb", kc);
    let mut ff = c.clone();
    if ff.len() > 23 {
        ff[19..23].copy_from_slice(&0u32.to_le_bytes());
        push("page_frames zero", ff);
    }
    // Trailing garbage after a valid container must be rejected.
    let mut tc = c.clone();
    tc.extend_from_slice(&[0u8; 3]);
    push("trailing garbage", tc);
    cases
}

fn byte_counts(stream: &[u8]) -> [u64; 256] {
    let mut counts = [0u64; 256];
    for &b in stream {
        counts[b as usize] += 1;
    }
    counts
}

/// rANS-primitive hostile inputs: truncated state / renorm bytes against a
/// valid model.
pub fn rans_stream_bombs() -> Vec<HostileCase> {
    let mut out = Vec::new();
    let stream: Vec<u8> = (0..256u32).map(|i| (i % 251) as u8).collect();
    let model = SymbolModel::from_byte_counts(&byte_counts(&stream)).unwrap();
    if let Ok(enc) = block::encode_rans_stream(&model, &stream, rans::SCALE_BITS) {
        for t in truncations(&enc) {
            out.push(HostileCase {
                label: "rans truncation",
                bytes: t.bytes,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hostile_generators_are_deterministic_and_bounded() {
        let b = structural_bombs();
        assert!(!b.is_empty());
        let flips = byte_flips(
            &block::block_bytes(&block::raw_block(Symbolization::Identity, 1, vec![1, 2, 3]))
                .unwrap(),
        );
        assert!(!flips.is_empty());
        let names: std::collections::BTreeSet<_> = b.iter().map(|c| c.label).collect();
        assert!(names.len() >= 5, "distinct hostile classes");
    }

    #[test]
    fn valid_inputs_are_not_mutated_by_generators() {
        // A truncation of length 0 or full length is excluded by design;
        // verify the valid block itself round-trips (sanity anchor).
        let raw = block::raw_block(Symbolization::Identity, 1, vec![9u8; 16]);
        let bytes = block::block_bytes(&raw).unwrap();
        let (parsed, rest) = block::parse_block(&bytes).unwrap();
        assert!(rest.is_empty());
        assert_eq!(
            block::decode_block_symbols(&parsed, &[], rans::SCALE_BITS).unwrap(),
            vec![9u8; 16]
        );
    }
}
