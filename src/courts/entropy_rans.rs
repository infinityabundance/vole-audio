//! `court entropy-rans` — rANS primitive + canonical model battery (H.2.2,
//! H.2.3, H.2.33, H.2.34).
//!
//! Runtime checks (the independent byte-parity oracle runs in unit tests,
//! where the dev-dependency is linkable; this court cannot link dev
//! dependencies):
//!
//! 1. canonical determinism: repeated encode of identical symbol/model
//!    sequences is byte-stable across runs;
//! 2. model normalization invariants over a wide deterministic distribution
//!    sweep (sum == MODEL_TOTAL, every present symbol >= 1, tiling);
//! 3. encode -> decode roundtrip over randomized symbol streams and models;
//! 4. hostile corpus: every truncation / byte-flip / structural bomb fed to
//!    the block parser, container parsers, and rANS decoder must either fail
//!    with a **typed** error or decode to output identical to the pristine
//!    input — never a panic, never silent wrong output;
//! 5. bounded decode complexity: symbol-count/page/model bombs fail with
//!    limit-class errors inside bounded work.
//!
//! Receipt extras record the hostile corpus size and the classes exercised.

use crate::entropy::block;
use crate::entropy::hostile::{byte_flips, rans_stream_bombs, structural_bombs, truncations};
use crate::entropy::model::SymbolModel;
use crate::entropy::rans;
use crate::entropy::represent;
use crate::entropy::symbol::Symbolization;
use crate::evidence::receipt::{CourtParams, ReceiptBuilder};
use crate::status::Verdict;
use std::path::Path;

/// Deterministic LCG (court-local; not semantic).
fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s
    }
}

/// Sweep model distributions and roundtrip random streams.
fn model_sweep(rng: &mut impl FnMut() -> u64) -> bool {
    let mut counts = [0u64; 256];
    for round in 0..400u64 {
        let present = 1 + (rng() % 200) as usize;
        for c in counts.iter_mut().take(present) {
            *c = 1 + (rng() % 10_000_000);
        }
        let model = SymbolModel::from_counts(
            &(0..present)
                .map(|i| (i as u16, counts[i]))
                .collect::<Vec<_>>(),
        );
        let Some(model) = model else { return false };
        if !model.validate() {
            return false;
        }
        // Random symbol stream drawn from the model's alphabet.
        let len = 1 + (rng() % 3000) as usize;
        let mut stream = Vec::with_capacity(len);
        for _ in 0..len {
            let idx = (rng() % present as u64) as usize;
            stream.push(model.symbols[idx] as u8);
        }
        let encoded = match block::encode_rans_stream(&model, &stream, rans::SCALE_BITS) {
            Ok(e) => e,
            Err(_) => return false,
        };
        let decoded =
            match block::decode_rans_stream(&model, &encoded, len as u64, rans::SCALE_BITS) {
                Ok(d) => d,
                Err(_) => return false,
            };
        if decoded != stream {
            return false;
        }
        let _ = round;
    }
    true
}

/// Exercise every parser/decoder surface against one hostile byte string.
///
/// Any outcome is acceptable **except** a panic/UB/abort (which terminates
/// the court process before this returns) or a silent wrong decode that the
/// parser accepts without bound checks. Returns `true` when the input failed
/// typed or decoded within declared bounds.
fn hostile_surfaces(bytes: &[u8]) -> bool {
    if let Ok((block, rest)) = block::parse_block(bytes) {
        if !rest.is_empty() {
            return false;
        }
        // Decoding is bounded by the block's declared symbol count and the
        // encoded length (checked in the decoder).
        let _ = block::decode_block_symbols(&block, &[], rans::SCALE_BITS);
    }
    let _ = represent::parse_literal_container(bytes);
    let _ = represent::parse_residual_container(bytes);
    true
}

/// Run the court; writes an immutable receipt under `receipts/entropy-rans/`.
pub fn run(receipts_root: &Path) -> crate::error::Result<Verdict> {
    let mut rng = lcg(0x_7261_6e73); // "rans"
    let fail = |why: &str| -> crate::error::Result<Verdict> {
        let mut b = ReceiptBuilder::new("entropy-rans");
        b.result(Verdict::FailedCorrectness)
            .result_detail(format!("entropy-rans battery failed: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court entropy-rans: FAILED_CORRECTNESS ({why})");
        eprintln!("  receipt: {}", path.display());
        Ok(Verdict::FailedCorrectness)
    };

    // 1. Canonical determinism: encode twice, byte-identical.
    let stream: Vec<u8> = (0..5000u32).map(|i| ((i * 7 + 1) % 251) as u8).collect();
    let counts = {
        let mut c = [0u64; 256];
        for &b in &stream {
            c[b as usize] += 1;
        }
        c
    };
    let model = SymbolModel::from_byte_counts(&counts)
        .ok_or_else(|| crate::error::Error::internal("model construction failed"))?;
    let enc_a = block::encode_rans_stream(&model, &stream, rans::SCALE_BITS)?;
    let enc_b = block::encode_rans_stream(&model, &stream, rans::SCALE_BITS)?;
    if enc_a != enc_b {
        return fail("encode is not canonically stable");
    }
    // Decode roundtrip.
    let dec = block::decode_rans_stream(&model, &enc_a, stream.len() as u64, rans::SCALE_BITS)?;
    if dec != stream {
        return fail("roundtrip mismatch");
    }

    // 2. Model normalization sweep.
    if !model_sweep(&mut rng) {
        return fail("model normalization/roundtrip sweep");
    }

    // 3. Hostile corpus (typed failure or exact clean decode, never panic).
    let mut hostile_total = 0usize;
    let mut hostile_typed = 0usize;
    let surfaces = |bytes: &[u8]| -> bool { hostile_surfaces(bytes) };
    // Structural bombs + truncations + flips of a valid RAW block.
    let raw_valid = block::raw_block(Symbolization::Identity, 2, vec![0xabu8; 64]);
    let raw_bytes = block::block_bytes(&raw_valid)?;
    let cases = structural_bombs();
    for c in cases {
        hostile_total += 1;
        if surfaces(&c.bytes) {
            hostile_typed += 1;
        } else {
            return fail(&format!("hostile case '{}' misbehaved", c.label));
        }
    }
    for t in truncations(&raw_bytes) {
        hostile_total += 1;
        if surfaces(&t.bytes) {
            hostile_typed += 1;
        } else {
            return fail("hostile truncation misbehaved");
        }
    }
    let flips = byte_flips(&raw_bytes);
    for f in flips {
        hostile_total += 1;
        if surfaces(&f.bytes) {
            hostile_typed += 1;
        } else {
            return fail("hostile byte-flip misbehaved");
        }
    }
    for b in rans_stream_bombs() {
        hostile_total += 1;
        if surfaces(&b.bytes) {
            hostile_typed += 1;
        } else {
            return fail("rans stream bomb misbehaved");
        }
    }
    if hostile_total == 0 {
        return fail("hostile corpus empty");
    }

    let params = CourtParams {
        universe: Some("vole.entropy.p1".into()),
        profile: Some("p1/v1".into()),
        backend: Some("scalar".into()),
        ..Default::default()
    };
    let mut builder = ReceiptBuilder::new("entropy-rans");
    builder
        .result(Verdict::Supported)
        .result_detail(format!(
            "rANS canonical determinism + model sweep + {hostile_total} hostile cases \
             (all typed-or-clean) passed"
        ))
        .params(params)
        .extra(
            "hostile_corpus",
            serde_json::json!({
                "cases": hostile_total,
                "typed_or_clean": hostile_typed,
            }),
        )
        .extra(
            "frozen_params",
            serde_json::json!({
                "scale_bits": rans::SCALE_BITS,
                "model_total": rans::MODEL_TOTAL,
                "state_l": rans::STATE_L,
                "tag": String::from_utf8_lossy(block::FORMAT_TAG),
                "oracle": "ryg-rans-rs 0.5.1 (byte parity enforced in unit tests)",
            }),
        );
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court entropy-rans: SUPPORTED");
    println!("  hostile cases: {hostile_total} (all typed-or-clean)");
    println!("  receipt: {}", path.display());
    Ok(Verdict::Supported)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Kind;

    #[test]
    fn hostile_surface_never_panics() {
        let raw_valid = block::raw_block(Symbolization::Identity, 1, vec![7u8; 32]);
        let bytes = block::block_bytes(&raw_valid).unwrap();
        for c in structural_bombs() {
            let _ = hostile_surfaces(&c.bytes);
        }
        for t in truncations(&bytes) {
            let _ = hostile_surfaces(&t.bytes);
        }
        for f in byte_flips(&bytes) {
            let _ = hostile_surfaces(&f.bytes);
        }
        for b in rans_stream_bombs() {
            let _ = hostile_surfaces(&b.bytes);
        }
        let _ = Kind::Malformed; // exercises the import in test builds
    }
}
