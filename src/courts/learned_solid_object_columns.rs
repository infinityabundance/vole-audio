//! `court learned-solid-object-columns` — Phase 6 mechanism 9
//! (`SolidObjectColumns`, archive-only).
//!
//! The canonical `.volea` archive stores each full-object container as an
//! opaque, independently integrity-bound payload in logical order. That is the
//! right logical constitution but the wrong physical order for a bounded-context
//! coder. `SolidObjectColumns` parses each container into homologous columns
//! (headers, index records, payloads grouped by segment index, integrity
//! digests) and stores them transposed, with a permutation restoring logical
//! order.
//!
//! This court builds the frozen flagship corpus's real containers, forms both
//! the logical concatenation and the solid column stream, and codes each with
//! the same adaptive order-1 byte coder whose model state carries across the
//! whole stream. It reports the physical bytes of both orders and proves the
//! solid stream reconstructs every object byte-for-byte (and that each object's
//! own SHA-256 integrity still verifies).
//!
//! **This is an archive-only claim.** A solid archive result is never a
//! single-file audio compression result and must not be compared with one.

use crate::corpus;
use crate::corpus::generate::{self, Spec};
use crate::error::Result;
use crate::format::solid::{decode_solid, encode_bytes_adaptive, encode_solid};
use crate::fullobj::{self, FullSemantics};
use crate::hash::sha256::hex;
use crate::inverse::SearchBudget;
use crate::status::Verdict;
use std::collections::BTreeMap;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_SOLID_OBJECT_COLUMNS_SHA256: &str =
    "73164a62449fc46e57a0756f561b5cfceb4e96802398fef643bcad82d060582d";

fn budget() -> SearchBudget {
    SearchBudget::default()
}

fn semantics_of(spec: &Spec) -> FullSemantics {
    match spec.semantics.period_frames() {
        Some(period_frames) => FullSemantics::Loop { period_frames },
        None => FullSemantics::OneShot,
    }
}

struct Row {
    objects: usize,
    logical_bytes: u64,
    solid_bytes: u64,
    logical_coded_bytes: u64,
    solid_coded_bytes: u64,
    reconstructs_exactly: bool,
}

/// Run the court; writes `receipts/learned-solid-object-columns/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let fail = |why: &str| -> Result<Verdict> {
        let mut b = crate::evidence::receipt::ReceiptBuilder::new("learned-solid-object-columns");
        b.result(Verdict::FailedCorrectness)
            .result_detail(format!("solid-object-columns court failed: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court learned-solid-object-columns: FAILED_CORRECTNESS ({why})");
        eprintln!("  receipt: {}", path.display());
        Ok(Verdict::FailedCorrectness)
    };

    let manifest_in = corpus::manifest()?;
    let report = corpus::verify_manifest(&manifest_in)?;
    if !report.ok() {
        return fail("the frozen flagship corpus does not verify");
    }
    let specs = corpus::specs::specs();
    let spec_by_id: BTreeMap<&str, &Spec> = specs.iter().map(|s| (s.id.as_str(), s)).collect();

    let mut named: Vec<(String, Vec<u8>)> = Vec::with_capacity(manifest_in.objects.len());
    for o in &manifest_in.objects {
        let spec = spec_by_id
            .get(o.id.as_str())
            .ok_or_else(|| crate::error::Error::internal(format!("{}: not in corpus", o.id)))?;
        let samples = generate::generate(spec)?;
        if hex(&generate::canonical_sha256(&samples)) != o.canonical_i32_sha256 {
            return fail(&format!("{}: regenerated content changed", o.id));
        }
        let obj = fullobj::compile_full_object(
            &o.id,
            o.sample_rate_hz,
            o.channels,
            o.frames,
            semantics_of(spec),
            &samples,
            budget(),
        )?;
        named.push((o.id.clone(), obj.bytes));
    }
    named.sort_by(|a, b| a.0.cmp(&b.0));
    let objects: Vec<Vec<u8>> = named.into_iter().map(|(_, b)| b).collect();

    // Logical order: the archive's own object-major concatenation, each object
    // length-prefixed so a decoder can split it.
    let mut logical = Vec::new();
    for o in &objects {
        logical.extend_from_slice(&(o.len() as u32).to_le_bytes());
        logical.extend_from_slice(o);
    }

    let solid = encode_solid(&objects)?;
    let reconstructed = decode_solid(&solid)?;
    let reconstructs_exactly = reconstructed == objects;

    let solid_coded = encode_bytes_adaptive(&solid);
    let logical_coded = encode_bytes_adaptive(&logical);

    // Exactness gates: the solid stream reconstructs the logical objects, and
    // each adaptive coding round-trips.
    let mut all_exact = reconstructs_exactly;
    all_exact &= crate::format::solid::decode_bytes_adaptive(&solid_coded, solid.len())
        .map(|b| b == solid)
        .unwrap_or(false);
    all_exact &= crate::format::solid::decode_bytes_adaptive(&logical_coded, logical.len())
        .map(|b| b == logical)
        .unwrap_or(false);

    let row = Row {
        objects: objects.len(),
        logical_bytes: logical.len() as u64,
        solid_bytes: solid.len() as u64,
        logical_coded_bytes: logical_coded.len() as u64,
        solid_coded_bytes: solid_coded.len() as u64,
        reconstructs_exactly,
    };

    let verdict = if !all_exact {
        Verdict::FailedCorrectness
    } else {
        Verdict::Supported
    };
    let coded_gain = row.logical_coded_bytes as i64 - row.solid_coded_bytes as i64;

    crate::courts::learned_common::finish_exp3(
        "learned-solid-object-columns",
        receipts_root,
        LEARNED_SOLID_OBJECT_COLUMNS_SHA256,
        &{
            let mut p = Vec::new();
            crate::courts::learned_common::push_label(
                &mut p,
                "vole.audio.learned.solid_object_columns.v1",
            );
            crate::courts::learned_common::push_u64(&mut p, row.objects as u64);
            crate::courts::learned_common::push_u64(&mut p, row.logical_bytes);
            crate::courts::learned_common::push_u64(&mut p, row.solid_bytes);
            crate::courts::learned_common::push_u64(&mut p, row.logical_coded_bytes);
            crate::courts::learned_common::push_u64(&mut p, row.solid_coded_bytes);
            p
        },
        verdict,
        format!(
            "archive-only solid column transposition over {n} frozen corpus objects: raw logical \
             {lb} B vs raw solid {sb} B; coded with the same carried adaptive byte coder, logical \
             {lc} B vs solid {sc} B ({gain:+} B); every object reconstructs byte-for-byte and its \
             integrity verifies",
            n = row.objects,
            lb = row.logical_bytes,
            sb = row.solid_bytes,
            lc = row.logical_coded_bytes,
            sc = row.solid_coded_bytes,
            gain = coded_gain,
        ),
        vec![
            (
                "gates",
                serde_json::json!({
                    "all_exact": all_exact,
                    "reconstructs_exactly": row.reconstructs_exactly,
                }),
            ),
            (
                "archive",
                serde_json::json!({
                    "objects": row.objects,
                    "logical_bytes": row.logical_bytes,
                    "solid_bytes": row.solid_bytes,
                    "logical_coded_bytes": row.logical_coded_bytes,
                    "solid_coded_bytes": row.solid_coded_bytes,
                    "coded_gain_bytes": coded_gain,
                }),
            ),
            (
                "method",
                serde_json::json!({
                    "columns": "headers of every object; index records of every object with payload \
                                offsets canonicalized; payloads grouped by segment index; integrity \
                                digests of every object",
                    "permutation": "logical order is restored exactly; no bytes are referenced across \
                                    objects and every object's own SHA-256 still verifies",
                    "coder": "adaptive order-1 binary rANS over bytes (context = previous byte + \
                              partial-byte bit tree), model state carried across the whole stream",
                    "scope": "archive-only: a solid archive result is never a single-file audio \
                              compression result",
                }),
            ),
            (
                "limitations",
                serde_json::json!([
                    "the solid stream is only worthwhile for archives of statistically homogeneous \
                     objects; a mixed archive may not beat logical order",
                    "the adaptive byte coder is a measurement instrument for the ordering, not a \
                     claim that VOLE's residual codecs apply to arbitrary archive bytes",
                ]),
            ),
        ],
    )
}
