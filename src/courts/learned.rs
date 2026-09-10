//! `court learned` — the Phase O aggregate (`O.17`, `O.64`).
//!
//! Runs every Phase O court in sequence and is `SUPPORTED` only when all of
//! them are. Like the other aggregates it re-runs its sub-courts so the
//! aggregate receipt is backed by fresh sub-court receipts.

use crate::error::Result;
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder};
use crate::hash::sha256::{Sha256, hex};
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const LEARNED_AGGREGATE_SHA256: &str =
    "3e3073f7948f8c7cebadc689a85665d808c6058a10aa1dcd6ab0ec922eefe849";

/// The frozen aggregate identity.
pub const PROTOCOL_SCHEMA: &str = "vole.audio.learned.aggregate.v1";

/// The Phase O courts this aggregate covers, in execution order.
pub const PHASE_O_COURTS: [&str; 13] = [
    "learned-determinism",
    "learned-residual-codec",
    "learned-linear",
    "learned-intrinsic",
    "learned-transfer",
    "learned-residual",
    "learned-quantization",
    "learned-capacity",
    "learned-shared",
    "learned-random-access",
    "learned-gpu",
    "learned-training-cost",
    "learned-inverse",
];

/// Run the aggregate; writes `receipts/learned/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut verdicts: Vec<(String, Verdict)> = Vec::with_capacity(PHASE_O_COURTS.len());
    let mut all_supported = true;
    for name in PHASE_O_COURTS {
        let verdict = super::run(name, receipts_root)?;
        all_supported &= verdict == Verdict::Supported;
        verdicts.push((name.to_string(), verdict));
    }

    let mut projection = Vec::new();
    projection.extend_from_slice(PROTOCOL_SCHEMA.as_bytes());
    projection.push(0);
    for (name, v) in &verdicts {
        projection.extend_from_slice(name.as_bytes());
        projection.push(0);
        projection.extend_from_slice(v.label().as_bytes());
        projection.push(0);
    }
    let result_hex = hex(&Sha256::digest(&projection));
    if LEARNED_AGGREGATE_SHA256.is_empty() {
        eprintln!("court learned: frozen result hash is unset; observed {result_hex}");
    } else if result_hex != LEARNED_AGGREGATE_SHA256 {
        let mut b = ReceiptBuilder::new("learned");
        b.result(Verdict::FailedCorrectness).result_detail(format!(
            "aggregate static result hash changed: frozen {LEARNED_AGGREGATE_SHA256}, observed {result_hex}"
        ));
        b.finish_write(receipts_root)?;
        eprintln!("court learned: FAILED_CORRECTNESS (static result hash changed)");
        return Ok(Verdict::FailedCorrectness);
    }

    let verdict = if all_supported {
        Verdict::Supported
    } else {
        Verdict::Inconclusive
    };
    let breakdown: Vec<serde_json::Value> = verdicts
        .iter()
        .map(|(n, v)| serde_json::json!({ "court": n, "verdict": v.label() }))
        .collect();
    let mut builder = ReceiptBuilder::new("learned");
    builder
        .result(verdict)
        .result_detail(format!(
            "Phase O aggregate: {} of {} sub-courts SUPPORTED; result sha256 {result_hex}",
            verdicts
                .iter()
                .filter(|(_, v)| *v == Verdict::Supported)
                .count(),
            verdicts.len(),
        ))
        .params(CourtParams {
            universe: Some(crate::learned::profile::LEARNED_UNIVERSE.into()),
            profile: Some(crate::learned::profile::LEARNED_PROFILE.into()),
            content_kind: Some("Phase O aggregate over its member courts".into()),
            ..Default::default()
        })
        .provenance(Provenance {
            reference_hash: Some(result_hex.clone()),
            ..Default::default()
        })
        .extra(
            "aggregate",
            serde_json::json!({
                "protocol_schema": PROTOCOL_SCHEMA,
                "courts": PHASE_O_COURTS,
                "verdicts": breakdown,
            }),
        )
        .limitation(
            "an aggregate is only as strong as its members: a member reporting an honest negative \
             (for example a learned family that loses to a simple predictor, or a device surface \
             that is unavailable) is not an aggregate failure, so the aggregate is SUPPORTED when \
             every member is SUPPORTED and otherwise INCONCLUSIVE",
        );
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court learned: {verdict}");
    for (n, v) in &verdicts {
        println!("  {n}: {v}");
    }
    println!("  result sha256: {result_hex}");
    println!("  receipt: {}", path.display());
    Ok(verdict)
}
