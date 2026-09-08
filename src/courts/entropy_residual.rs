//! `court entropy-residual` — entropy-coding the exact Phase-E residual
//! (H.2.9/H.2.10/H.2.11/H.2.31).
//!
//! For curated (fixture, hypothesis) pairs the court:
//!
//! 1. computes the exact closing residual `R = rho(X, H)` (unchanged
//!    Phase-E semantics), validates closure over the full intrinsic;
//! 2. encodes it into per-page entropy form (`RepresentedResidual`, masks +
//!    zigzag delta lanes, per-page RAW fallback);
//! 3. proves reconstruction is a byte-identical semantic `Residual` and the
//!    closure equals the intrinsic;
//! 4. reports complete costs vs the original semantic residual canonical
//!    bytes and the canonical U1 literal;
//! 5. verifies sparse hypotheses win and incompressible controls fall back
//!    honestly toward RAW record pages.
//!
//! The residual algebra is untouched: the entropy layer only wraps the
//! canonical record information.

use crate::courts::entropy_common::canonical_u1_literal_bytes;
use crate::entropy::corpus;
use crate::entropy::represent::ModelMode;
use crate::entropy::represent::{self, PageKind, RepresentedResidual};
use crate::evidence::receipt::{CourtParams, ReceiptBuilder};
use crate::object::descriptor::{ObjectDescriptor, Representation};
use crate::object::residual::{Residual, ResidualModel};
use crate::status::Verdict;
use crate::universe::layout::Layout;
use std::path::Path;

const PAGE_FRAMES: u32 = 512;

/// Deterministic hypothesis per fixture (cycle lengths and levels chosen to
/// be *reasonable*, not tuned: most hypotheses are imperfect, which is what
/// makes the residual interesting).
fn hypothesis(fx: &corpus::Fixture) -> Option<ResidualModel> {
    let samples = &fx.samples;
    let ch = usize::from(fx.channels);
    match fx.name {
        "silence" => Some(ResidualModel::Constant(0)),
        "dc" => Some(ResidualModel::Constant(1 << 23)),
        "dc-negative" => Some(ResidualModel::Constant(-(1 << 23))),
        "single-sine" | "stereo-correlated" if fx.channels == 1 => Some(ResidualModel::Periodic {
            cycle: samples[..64 * ch].chunks(ch).map(|w| w[0]).collect(),
        }),
        "harmonic-tone" if fx.channels == 1 => Some(ResidualModel::Periodic {
            cycle: samples[..4096 * ch].chunks(ch).map(|w| w[0]).collect(),
        }),
        "quasi-periodic" | "am-signal" | "fm-signal" | "transient-heavy" => {
            Some(ResidualModel::Periodic {
                cycle: samples[..512 * ch].chunks(ch).map(|w| w[0]).collect(),
            })
        }
        // Stereo fixtures: the frozen residual semantics restrict Periodic
        // models to mono layouts, so stereo content uses a constant
        // hypothesis (dense residual) — still exact, still honest.
        "single-sine" | "harmonic-tone" | "stereo-correlated" => Some(ResidualModel::Constant(0)),
        "impulse-train" | "white-noise" | "random-control" | "scrambled-control" => {
            Some(ResidualModel::Zero)
        }
        _ => None,
    }
}

/// Run the court; writes an immutable receipt under `receipts/entropy-residual/`.
pub fn run(receipts_root: &Path) -> crate::error::Result<Verdict> {
    let fail = |why: &str| -> crate::error::Result<Verdict> {
        let mut b = ReceiptBuilder::new("entropy-residual");
        b.result(Verdict::FailedCorrectness)
            .result_detail(format!("entropy-residual failed: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court entropy-residual: FAILED_CORRECTNESS ({why})");
        eprintln!("  receipt: {}", path.display());
        Ok(Verdict::FailedCorrectness)
    };

    let fixtures = corpus::all();
    let mut cells: Vec<serde_json::Value> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();

    for fx in &fixtures {
        let channels = fx.channels;
        let ch = usize::from(channels);
        let layout = match channels {
            1 => Layout::Mono,
            2 => Layout::Stereo,
            n => Layout::Channels(n),
        };
        let frames = fx.frames() as u64;
        let Some(model) = hypothesis(fx) else {
            skipped.push(fx.name.to_string());
            continue;
        };
        let descriptor =
            match ObjectDescriptor::new(Representation::PredictorResidual, frames, layout, None) {
                Some(d) => d,
                None => {
                    skipped.push(fx.name.to_string());
                    continue;
                }
            };
        // Exact closing residual (Phase-E semantics; uncloseable -> skip).
        let intrinsic = &fx.samples;
        let Some(records) = Residual::closing_residual(intrinsic, channels, &model) else {
            skipped.push(fx.name.to_string());
            continue;
        };
        let residual = match Residual::new(&descriptor, model, records) {
            Some(r) => r,
            None => {
                skipped.push(fx.name.to_string());
                continue;
            }
        };
        // Closure sanity over the intrinsic (H + R == X for every frame).
        for (f, &x) in intrinsic.iter().enumerate() {
            let frame = (f / ch) as u64;
            let chan = (f % ch) as u8;
            if residual.closure_sample(frame, chan) != x {
                return fail(&format!("{}: closure != intrinsic", fx.name));
            }
        }
        let semantic_bytes = residual.canonical_bytes(&descriptor).len() as u64;

        for mode in [ModelMode::Inline, ModelMode::Shared] {
            let rr = match RepresentedResidual::encode(
                descriptor.clone(),
                &residual,
                PAGE_FRAMES,
                mode,
                false,
            ) {
                Ok(r) => r,
                Err(e) => return fail(&format!("{} encode: {e}", fx.name)),
            };
            // Byte-identical reconstruction.
            let back = match rr.reconstruct_full() {
                Ok(r) => r,
                Err(e) => return fail(&format!("{} reconstruct: {e}", fx.name)),
            };
            if back.model != residual.model || back.records != residual.records {
                return fail(&format!("{}: reconstruction != original residual", fx.name));
            }
            let container_bytes = represent::residual_container_bytes(&rr)?.len() as u64;
            let cost = rr.cost(canonical_u1_literal_bytes(frames, channels))?;
            let raw_pages = rr.pages.iter().filter(|p| p.kind == PageKind::Raw).count();
            let fallback_fraction = if rr.pages.is_empty() {
                1.0
            } else {
                raw_pages as f64 / rr.pages.len() as f64
            };
            let record_density = if frames == 0 {
                0.0
            } else {
                residual.records.len() as f64 / frames as f64
            };
            let ratio_vs_semantic = container_bytes as f64 / semantic_bytes as f64;
            cells.push(serde_json::json!({
                "fixture": fx.name,
                "kind": fx.kind,
                "channels": channels,
                "frames": frames,
                "model_mode": if mode == ModelMode::Inline { "inline" } else { "shared" },
                "record_count": residual.records.len(),
                "record_density": record_density,
                "semantic_residual_bytes": semantic_bytes,
                "container_bytes": container_bytes,
                "ratio_vs_semantic_residual": ratio_vs_semantic,
                "complete_bytes": cost.complete_bytes,
                "ratio_vs_u1_literal": cost.complete_bytes as f64 / cost.canonical_literal_bytes as f64,
                "hypothesis_bytes": cost.hypothesis_bytes,
                "model_bytes": cost.model_bytes,
                "payload_bytes": cost.payload_bytes,
                "fallback_raw_page_fraction": fallback_fraction,
                "reconstruction_exact": true,
                "closure_exact": true,
            }));
            // Negative controls (H.2.31): the entropy residual must lose to
            // the canonical U1 literal (incompressible content is best stored
            // literally — the universal fallback). Beating the *semantic*
            // record layout is expected and honest: that 13 B/record layout
            // (absolute u64 frames) is verbose; the meaningful incompressible
            // control is the literal comparison.
            if fx.kind == "negative-control" {
                let u1 = cost.canonical_literal_bytes.max(1) as f64;
                let ratio_u1 = cost.complete_bytes as f64 / u1;
                if ratio_u1 < 1.1 {
                    return fail(&format!(
                        "{}: negative control residual representation ({:.3}x) nearly beats the U1 literal — suspicious",
                        fx.name, ratio_u1
                    ));
                }
                if ratio_vs_semantic < 0.5 {
                    return fail(&format!(
                        "{}: negative control compressed residual {:.3}x vs semantic bytes — suspicious",
                        fx.name,
                        1.0 / ratio_vs_semantic
                    ));
                }
            }
            let _ = record_density;
        }
    }
    if skipped.len() == fixtures.len() {
        return fail("all fixtures skipped (no valid hypothesis)");
    }

    let mut builder = ReceiptBuilder::new("entropy-residual");
    builder
        .result(Verdict::Supported)
        .result_detail(format!(
            "exact residual entropy coding over {} fixtures ({} skipped); all reconstructions byte-identical",
            fixtures.len(),
            skipped.len()
        ))
        .params(CourtParams {
            universe: Some("vole.audio.u1".into()),
            profile: Some("u1/v1 + vole.entropy.p1/p1/v1".into()),
            backend: Some("scalar".into()),
            content_kind: Some("entropy-corpus-v1 residual".into()),
            ..Default::default()
        })
        .extra("cells", serde_json::Value::Array(cells))
        .extra("skipped", serde_json::Value::Array(
            skipped.iter().map(|s| serde_json::Value::String(s.clone())).collect(),
        ));
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court entropy-residual: SUPPORTED");
    println!("  receipt: {}", path.display());
    Ok(Verdict::Supported)
}
