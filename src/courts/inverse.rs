//! `court inverse` — bounded inverse proceduralization (Phase K).
//!
//! For every fixture of the frozen H.2 corpus (truncated to a bounded inverse
//! window) the court runs the inverse compiler and reports, per candidate:
//!
//! * exact acceptance (intrinsic closure **and** scalar-oracle observation
//!   both equal the observed window, plus a bounded seek window);
//! * the complete cost (H.2 cost oracle where an entropy representation
//!   exists) and the abstract universe work;
//! * the measured proposal/materialization/seek times and accounted memory;
//! * frontier membership.
//!
//! Two questions are kept separate, because they are different questions:
//!
//! * **explanation** — the cheapest exact deterministic explanation of the
//!   window *from scratch* (the main search runs against an empty library, so
//!   its Pareto frontier is the hypothesis frontier); and
//! * **deduplication** — once content is already in the archive, the cheapest
//!   exact representation is a shared reference (32 dependency bytes, zero
//!   persistent bytes).
//!
//! Court assertions (all independent of hardware):
//!
//! 1. `Literal` is always accepted (the universal fallback is never optional);
//! 2. every accepted candidate is exact on all three checks (a candidate that
//!    is merely close must never be accepted);
//! 3. the frontier is a valid Pareto set and every accepted candidate is either
//!    a member or dominated by one;
//! 4. silence/DC fixtures have a non-literal explanation strictly cheaper than
//!    the literal floor, and at least five fixtures overall do;
//! 5. negative controls (white noise / random / scrambled) are **never**
//!    "compressed" by a hypothesis: the cheapest accepted candidate stays
//!    within 10% of the literal floor (anti-self-deception gate);
//! 6. the search is deterministic: a second compile yields identical static
//!    results;
//! 7. archive dedup works: every fixture is accepted as an exact shared
//!    reference once it is in the library, and a *procedural* library object
//!    can be referenced too.

use crate::entropy::corpus;
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder};
use crate::evidence::timing::Stopwatch;
use crate::hash::sha256::Sha256;
use crate::inverse::{
    Acceptance, CandidateKind, Intrinsic, ReferenceLibrary, SearchBudget, SearchReport,
};
use crate::object::descriptor::{ObjectDescriptor, Representation};
use crate::object::{Cycle, ObjectData};
use crate::status::Verdict;
use crate::universe::layout::Layout;
use std::path::Path;

/// Inverse window per fixture (frames). Bounded, frozen.
pub const INVERSE_FRAMES: usize = 4096;
/// Nominal rate for the receipt params (content classes are rate-independent).
pub const INVERSE_RATE_HZ: u32 = 48_000;
/// Frozen static-result hash: the court fails if a change silently alters the
/// inverse results. Re-freeze only with a documented reason.
pub const INVERSE_RESULT_SHA256: &str =
    "5b83600646c7e013e74767ff41e4dcad3fdc36199a264b360a1a35a5448354af";

/// Build the bounded inverse fixture set from the frozen corpus.
pub fn fixtures() -> crate::error::Result<Vec<Intrinsic>> {
    let mut out = Vec::new();
    for fx in corpus::all() {
        let ch = usize::from(fx.channels);
        let frames = fx.frames().min(INVERSE_FRAMES);
        let samples = fx.samples[..frames * ch].to_vec();
        out.push(Intrinsic::new(fx.name, fx.channels, samples)?);
    }
    Ok(out)
}

/// The reference library: the canonical literal object of every fixture (a
/// prior archive).
pub fn library_for(fixtures: &[Intrinsic]) -> crate::error::Result<ReferenceLibrary> {
    let mut lib = ReferenceLibrary::new();
    for fx in fixtures {
        lib.register_literal(fx.channels, fx.samples.clone())?;
    }
    Ok(lib)
}

/// Deterministic static-result hash over every report (no measured quantity),
/// used as the court's frozen reference vector.
fn static_result_hash(reports: &[SearchReport]) -> [u8; 32] {
    let mut h = Sha256::new();
    for r in reports {
        h.update(r.intrinsic_name.as_bytes());
        h.update(&[0]);
        h.update(&r.intrinsic_sha256);
        h.update(&r.frames.to_le_bytes());
        h.update(&[r.channels]);
        h.update(&(r.proposed as u64).to_le_bytes());
        h.update(&(r.rejected as u64).to_le_bytes());
        for a in &r.accepted {
            h.update(&[a.kind.tag()]);
            h.update(a.label.as_bytes());
            h.update(&[0]);
            h.update(&a.cost.complete_bytes.to_le_bytes());
            h.update(&a.cost.dependency_bytes.to_le_bytes());
            h.update(&a.cost.persistent_sample_domain_bytes.to_le_bytes());
            h.update(&a.total_ops.to_le_bytes());
            h.update(&a.seek_ops.to_le_bytes());
            h.update(&a.content_id.to_bytes());
        }
        h.update(&[0xEE]);
        for i in r.frontier.indices() {
            h.update(&(*i as u64).to_le_bytes());
        }
        h.update(&[0xFF]);
    }
    h.finalize()
}

fn acceptance_cell(a: &Acceptance) -> serde_json::Value {
    serde_json::json!({
        "kind": a.kind.name(),
        "label": a.label,
        "content_id": a.content_id.to_string(),
        "reference_target": a.reference_target.map(|c| c.to_string()),
        "complete_bytes": a.cost.complete_bytes,
        "metadata_bytes": a.cost.metadata_bytes,
        "hypothesis_bytes": a.cost.hypothesis_bytes,
        "model_bytes": a.cost.model_bytes,
        "payload_bytes": a.cost.payload_bytes,
        "index_bytes": a.cost.index_bytes,
        "checkpoint_bytes": a.cost.checkpoint_bytes,
        "dependency_bytes": a.cost.dependency_bytes,
        "integrity_bytes": a.cost.integrity_bytes,
        "cost_source": a.cost.cost_source,
        "raw_sample_bytes": a.cost.raw_sample_bytes,
        "canonical_literal_bytes": a.cost.canonical_literal_bytes,
        "persistent_sample_domain_bytes": a.cost.persistent_sample_domain_bytes,
        "decoded_sample_state_bytes": a.cost.decoded_sample_state_bytes,
        "decoded_residual_state_bytes": a.cost.decoded_residual_state_bytes,
        "decoded_window_state_bytes": a.cost.decoded_window_state_bytes,
        "generator_ops": a.work.generator_ops,
        "residual_ops": a.work.residual_ops,
        "lookup_ops": a.work.lookup_ops,
        "filter_ops": a.work.filter_ops,
        "total_ops": a.total_ops,
        "seek_ops": a.seek_ops,
        "intrinsic_exact": a.intrinsic_exact,
        "evaluator_exact": a.evaluator_exact,
        "seek_exact": a.seek_exact,
        "materialize_ns": a.materialize_ns,
        "seek_latency_ns": a.seek_latency_ns,
        "intrinsic_ns": a.intrinsic_ns,
        "seek_start": a.seek_start,
        "seek_frames": a.seek_frames,
        "alloc": {
            "search_input_bytes": a.alloc.search_input_bytes,
            "candidate_semantic_state_bytes": a.alloc.candidate_semantic_state_bytes,
            "intrinsic_reconstruction_peak": a.alloc.intrinsic_reconstruction_peak,
            "oracle_observation_peak": a.alloc.oracle_observation_peak,
            "seek_observation_peak": a.alloc.seek_observation_peak,
        },
        "proposal_ns": a.proposal_ns,
    })
}

/// Run the court; writes an immutable receipt under `receipts/inverse/`.
pub fn run(receipts_root: &Path) -> crate::error::Result<Verdict> {
    let fail = |why: &str| -> crate::error::Result<Verdict> {
        let mut b = ReceiptBuilder::new("inverse");
        b.result(Verdict::FailedCorrectness)
            .result_detail(format!("inverse compiler failed: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court inverse: FAILED_CORRECTNESS ({why})");
        eprintln!("  receipt: {}", path.display());
        Ok(Verdict::FailedCorrectness)
    };

    let fixtures = fixtures()?;
    let budget = SearchBudget::default();
    // The explanation search runs against an EMPTY library: it answers "what is
    // the cheapest exact deterministic explanation from scratch?".
    let empty = ReferenceLibrary::new();

    let sw = Stopwatch::start();
    let mut reports: Vec<SearchReport> = Vec::with_capacity(fixtures.len());
    let mut cells: Vec<serde_json::Value> = Vec::new();
    let mut non_literal_wins = 0u64;

    for fx in &fixtures {
        let report = crate::inverse::compile(fx, &empty, budget)?;

        // (1) Literal is always accepted.
        if !report.literal_accepted() {
            return fail(&format!("{}: literal fallback was not accepted", fx.name));
        }
        // (2) Every accepted candidate is exact on all three checks.
        for a in &report.accepted {
            if !(a.intrinsic_exact && a.evaluator_exact && a.seek_exact) {
                return fail(&format!(
                    "{}: accepted candidate '{}' is not exact (intrinsic={}, evaluator={}, seek={})",
                    fx.name, a.label, a.intrinsic_exact, a.evaluator_exact, a.seek_exact
                ));
            }
            if a.kind == CandidateKind::SharedReference {
                return fail(&format!(
                    "{}: an empty library must never propose a shared reference",
                    fx.name
                ));
            }
        }
        // (3) Frontier validity and coverage.
        if !report.frontier.is_valid(&report.accepted) {
            return fail(&format!("{}: frontier is not a valid Pareto set", fx.name));
        }
        for (i, a) in report.accepted.iter().enumerate() {
            if report.frontier.contains(i) {
                continue;
            }
            let dominated = report
                .accepted
                .iter()
                .enumerate()
                .any(|(j, b)| j != i && crate::inverse::frontier::dominates(b, a));
            if !dominated {
                return fail(&format!(
                    "{}: accepted candidate '{}' is neither on the frontier nor dominated",
                    fx.name, a.label
                ));
            }
        }

        let literal = report
            .accepted
            .iter()
            .find(|a| a.kind == CandidateKind::Literal)
            .expect("literal accepted");
        let literal_complete = literal.cost.complete_bytes;
        let cheapest = report.cheapest().expect("accepted non-empty");
        let cheapest_non_literal = report
            .accepted
            .iter()
            .filter(|a| a.kind != CandidateKind::Literal)
            .min_by_key(|a| a.cost.complete_bytes);

        let is_negative = fx.name.contains("noise")
            || fx.name.contains("random")
            || fx.name.contains("scrambled");

        if is_negative {
            // (5) Negative controls must not be "compressed" by a hypothesis.
            if cheapest.cost.complete_bytes * 10 < literal_complete * 9 {
                return fail(&format!(
                    "{}: negative control '{}' beats the literal floor by more than 10% \
                     ({} vs {}) — suspicious",
                    fx.name, cheapest.label, cheapest.cost.complete_bytes, literal_complete
                ));
            }
        } else if fx.name == "silence" || fx.name == "dc" || fx.name == "dc-negative" {
            // (4) These have unambiguous non-literal explanations.
            let win = cheapest_non_literal
                .map(|a| a.cost.complete_bytes < literal_complete)
                .unwrap_or(false);
            if !win {
                return fail(&format!(
                    "{}: no non-literal explanation beats the literal floor",
                    fx.name
                ));
            }
        }
        if cheapest.kind != CandidateKind::Literal {
            non_literal_wins += 1;
        }

        cells.push(serde_json::json!({
            "fixture": fx.name,
            "channels": fx.channels,
            "frames": fx.frames,
            "intrinsic_sha256": crate::hash::sha256::hex(&report.intrinsic_sha256),
            "proposed": report.proposed,
            "evaluated": report.evaluated,
            "rejected": report.rejected,
            "accepted": report.accepted.len(),
            "frontier": report.frontier.len(),
            "literal_complete_bytes": literal_complete,
            "cheapest": cheapest.label,
            "cheapest_complete_bytes": cheapest.cost.complete_bytes,
            "cheapest_ratio_vs_literal": cheapest.cost.complete_bytes as f64
                / literal_complete.max(1) as f64,
            "cheapest_on_frontier": report
                .cheapest_on_frontier()
                .map(|a| a.label.clone()),
            "negative_control": is_negative,
            "search_ns": report.search_ns,
            "candidates": report.accepted.iter().map(acceptance_cell).collect::<Vec<_>>(),
        }));

        // (6) Determinism: a second compile is statically identical.
        let again = crate::inverse::compile(fx, &empty, budget)?;
        if static_projection(&report) != static_projection(&again) {
            return fail(&format!("{}: inverse search is not deterministic", fx.name));
        }

        reports.push(report);
    }

    if non_literal_wins < 5 {
        return fail(&format!(
            "only {non_literal_wins} fixtures found a non-literal explanation; \
             the search is not doing useful work"
        ));
    }

    let result_hash = static_result_hash(&reports);
    let result_hex = crate::hash::sha256::hex(&result_hash);
    if INVERSE_RESULT_SHA256.is_empty() {
        eprintln!("court inverse: frozen result hash is unset; observed {result_hex}");
    } else if result_hex != INVERSE_RESULT_SHA256 {
        return fail(&format!(
            "static result hash changed: frozen {INVERSE_RESULT_SHA256}, observed {result_hex}"
        ));
    }

    // (7) Archive dedup: compile every fixture against a library containing
    // its own content and require an accepted exact shared reference.
    let archive = library_for(&fixtures)?;
    let mut dedup_cells = Vec::new();
    let mut dedup_saved = 0i64;
    for fx in &fixtures {
        let r = crate::inverse::compile(fx, &archive, budget)?;
        let reference = r
            .accepted
            .iter()
            .find(|a| a.kind == CandidateKind::SharedReference)
            .ok_or_else(|| {
                crate::error::Error::internal(format!(
                    "{}: archive library did not yield a shared reference",
                    fx.name
                ))
            })?;
        if reference.cost.dependency_bytes != 32
            || reference.cost.persistent_sample_domain_bytes != 0
        {
            return fail(&format!(
                "{}: shared reference accounting wrong (dep={}, persistent={})",
                fx.name,
                reference.cost.dependency_bytes,
                reference.cost.persistent_sample_domain_bytes
            ));
        }
        let literal = r
            .accepted
            .iter()
            .find(|a| a.kind == CandidateKind::Literal)
            .expect("literal");
        dedup_saved += literal.cost.complete_bytes as i64 - reference.cost.complete_bytes as i64;
        dedup_cells.push(serde_json::json!({
            "fixture": fx.name,
            "reference_complete_bytes": reference.cost.complete_bytes,
            "reference_dependency_bytes": reference.cost.dependency_bytes,
            "literal_complete_bytes": literal.cost.complete_bytes,
            "saved_bytes": literal.cost.complete_bytes as i64
                - reference.cost.complete_bytes as i64,
        }));
    }

    // A procedural library entry must be referenceable too.
    let procedural = procedural_library_reference(&fixtures)?;
    if !procedural.accepted {
        return fail("procedural library entry was not usable as a shared reference");
    }

    let total_ns = sw.elapsed_ns().max(0) as u64;
    let params = CourtParams {
        universe: Some("vole.audio.u1".into()),
        profile: Some("u1/v1 + vole.inverse.k1".into()),
        backend: Some("scalar".into()),
        sample_rate_hz: Some(INVERSE_RATE_HZ),
        channels: Some(0),
        quantum_frames: Some(INVERSE_FRAMES as u32),
        content_kind: Some("entropy-corpus-v1 (inverse window)".into()),
        ..Default::default()
    };
    let mut builder = ReceiptBuilder::new("inverse");
    builder
        .result(Verdict::Supported)
        .result_detail(format!(
            "inverse compiler over {} fixtures; {} non-literal explanations; \
             archive dedup saves {dedup_saved} bytes; result sha256 {result_hex}",
            fixtures.len(),
            non_literal_wins
        ))
        .params(params)
        .provenance(Provenance {
            reference_hash: Some(result_hex.clone()),
            ..Default::default()
        })
        .extra("fixtures", serde_json::json!(fixtures.len()))
        .extra("archive_objects", serde_json::json!(archive.len()))
        .extra("non_literal_wins", serde_json::json!(non_literal_wins))
        .extra(
            "budget",
            serde_json::json!({
                "max_period_scan": budget.max_period_scan,
                "max_residual_period_candidates": budget.max_residual_period_candidates,
                "max_candidates": budget.max_candidates,
                "placement": budget.placement.label(),
            }),
        )
        .extra("procedural_library_reference", procedural.cell)
        .extra("dedup_saved_bytes", serde_json::json!(dedup_saved))
        .extra("dedup", serde_json::Value::Array(dedup_cells))
        .extra("total_search_ns", serde_json::json!(total_ns))
        .extra("cells", serde_json::Value::Array(cells));
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court inverse: SUPPORTED");
    println!("  fixtures: {}", fixtures.len());
    println!("  non-literal explanations: {non_literal_wins}");
    println!("  archive dedup saved: {dedup_saved} bytes");
    println!("  result sha256: {result_hex}");
    println!("  receipt: {}", path.display());
    Ok(Verdict::Supported)
}

/// Static projection used for the determinism check.
fn static_projection(r: &SearchReport) -> Vec<u8> {
    let mut out = Vec::new();
    for a in &r.accepted {
        out.push(a.kind.tag());
        out.extend_from_slice(a.label.as_bytes());
        out.push(0);
        out.extend_from_slice(&a.cost.complete_bytes.to_le_bytes());
        out.extend_from_slice(&a.total_ops.to_le_bytes());
        out.extend_from_slice(&a.seek_ops.to_le_bytes());
    }
    out.push(0xEE);
    for i in r.frontier.indices() {
        out.extend_from_slice(&(*i as u64).to_le_bytes());
    }
    out
}

struct ProceduralReference {
    accepted: bool,
    cell: serde_json::Value,
}

/// Demonstrate a shared reference to a *procedural* library object: register a
/// periodic fixture's exact period as an `ExactRepeat` object and require the
/// compiler to find and accept it as a reference target.
fn procedural_library_reference(
    fixtures: &[Intrinsic],
) -> crate::error::Result<ProceduralReference> {
    // A mono periodic fixture (not silence/DC, so the library entry is a real
    // procedural object rather than a degenerate constant).
    let Some(fx) = fixtures
        .iter()
        .find(|f| f.channels == 1 && f.name == "impulse-train")
        .or_else(|| fixtures.iter().find(|f| f.channels == 1))
    else {
        return Ok(ProceduralReference {
            accepted: false,
            cell: serde_json::json!({ "available": false, "reason": "no mono fixture" }),
        });
    };
    let frames = fx.frames as usize;
    let period = minimal_period(&fx.samples, frames);
    if period >= frames {
        return Ok(ProceduralReference {
            accepted: false,
            cell: serde_json::json!({
                "available": false,
                "reason": "fixture has no exact period smaller than its extent",
                "period": period,
            }),
        });
    }
    let mut lib = ReferenceLibrary::new();
    let layout = Layout::Mono;
    let desc = ObjectDescriptor::new(Representation::ExactRepeat, period as u64, layout, None)
        .ok_or_else(|| crate::error::Error::malformed("procedural library descriptor"))?;
    let cycle = Cycle::new(&desc, fx.samples[..period].to_vec())
        .ok_or_else(|| crate::error::Error::malformed("procedural library cycle"))?;
    let content = lib.register(desc, ObjectData::ExactRepeat(cycle))?;

    let report = crate::inverse::compile(fx, &lib, SearchBudget::default())?;
    let accepted = report
        .accepted
        .iter()
        .find(|a| a.kind == CandidateKind::SharedReference);
    let matches_registered = accepted
        .map(|a| a.reference_target == Some(content))
        .unwrap_or(false);
    let cell = serde_json::json!({
        "fixture": fx.name,
        "period": period,
        "registered_content_id": content.to_string(),
        "accepted": accepted.is_some(),
        "matches_registered_object": matches_registered,
        "dependency_bytes": accepted.map(|a| a.cost.dependency_bytes),
        "complete_bytes": accepted.map(|a| a.cost.complete_bytes),
    });
    Ok(ProceduralReference {
        accepted: accepted.is_some() && matches_registered,
        cell,
    })
}

/// Minimal exact frame period of a mono window (for the library entry).
fn minimal_period(samples: &[i32], frames: usize) -> usize {
    let mut pi = vec![0usize; frames];
    for i in 1..frames {
        let mut k = pi[i - 1];
        while k > 0 && samples[i] != samples[k] {
            k = pi[k - 1];
        }
        if samples[i] == samples[k] {
            k += 1;
        }
        pi[i] = k;
    }
    frames - pi[frames - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixtures_are_the_frozen_corpus_window() {
        let f = fixtures().unwrap();
        assert_eq!(f.len(), corpus::all().len());
        for fx in &f {
            assert_eq!(
                fx.samples.len(),
                fx.frames as usize * usize::from(fx.channels)
            );
            assert!(fx.frames as usize <= INVERSE_FRAMES);
        }
    }

    #[test]
    fn literal_is_always_accepted_and_search_is_deterministic() {
        let fixtures = fixtures().unwrap();
        let budget = SearchBudget::default();
        for fx in &fixtures {
            let empty = ReferenceLibrary::new();
            let a = crate::inverse::compile(fx, &empty, budget).unwrap();
            assert!(a.literal_accepted(), "{}: literal not accepted", fx.name);
            for c in &a.accepted {
                assert!(c.intrinsic_exact && c.evaluator_exact && c.seek_exact);
            }
            assert!(a.frontier.is_valid(&a.accepted));
            let b = crate::inverse::compile(fx, &empty, budget).unwrap();
            assert_eq!(static_projection(&a), static_projection(&b));
        }
    }

    #[test]
    fn silence_and_dc_have_cheap_procedural_explanations() {
        let fixtures = fixtures().unwrap();
        let budget = SearchBudget::default();
        for name in ["silence", "dc", "dc-negative"] {
            let fx = fixtures.iter().find(|f| f.name == name).unwrap();
            let r = crate::inverse::compile(fx, &ReferenceLibrary::new(), budget).unwrap();
            let literal = r
                .accepted
                .iter()
                .find(|a| a.kind == CandidateKind::Literal)
                .unwrap();
            let best_non_literal = r
                .accepted
                .iter()
                .filter(|a| a.kind != CandidateKind::Literal)
                .min_by_key(|a| a.cost.complete_bytes)
                .unwrap();
            assert!(
                best_non_literal.cost.complete_bytes < literal.cost.complete_bytes,
                "{name}: non-literal {} did not beat literal {}",
                best_non_literal.cost.complete_bytes,
                literal.cost.complete_bytes
            );
        }
    }

    #[test]
    fn frozen_result_hash_is_stable() {
        let fixtures = fixtures().unwrap();
        let budget = SearchBudget::default();
        let reports: Vec<SearchReport> = fixtures
            .iter()
            .map(|fx| crate::inverse::compile(fx, &ReferenceLibrary::new(), budget).unwrap())
            .collect();
        let h = crate::hash::sha256::hex(&static_result_hash(&reports));
        assert_eq!(
            h, INVERSE_RESULT_SHA256,
            "the inverse result vector changed (actual {h}); re-freeze only with a \
             documented reason"
        );
    }

    #[test]
    fn archive_library_yields_exact_shared_references() {
        let fixtures = fixtures().unwrap();
        let archive = library_for(&fixtures).unwrap();
        for fx in &fixtures {
            let r = crate::inverse::compile(fx, &archive, SearchBudget::default()).unwrap();
            let reference = r
                .accepted
                .iter()
                .find(|a| a.kind == CandidateKind::SharedReference)
                .unwrap_or_else(|| panic!("{}: no shared reference", fx.name));
            assert_eq!(reference.cost.dependency_bytes, 32);
            assert_eq!(reference.cost.persistent_sample_domain_bytes, 0);
            assert_eq!(reference.cost.complete_bytes, 103);
            assert!(reference.cost.decomposition_is_consistent());
        }
    }
}
