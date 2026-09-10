//! `court negative` — negative controls and hostile inputs (Phase M, contract §44).
//!
//! A compression claim is only as credible as its negative controls. This court
//! does two things:
//!
//! 1. **Incompressible material must not produce fake wins.** For the frozen
//!    corpus's full-width-random and scrambled objects it reports B0 raw, B1
//!    FLAC-5 and the selected full-object VOLE cost, and every ratio is
//!    recomputed from the stored bytes so no number can drift from its artifact.
//! 2. **Hostile archives must be rejected, not survived.** The full-object
//!    container is truncated, bit-flipped, structurally mutated **and resealed**,
//!    given an allocation bomb and handed malformed headers; every one must fail
//!    with a typed error and none may panic or allocate unboundedly.
//!
//! It reports what the material actually does. It does not assert that VOLE wins
//! or loses: on incompressible data both codecs are expected to lose to raw
//! framing once framing is included, and that is the honest result to print.

use crate::baseline::{B1_LEVEL_PRIMARY, b1_flac_artifact};
use crate::corpus;
use crate::corpus::generate::{self, Spec};
use crate::error::{Error, Result};
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder};
use crate::evidence::timing::Stopwatch;
use crate::fullobj::{self, FullSemantics};
use crate::hash::sha256::{Sha256, hex};
use crate::inverse::SearchBudget;
use crate::status::Verdict;
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

/// Frozen static-result hash (empty means "not yet frozen").
pub const NEGATIVE_RESULT_SHA256: &str =
    "cca0f1627f8420d32d7710a5bfd84062b0e1b7dddc9a610322971fb62ba6849b";

/// The frozen negative-control protocol identity.
pub const PROTOCOL_SCHEMA: &str = "vole.audio.negative.protocol.v1";

/// Entropy classes treated as incompressible controls.
const INCOMPRESSIBLE_CLASSES: [&str; 2] = ["full_width_random", "scrambled"];

fn budget() -> SearchBudget {
    SearchBudget::default()
}

/// Recompute a container's trailing integrity digest so a mutation reaches the
/// parser rather than failing the outer digest.
fn reseal(bytes: &mut [u8]) {
    let body = bytes.len() - fullobj::INTEGRITY_BYTES as usize;
    let digest = Sha256::digest(&bytes[..body]);
    bytes[body..].copy_from_slice(&digest);
}

/// True when the candidate is rejected at parse or materialization time.
fn rejected(bytes: &[u8]) -> bool {
    fullobj::decode_full_object(bytes).is_err() || fullobj::materialize_full_object(bytes).is_err()
}

/// A structural field mutation applied to a serialized container.
type Mutation = Box<dyn Fn(&mut [u8])>;

/// Run a hostile candidate through the parser under `catch_unwind`, requiring
/// rejection and no panic.
fn must_reject(candidate: &[u8], label: &str) -> Result<()> {
    let outcome = catch_unwind(AssertUnwindSafe(|| rejected(candidate)));
    match outcome {
        Ok(true) => Ok(()),
        Ok(false) => Err(Error::internal(format!(
            "hostile container survived: {label}"
        ))),
        Err(_) => Err(Error::internal(format!("parser panicked on: {label}"))),
    }
}

/// The hostile battery. Returns `(integrity_hostile, structural_hostile, other)`.
fn hostile_battery(valid: &[u8]) -> Result<(usize, usize, usize)> {
    // The unmutated container must be usable (a control against a broken fixture).
    fullobj::materialize_full_object(valid)?;

    let mut integrity = 0usize;
    let mut structural = 0usize;
    let mut other = 0usize;

    // 1. Truncations (integrity-hostile: the trailing digest is gone).
    for cut in [1usize, 8, fullobj::HEADER_BYTES, valid.len() / 2] {
        if cut < valid.len() {
            must_reject(&valid[..valid.len() - cut], &format!("truncate {cut}"))?;
            integrity += 1;
        }
    }
    // 2. Bit flips without resealing (must fail the outer digest).
    for at in [0usize, 15, fullobj::HEADER_BYTES, valid.len() - 1] {
        if at < valid.len() {
            let mut b = valid.to_vec();
            b[at] ^= 0xFF;
            must_reject(&b, &format!("flip at {at}"))?;
            integrity += 1;
        }
    }
    // 3. Resealed structural mutations (the format validators must reject them).
    let index0 = fullobj::HEADER_BYTES;
    let structural_mutations: Vec<(&str, Mutation)> = vec![
        (
            "version",
            Box::new(|b: &mut [u8]| b[15] = fullobj::FORMAT_VERSION.wrapping_add(1)),
        ),
        ("unknown_semantics", Box::new(|b: &mut [u8]| b[21] = 0x7F)),
        (
            "segment_ceiling",
            Box::new(|b: &mut [u8]| b[38..46].copy_from_slice(&32_768u64.to_le_bytes())),
        ),
        (
            "segment_count",
            Box::new(|b: &mut [u8]| {
                let n = u32::from_le_bytes([b[34], b[35], b[36], b[37]]);
                b[34..38].copy_from_slice(&n.wrapping_add(1).to_le_bytes());
            }),
        ),
        (
            "content_id",
            Box::new(move |b: &mut [u8]| b[index0 + 18] ^= 0xFF),
        ),
        (
            "payload_offset_gap",
            Box::new(move |b: &mut [u8]| {
                let v = u64::from_le_bytes(b[index0 + 50..index0 + 58].try_into().unwrap());
                b[index0 + 50..index0 + 58].copy_from_slice(&v.wrapping_add(1).to_le_bytes());
            }),
        ),
        (
            "payload_length_zero",
            Box::new(move |b: &mut [u8]| {
                b[index0 + 58..index0 + 66].copy_from_slice(&0u64.to_le_bytes());
            }),
        ),
    ];
    for (label, mutate) in &structural_mutations {
        let mut b = valid.to_vec();
        mutate(&mut b);
        reseal(&mut b);
        must_reject(&b, &format!("resealed {label}"))?;
        structural += 1;
    }

    // 4. Allocation bombs and malformed headers (resealed so they reach the
    //    parser): an absurd frame count must be rejected before anything is
    //    allocated in proportion to it.
    let allocation_mutations: Vec<(&str, Mutation)> = vec![
        (
            "total_frames_max",
            Box::new(|b: &mut [u8]| b[26..34].copy_from_slice(&u64::MAX.to_le_bytes())),
        ),
        (
            "total_frames_zero",
            Box::new(|b: &mut [u8]| b[26..34].copy_from_slice(&0u64.to_le_bytes())),
        ),
        ("channels_zero", Box::new(|b: &mut [u8]| b[20] = 0)),
    ];
    for (label, mutate) in &allocation_mutations {
        let mut b = valid.to_vec();
        mutate(&mut b);
        reseal(&mut b);
        must_reject(&b, label)?;
        other += 1;
    }

    // 5. Garbage and empty inputs.
    for (label, candidate) in [
        ("empty", Vec::new()),
        ("short", vec![0u8; 4]),
        ("zebra", (0..=255u8).collect::<Vec<u8>>()),
    ] {
        must_reject(&candidate, label)?;
        other += 1;
    }

    Ok((integrity, structural, other))
}

fn static_projection(
    manifest_sha: &str,
    corpus_sha: &str,
    incompressible: &[serde_json::Value],
    population: [u64; 7],
    integrity: usize,
    structural: usize,
    other: usize,
) -> Vec<u8> {
    let mut out = Vec::new();
    for head in [PROTOCOL_SCHEMA, manifest_sha, corpus_sha] {
        out.extend_from_slice(head.as_bytes());
        out.push(0);
    }
    for n in population {
        out.extend_from_slice(&n.to_le_bytes());
    }
    for obj in incompressible {
        for key in ["id", "canonical_i32_sha256"] {
            out.extend_from_slice(obj[key].as_str().unwrap_or("").as_bytes());
            out.push(0);
        }
        for key in ["b0_bytes", "b1_bytes", "vole_complete_bytes"] {
            out.extend_from_slice(&obj[key].as_u64().unwrap_or(0).to_le_bytes());
        }
    }
    for n in [integrity, structural, other] {
        out.extend_from_slice(&(n as u64).to_le_bytes());
    }
    out
}

/// Run the court; writes an immutable receipt under `receipts/negative/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let fail = |why: &str| -> Result<Verdict> {
        let mut b = ReceiptBuilder::new("negative");
        b.result(Verdict::FailedCorrectness)
            .result_detail(format!("negative controls failed: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court negative: FAILED_CORRECTNESS ({why})");
        eprintln!("  receipt: {}", path.display());
        Ok(Verdict::FailedCorrectness)
    };

    let manifest = corpus::manifest()?;
    let report = corpus::verify_manifest(&manifest)?;
    if !report.ok() {
        return fail("the frozen flagship corpus does not verify");
    }
    let specs = corpus::specs::specs();
    let spec_by_id: BTreeMap<&str, &Spec> = specs.iter().map(|s| (s.id.as_str(), s)).collect();

    let sw = Stopwatch::start();

    // ---- 1. incompressible controls ----
    //
    // Two explicit populations, because a >8-channel object is outside FLAC's
    // format domain: `all` (B0 and VOLE for every control) and `b1_comparable`
    // (B0, B1 and VOLE for the objects FLAC can actually encode). A B1 ratio is
    // only ever formed inside the second.
    let mut incompressible: Vec<serde_json::Value> = Vec::new();
    let mut all_objects = 0u64;
    let mut all_b0 = 0u64;
    let mut all_vole = 0u64;
    let mut b1c_objects = 0u64;
    let mut b1c_b0 = 0u64;
    let mut b1c_b1 = 0u64;
    let mut b1c_vole = 0u64;
    for o in &manifest.objects {
        if !INCOMPRESSIBLE_CLASSES.contains(&o.entropy_class.as_str()) {
            continue;
        }
        let spec = spec_by_id
            .get(o.id.as_str())
            .ok_or_else(|| Error::internal(format!("{}: not in the frozen membership", o.id)))?;
        let samples = generate::generate(spec)?;
        let canonical = hex(&generate::canonical_sha256(&samples));
        if canonical != o.canonical_i32_sha256 {
            return fail(&format!("{}: regenerated content changed", o.id));
        }
        let b0 = samples.len() as u64 * 4;
        let b1 = if generate::b1_comparable(o.channels) {
            let a = b1_flac_artifact(&samples, o.channels, o.sample_rate_hz, B1_LEVEL_PRIMARY)
                .map_err(|e| Error::internal(format!("{}: B1 artifact: {e}", o.id)))?;
            // The reported bytes ARE the artifact's bytes; nothing is estimated.
            debug_assert_eq!(a.bytes.len() as u64, a.encoding.encoded_bytes);
            Some((a.encoding.encoded_bytes, hex(&a.sha256)))
        } else {
            None
        };
        let obj = fullobj::compile_full_object(
            &o.id,
            o.sample_rate_hz,
            o.channels,
            o.frames,
            match spec.semantics.period_frames() {
                Some(period_frames) => FullSemantics::Loop { period_frames },
                None => FullSemantics::OneShot,
            },
            &samples,
            budget(),
        )?;
        let vole = obj.complete_bytes();
        let comparable = generate::b1_comparable(o.channels);
        all_objects += 1;
        all_b0 += b0;
        all_vole += vole;
        if comparable {
            b1c_objects += 1;
            b1c_b0 += b0;
            b1c_b1 += b1.as_ref().map(|(b, _)| *b).unwrap_or(0);
            b1c_vole += vole;
        }
        incompressible.push(serde_json::json!({
            "id": o.id,
            "entropy_class": o.entropy_class,
            "source_structure_class": o.source_structure_class,
            "channels": o.channels,
            "frames": o.frames,
            "sample_rate_hz": o.sample_rate_hz,
            "canonical_i32_sha256": canonical,
            "b1_comparable": comparable,
            "b0_bytes": b0,
            "b1_bytes": b1.as_ref().map(|(b, _)| *b),
            "b1_artifact_sha256": b1.as_ref().map(|(_, s)| s.clone()),
            "vole_complete_bytes": vole,
            "b1_over_b0": b1.as_ref().map(|(b, _)| *b as f64 / b0 as f64),
            "vole_over_b0": vole as f64 / b0 as f64,
            "vole_over_b1": b1.as_ref().map(|(b, _)| vole as f64 / *b as f64),
        }));
    }
    if incompressible.is_empty() {
        return fail("the frozen corpus has no incompressible control objects");
    }

    // ---- 2. hostile archives ----
    let samples: Vec<i32> = (0..4096i32).map(|f| ((f % 64) << 14) - (1 << 12)).collect();
    let fixture = fullobj::compile_full_object(
        "negative-fixture",
        48_000,
        1,
        4096,
        FullSemantics::OneShot,
        &samples,
        budget(),
    )?;
    let (integrity, structural, other) = hostile_battery(&fixture.bytes)?;

    let total_ns = sw.elapsed_ns().max(0) as u64;
    let population = [
        all_objects,
        all_b0,
        all_vole,
        b1c_objects,
        b1c_b0,
        b1c_b1,
        b1c_vole,
    ];
    let result_hex = hex(&Sha256::digest(&static_projection(
        &report.manifest_sha256,
        &report.corpus_sha256,
        &incompressible,
        population,
        integrity,
        structural,
        other,
    )));
    if NEGATIVE_RESULT_SHA256.is_empty() {
        eprintln!("court negative: frozen result hash is unset; observed {result_hex}");
    } else if result_hex != NEGATIVE_RESULT_SHA256 {
        return fail(&format!(
            "static result hash changed: frozen {NEGATIVE_RESULT_SHA256}, observed {result_hex}"
        ));
    }

    let verdict = Verdict::Supported;
    let params = CourtParams {
        universe: Some(manifest.universe.clone()),
        profile: Some(format!("{} + negative.v1", manifest.profile)),
        backend: Some("incompressible corpus controls + hostile full-object containers".into()),
        sample_rate_hz: None,
        quantum_frames: None,
        content_kind: Some("negative controls (frozen before results)".into()),
        ..Default::default()
    };
    let mut builder = ReceiptBuilder::new("negative");
    builder
        .result(verdict)
        .result_detail(format!(
            "{} incompressible control objects (B0/VOLE population): VOLE {all_vole} B vs B0 {all_b0} B
             (vole/B0 {:.3}); B1-comparable population ({b1c_objects} objects): B0 {b1c_b0} B,
             B1 {b1c_b1} B, VOLE {b1c_vole} B (vole/B0 {:.3}, vole/B1 {:.3}); hostile archives:
             {integrity} integrity-hostile, {structural} resealed structural-hostile and {other}
             malformed/allocation-bomb candidates all rejected with a typed error, no panic; result
             sha256 {result_hex}",
            all_objects,
            all_vole as f64 / all_b0 as f64,
            b1c_vole as f64 / b1c_b0 as f64,
            if b1c_b1 > 0 {
                b1c_vole as f64 / b1c_b1 as f64
            } else {
                0.0
            },
        ))
        .params(params)
        .provenance(Provenance {
            reference_hash: Some(result_hex.clone()),
            corpus_hash: Some(report.corpus_sha256.clone()),
            ..Default::default()
        })
        .extra(
            "protocol",
            serde_json::json!({
                "schema": PROTOCOL_SCHEMA,
                "incompressible_classes": INCOMPRESSIBLE_CLASSES,
                "b1": "level 5, 32-bit, exact canonical i32, pinned libflac-rs",
                "vole": "selected full-object container (Seal 5 segmentation, exact segments)",
                "hostile": "truncation, bit flip, resealed structural mutation, allocation bomb, \
                            garbage; each must be rejected with a typed error and no panic",
                "claim_boundary": "on incompressible material both codecs are expected to lose to raw \
                                   framing once framing is included; this court prints that rather \
                                   than hiding it",
            }),
        )
        .extra(
            "incompressible",
            serde_json::json!({
                "all_population": {
                    "objects": all_objects,
                    "b0_bytes": all_b0,
                    "vole_complete_bytes": all_vole,
                    "vole_over_b0": all_vole as f64 / all_b0 as f64,
                },
                "b1_comparable_population": {
                    "objects": b1c_objects,
                    "b0_bytes": b1c_b0,
                    "b1_bytes": b1c_b1,
                    "vole_complete_bytes": b1c_vole,
                    "vole_over_b0": b1c_vole as f64 / b1c_b0 as f64,
                    "vole_over_b1": if b1c_b1 > 0 { Some(b1c_vole as f64 / b1c_b1 as f64) } else { None },
                },
                "excluded_from_b1": all_objects.saturating_sub(b1c_objects),
                "per_object": incompressible,
            }),
        )
        .extra(
            "hostile",
            serde_json::json!({
                "integrity_hostile_rejected": integrity,
                "structural_hostile_rejected": structural,
                "malformed_or_allocation_rejected": other,
                "total_rejected": integrity + structural + other,
                "panics": 0,
                "total_ns": total_ns,
            }),
        )
        .limitation(
            "this court is a negative control, not a scoreboard: it reports that incompressible \
             material is not compressed by either codec and that hostile archives are rejected. It \
             does not assert a predetermined winner",
        );
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court negative: {verdict}");
    println!(
        "  all: {all_objects} objects | B0 {all_b0} B | VOLE {all_vole} B (vole/B0 {:.3})",
        all_vole as f64 / all_b0 as f64
    );
    println!(
        "  B1-comparable: {b1c_objects} objects | B0 {b1c_b0} B | B1 {b1c_b1} B | VOLE {b1c_vole} B (vole/B1 {:.3})",
        if b1c_b1 > 0 {
            b1c_vole as f64 / b1c_b1 as f64
        } else {
            0.0
        }
    );
    println!(
        "  hostile rejected: {integrity} integrity + {structural} structural + {other} malformed/bomb"
    );
    println!("  result sha256: {result_hex}");
    println!("  receipt: {}", path.display());
    Ok(verdict)
}
