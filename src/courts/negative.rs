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
    "8e6e1a34cecc880953b54d967d30380d1082c0caed48b1b77203e041aca0266c";

/// The frozen negative-control protocol identity.
pub const PROTOCOL_SCHEMA: &str = "vole.audio.negative.protocol.v1";

/// Entropy classes with random character (the hostile *and* structured-random
/// strata both live here; [`generate::is_hostile_incompressible`] separates them).
const RANDOM_CHARACTER_CLASSES: [&str; 2] = ["full_width_random", "scrambled"];

/// Accumulated bytes for one population.
#[derive(Debug, Clone, Copy, Default)]
struct Pop {
    objects: u64,
    b0: u64,
    b1: u64,
    vole: u64,
}

impl Pop {
    fn add(&mut self, b0: u64, b1: Option<u64>, vole: u64) {
        self.objects += 1;
        self.b0 += b0;
        self.b1 += b1.unwrap_or(0);
        self.vole += vole;
    }
}

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

/// Everything the negative court's static projection binds.
struct Projection<'a> {
    manifest_sha: &'a str,
    corpus_sha: &'a str,
    hostile: &'a [serde_json::Value],
    structured: &'a [serde_json::Value],
    /// `[hostile: objects, b0, b1, vole, b1_domain: objects, b0, b1, vole]`.
    population: [u64; 8],
    integrity: usize,
    structural: usize,
    other: usize,
}

fn static_projection(p: Projection<'_>) -> Vec<u8> {
    let mut out = Vec::new();
    for head in [PROTOCOL_SCHEMA, p.manifest_sha, p.corpus_sha] {
        out.extend_from_slice(head.as_bytes());
        out.push(0);
    }
    for n in p.population {
        out.extend_from_slice(&n.to_le_bytes());
    }
    // Hostile controls first (the claim under test), structured-random controls
    // second (reported, but explicitly not incompressible).
    for (tag, group) in [(1u8, p.hostile), (2u8, p.structured)] {
        out.push(tag);
        for obj in group {
            for key in ["id", "canonical_i32_sha256"] {
                out.extend_from_slice(obj[key].as_str().unwrap_or("").as_bytes());
                out.push(0);
            }
            for key in ["b0_bytes", "b1_bytes", "vole_complete_bytes"] {
                out.extend_from_slice(&obj[key].as_u64().unwrap_or(0).to_le_bytes());
            }
        }
    }
    for n in [p.integrity, p.structural, p.other] {
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

    // ---- 1. hostile incompressible controls ----
    //
    // A control is *incompressible* only if it is full-width, random in
    // character **and** genuinely independent across channels — the same
    // predicate the corpus membership tests use (`is_hostile_incompressible`),
    // so the two cannot drift. Objects that are temporally random but carry
    // exploitable structure (anticorrelated stereo, reduced amplitude occupancy)
    // are reported separately as *structured-random controls*.
    //
    // Two explicit populations, because a >8-channel object is outside FLAC's
    // format domain: `all` (B0 and VOLE for every hostile control) and
    // `b1_comparable` (B0, B1 and VOLE for those FLAC can encode). A B1 ratio is
    // formed only inside the second.
    let mut hostile: Vec<serde_json::Value> = Vec::new();
    let mut structured: Vec<serde_json::Value> = Vec::new();
    let mut h_all = Pop::default();
    let mut h_b1 = Pop::default();
    for o in &manifest.objects {
        let spec = spec_by_id
            .get(o.id.as_str())
            .ok_or_else(|| Error::internal(format!("{}: not in the frozen membership", o.id)))?;
        let is_random_character = RANDOM_CHARACTER_CLASSES.contains(&o.entropy_class.as_str());
        if !is_random_character {
            continue;
        }
        // The string class filter and the typed predicate must agree.
        debug_assert_eq!(
            is_random_character,
            generate::is_hostile_incompressible(spec)
                || generate::is_structured_random_control(spec),
            "{}: manifest entropy class disagrees with the typed predicate",
            o.id
        );
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
        let b1_bytes = b1.as_ref().map(|(b, _)| *b);
        let entry = serde_json::json!({
            "id": o.id,
            "entropy_class": o.entropy_class,
            "amplitude_class": o.amplitude_class,
            "channel_structure": o.channel_structure,
            "source_structure_class": o.source_structure_class,
            "channels": o.channels,
            "frames": o.frames,
            "sample_rate_hz": o.sample_rate_hz,
            "canonical_i32_sha256": canonical,
            "b1_comparable": comparable,
            "b0_bytes": b0,
            "b1_bytes": b1_bytes,
            "b1_artifact_sha256": b1.as_ref().map(|(_, s)| s.clone()),
            "vole_complete_bytes": vole,
            "b1_over_b0": b1_bytes.map(|b| b as f64 / b0 as f64),
            "vole_over_b0": vole as f64 / b0 as f64,
            "vole_over_b1": b1_bytes.map(|b| vole as f64 / b as f64),
        });
        if generate::is_hostile_incompressible(spec) {
            h_all.add(b0, b1_bytes, vole);
            if comparable {
                h_b1.add(b0, b1_bytes, vole);
            }
            hostile.push(entry);
        } else {
            structured.push(entry);
        }
    }
    if hostile.is_empty() {
        return fail("the frozen corpus has no hostile incompressible controls");
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
        h_all.objects,
        h_all.b0,
        h_all.b1,
        h_all.vole,
        h_b1.objects,
        h_b1.b0,
        h_b1.b1,
        h_b1.vole,
    ];
    let result_hex = hex(&Sha256::digest(&static_projection(Projection {
        manifest_sha: &report.manifest_sha256,
        corpus_sha: &report.corpus_sha256,
        hostile: &hostile,
        structured: &structured,
        population,
        integrity,
        structural,
        other,
    })));
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
            "{} hostile incompressible controls (B0/VOLE population): VOLE {} B vs B0 {} B (VOLE/B0
             {:.5}); B1-comparable hostile population ({} objects): B0 {} B, B1 {} B, VOLE {} B (B1/B0
             {:.5}, VOLE/B1 {:.5}); {} structured-random controls reported separately; hostile archives:
             {integrity} integrity-hostile, {structural} resealed structural-hostile and {other}
             malformed/allocation-bomb candidates all rejected with a typed error, no panic; result
             sha256 {result_hex}",
            h_all.objects,
            h_all.vole,
            h_all.b0,
            h_all.vole as f64 / h_all.b0 as f64,
            h_b1.objects,
            h_b1.b0,
            h_b1.b1,
            h_b1.vole,
            h_b1.b1 as f64 / h_b1.b0 as f64,
            h_b1.vole as f64 / h_b1.b1 as f64,
            structured.len(),
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
                "hostile_predicate": "is_hostile_incompressible: entropy in {full_width_random, scrambled}
                                      AND amplitude full AND channel structure in {mono,
                                      independent_stereo, multichannel}. Shared with the corpus
                                      membership tests",
                "structured_random": "temporally random but compressible (cross-channel relation or
                                      reduced amplitude occupancy); reported separately and never
                                      counted as incompressible",
                "b1": "level 5, 32-bit, exact canonical i32, pinned libflac-rs",
                "vole": "selected full-object container (Seal 5 segmentation, exact segments)",
                "hostile_archives": "truncation, bit flip, resealed structural mutation, allocation bomb,
                                     garbage; each must be rejected with a typed error and no panic",
                "claim_boundary": "on genuinely incompressible material both codecs are expected to land
                                   near raw plus their framing overhead; this court prints that rather
                                   than hiding it",
            }),
        )
        .extra(
            "hostile_populations",
            serde_json::json!({
                "all_population": {
                    "objects": h_all.objects,
                    "b0_bytes": h_all.b0,
                    "vole_complete_bytes": h_all.vole,
                    "vole_over_b0": h_all.vole as f64 / h_all.b0 as f64,
                },
                "b1_comparable_population": {
                    "objects": h_b1.objects,
                    "b0_bytes": h_b1.b0,
                    "b1_bytes": h_b1.b1,
                    "vole_complete_bytes": h_b1.vole,
                    "b1_over_b0": h_b1.b1 as f64 / h_b1.b0 as f64,
                    "vole_over_b0": h_b1.vole as f64 / h_b1.b0 as f64,
                    "vole_over_b1": h_b1.vole as f64 / h_b1.b1 as f64,
                },
                "excluded_from_b1": h_all.objects.saturating_sub(h_b1.objects),
                "hostile_per_object": hostile,
            }),
        )
        .extra(
            "structured_random_controls",
            serde_json::json!({
                "objects": structured.len(),
                "note": "temporal randomness is not incompressibility: these controls compress because of
                         cross-channel structure or limited amplitude occupancy, so they are excluded
                         from the hostile aggregate and shown here",
                "per_object": structured,
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
            "this court is a negative control, not a scoreboard: it reports that genuinely incompressible
             material is not compressed by either codec and that hostile archives are rejected.
             Structured-random controls are reported separately because temporal randomness alone is
             not incompressibility. No predetermined winner is asserted",
        );
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court negative: {verdict}");
    println!(
        "  hostile all: {} objects | B0 {} B | VOLE {} B (VOLE/B0 {:.5})",
        h_all.objects,
        h_all.b0,
        h_all.vole,
        h_all.vole as f64 / h_all.b0 as f64
    );
    println!(
        "  hostile B1-domain: {} objects | B0 {} B | B1 {} B | VOLE {} B (B1/B0 {:.5}, VOLE/B1 {:.5})",
        h_b1.objects,
        h_b1.b0,
        h_b1.b1,
        h_b1.vole,
        h_b1.b1 as f64 / h_b1.b0 as f64,
        h_b1.vole as f64 / h_b1.b1 as f64
    );
    println!(
        "  structured-random controls reported separately: {}",
        structured.len()
    );
    println!(
        "  hostile rejected: {integrity} integrity + {structural} structural + {other} malformed/bomb"
    );
    println!("  result sha256: {result_hex}");
    println!("  receipt: {}", path.display());
    Ok(verdict)
}
