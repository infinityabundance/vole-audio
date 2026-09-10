//! `court fullobj` — the full-object archival container mechanism (Phase M,
//! Seal 5).
//!
//! This court freezes the **mechanism** before any flagship result exists. It
//! deliberately does *not* touch the flagship corpus; it exercises the container
//! on hostile, non-flagship fixtures whose only purpose is to pin the format and
//! prove exactness at the places a container breaks:
//!
//! * the segmentation boundaries — `1`, `65,535`, `65,536`, `65,537`,
//!   `131,072`, `131,073` frames (one below, at, and above the ceiling, and
//!   exact/plus-one multiples);
//! * mono, stereo and 3-channel geometry;
//! * silence, constant, exact-repeat, full-width noise and mixed content;
//! * observation at `boundary - 1`, `boundary`, `boundary + 1`, at a window
//!   spanning two segments, at the last frame, and past a loop root's finite
//!   extent (the declared loop region must repeat);
//! * hostile containers (truncated, corrupted, wrong version, wrong segmentation
//!   ceiling, empty payload) must fail typed, never decode.
//!
//! Selection is frozen: minimum Phase-K `complete_bytes`, deterministic ties.
//! The court asserts that every segment is exact, that the container's
//! `complete_bytes` is its *actual* serialized length, and that reconstruction is
//! sample-for-sample across the whole extent. All measured timing is excluded
//! from the frozen static hash.

use crate::error::{Error, Result};
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder};
use crate::evidence::timing::Stopwatch;
use crate::fullobj::{
    self, FullSemantics, MAX_SEGMENT_FRAMES, MaterializedFullObject, candidate_tag_name,
};
use crate::hash::sha256::Sha256;
use crate::inverse::{CandidateKind, SearchBudget};
use crate::status::Verdict;
use std::path::Path;

/// Frozen static-result hash: the court fails if a change silently alters the
/// container mechanism. Empty means "not yet frozen" (the observed value is
/// printed); re-freeze only with a documented reason.
pub const FULLOBJ_RESULT_SHA256: &str =
    "4b517ea0d564662d0b5c004434d1cea5b7358c5e1e563df5f6628a4665993a87";

/// Nominal rate recorded in the receipt (content classes are rate-independent).
pub const FULLOBJ_RATE_HZ: u32 = 48_000;

/// The frozen boundary lengths the mechanism must handle.
pub const BOUNDARY_LENGTHS: [u64; 6] = [1, 65_535, 65_536, 65_537, 131_072, 131_073];

/// The mechanism court's search budget. Small deliberately: this court pins
/// exactness and the container format, not search economics.
pub fn budget() -> SearchBudget {
    SearchBudget {
        max_period_scan: 128,
        max_residual_period_candidates: 2,
        max_candidates: 12,
        ..SearchBudget::default()
    }
}

/// Deterministic fixture stream (fixtures only; never the universe PRNG).
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// One mechanism fixture.
struct Fixture {
    name: String,
    channels: u8,
    frames: u64,
    semantics: FullSemantics,
    samples: Vec<i32>,
}

fn tone(frames: usize, channels: usize, period: usize) -> Vec<i32> {
    let ch = channels as i32;
    (0..frames)
        .flat_map(|f| {
            let s = (((f % period) as i32) << 14) - (1 << 12);
            (0..channels).map(move |c| s.wrapping_sub(c as i32 * ch * 37))
        })
        .collect()
}

fn noise(frames: usize, channels: usize, seed: u64) -> Vec<i32> {
    let mut s = seed;
    (0..frames * channels)
        .map(|_| splitmix(&mut s) as i32)
        .collect()
}

fn silence(frames: usize, channels: usize) -> Vec<i32> {
    vec![0i32; frames * channels]
}

fn constant(frames: usize, channels: usize, level: i32) -> Vec<i32> {
    vec![level; frames * channels]
}

/// The mechanism fixture set. Non-flagship by construction, deterministic, and
/// deliberately hostile at the segment boundaries.
fn fixtures() -> Vec<Fixture> {
    let mut out = Vec::new();
    // 1. Silence and constant across every boundary length (procedural).
    for &frames in &BOUNDARY_LENGTHS {
        let n = frames as usize;
        out.push(Fixture {
            name: format!("silence-{frames}f-1ch"),
            channels: 1,
            frames,
            semantics: FullSemantics::OneShot,
            samples: silence(n, 1),
        });
    }
    for &frames in &[65_536u64, 65_537, 131_073] {
        let n = frames as usize;
        out.push(Fixture {
            name: format!("constant-{frames}f-1ch"),
            channels: 1,
            frames,
            semantics: FullSemantics::OneShot,
            samples: constant(n, 1, -1_073_741_824),
        });
    }
    // 2. Exact-repeat tones across every boundary length.
    for &frames in &BOUNDARY_LENGTHS {
        let n = frames as usize;
        out.push(Fixture {
            name: format!("tone64-{frames}f-1ch"),
            channels: 1,
            frames,
            semantics: FullSemantics::OneShot,
            samples: tone(n, 1, 64),
        });
    }
    // 3. Full-width noise (literal) and mixed content.
    for &frames in &[65_537u64, 131_073] {
        let n = frames as usize;
        out.push(Fixture {
            name: format!("noise-{frames}f-1ch"),
            channels: 1,
            frames,
            semantics: FullSemantics::OneShot,
            samples: noise(n, 1, 0x0123_4567_89ab_cdef),
        });
        let mut mixed = noise(n / 2, 1, 0xdead_beef_1234_5678);
        mixed.extend(std::iter::repeat_n(0i32, n - n / 2));
        out.push(Fixture {
            name: format!("mixed-{frames}f-1ch"),
            channels: 1,
            frames,
            semantics: FullSemantics::OneShot,
            samples: mixed,
        });
    }
    // 4. Multi-channel geometry at and around the boundary.
    for &channels in &[2u8, 3] {
        for &frames in &[65_535u64, 65_536, 65_537, 131_073] {
            let n = frames as usize;
            let ch = usize::from(channels);
            out.push(Fixture {
                name: format!("tone64-{frames}f-{channels}ch"),
                channels,
                frames,
                semantics: FullSemantics::OneShot,
                samples: tone(n, ch, 64),
            });
            out.push(Fixture {
                name: format!("noise-{frames}f-{channels}ch"),
                channels,
                frames,
                semantics: FullSemantics::OneShot,
                samples: noise(n, ch, 0xfeed_0000 + u64::from(channels) * 31 + frames),
            });
        }
    }
    // 5. A loop root whose period does not divide the extent.
    let frames = 131_073u64;
    let period = 64usize;
    let base = tone(period, 1, period);
    let samples: Vec<i32> = (0..frames as usize).map(|f| base[f % period]).collect();
    out.push(Fixture {
        name: format!("loop64-{frames}f-1ch"),
        channels: 1,
        frames,
        semantics: FullSemantics::Loop {
            period_frames: period as u32,
        },
        samples,
    });
    out
}

/// Observation checkpoints: `boundary - 1`, `boundary`, `boundary + 1` for every
/// segment boundary, plus the last frame, plus a window spanning each boundary.
fn observation_starts(mat: &MaterializedFullObject) -> Vec<u64> {
    let total = mat.total_frames();
    let mut starts: Vec<u64> = vec![0, total - 1];
    for seg in &mat.decoded.segments {
        let b = seg.plan.start_frame;
        if b > 0 {
            starts.push(b - 1);
        }
        if b < total {
            starts.push(b);
        }
        if b + 1 < total {
            starts.push(b + 1);
        }
    }
    starts.sort_unstable();
    starts.dedup();
    starts
}

/// Verify every boundary observation and (for loops) the declared continuation.
fn boundary_observations(mat: &MaterializedFullObject, samples: &[i32]) -> Result<(usize, usize)> {
    let ch = mat.channels();
    let total = mat.total_frames();
    let mut checked = 0usize;
    let mut spanning = 0usize;
    for start in observation_starts(mat) {
        for frames in [1u64, 2, 3, MAX_SEGMENT_FRAMES] {
            if start + frames > total {
                continue;
            }
            if frames == MAX_SEGMENT_FRAMES && start != 0 {
                continue; // one full-extent window is enough
            }
            let got = mat.observe(start, frames)?;
            let lo = start as usize * ch;
            let hi = (start + frames) as usize * ch;
            if got != samples[lo..hi] {
                return Err(Error::internal(format!(
                    "full-object observation [{start}, +{frames}) is not exact"
                )));
            }
            checked += 1;
            if frames == 3 {
                spanning += 1;
            }
        }
    }
    if let FullSemantics::Loop { period_frames } = mat.decoded.semantics {
        // Past the finite extent the declared loop region repeats.
        let p = u64::from(period_frames);
        for frames in [1u64, 2, p] {
            let got = mat.observe(total, frames)?;
            if got != samples[..frames as usize * ch] {
                return Err(Error::internal(
                    "loop continuation does not repeat the declared loop region",
                ));
            }
        }
    }
    Ok((checked, spanning))
}

/// Hostile decode checks on one compiled container: every mutation must fail
/// typed rather than decode.
fn hostile_decode_checks(bytes: &[u8]) -> Result<usize> {
    let mut rejected = 0usize;
    let check = |candidate: Vec<u8>| -> Result<()> {
        if fullobj::decode_full_object(&candidate).is_err() {
            Ok(())
        } else {
            Err(Error::internal("a hostile container decoded"))
        }
    };
    if bytes.len() > 8 {
        check(bytes[..bytes.len() - 4].to_vec())?;
        rejected += 1;
    }
    let mut corrupted = bytes.to_vec();
    let at = fullobj::HEADER_BYTES.min(corrupted.len().saturating_sub(1));
    corrupted[at] ^= 0xFF;
    check(corrupted)?;
    rejected += 1;
    let mut version = bytes.to_vec();
    if version.len() > 16 {
        version[15] = fullobj::FORMAT_VERSION.wrapping_add(1);
        check(version)?;
        rejected += 1;
    }
    let mut ceiling = bytes.to_vec();
    if ceiling.len() >= 46 {
        ceiling[38..46].copy_from_slice(&32_768u64.to_le_bytes());
        check(ceiling)?;
        rejected += 1;
    }
    let mut empty = bytes.to_vec();
    let len_at = fullobj::HEADER_BYTES + 58;
    if empty.len() >= len_at + 8 {
        empty[len_at..len_at + 8].copy_from_slice(&0u64.to_le_bytes());
        check(empty)?;
        rejected += 1;
    }
    // The unmutated container must decode.
    fullobj::decode_full_object(bytes)?;
    Ok(rejected)
}

/// Deterministic static-result projection (no measured quantity).
fn static_projection(cells: &[serde_json::Value]) -> Vec<u8> {
    let mut out = Vec::new();
    for c in cells {
        out.extend_from_slice(c["name"].as_str().unwrap_or("").as_bytes());
        out.push(0);
        out.extend_from_slice(&c["channels"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&c["frames"].as_u64().unwrap_or(0).to_le_bytes());
        out.push(c["semantics_code"].as_u64().unwrap_or(0) as u8);
        out.extend_from_slice(&c["loop_period_frames"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&c["complete_bytes"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&c["objective_bytes"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&c["header_bytes"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&c["index_bytes"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&c["payload_bytes"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&c["integrity_bytes"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(
            &(c["exact_reconstruction"].as_bool().unwrap_or(false) as u8).to_le_bytes(),
        );
        out.extend_from_slice(
            &c["boundary_observations"]
                .as_u64()
                .unwrap_or(0)
                .to_le_bytes(),
        );
        if let Some(segs) = c["segments"].as_array() {
            for s in segs {
                out.extend_from_slice(&s["start_frame"].as_u64().unwrap_or(0).to_le_bytes());
                out.extend_from_slice(&s["frame_count"].as_u64().unwrap_or(0).to_le_bytes());
                out.push(s["representation_tag"].as_u64().unwrap_or(0) as u8);
                out.push(s["candidate_tag"].as_u64().unwrap_or(0) as u8);
                out.extend_from_slice(&s["objective_bytes"].as_u64().unwrap_or(0).to_le_bytes());
                out.extend_from_slice(&s["stored_bytes"].as_u64().unwrap_or(0).to_le_bytes());
                if let Some(cid) = s["content_id"].as_str() {
                    out.extend_from_slice(cid.as_bytes());
                }
                out.push(0);
            }
        }
        out.push(0xEE);
    }
    out
}

/// Run the court; writes an immutable receipt under `receipts/fullobj/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let fail = |why: &str| -> Result<Verdict> {
        let mut b = ReceiptBuilder::new("fullobj");
        b.result(Verdict::FailedCorrectness)
            .result_detail(format!("full-object container mechanism failed: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court fullobj: FAILED_CORRECTNESS ({why})");
        eprintln!("  receipt: {}", path.display());
        Ok(Verdict::FailedCorrectness)
    };

    let fixtures = fixtures();
    let sw = Stopwatch::start();
    let mut cells: Vec<serde_json::Value> = Vec::with_capacity(fixtures.len());
    let mut total_observations = 0usize;
    let mut hostile_rejections = 0usize;
    let mut procedural_objects = 0usize;
    let mut literal_objects = 0usize;
    let mut mixed_objects = 0usize;
    let mut selected_kinds: std::collections::BTreeMap<&'static str, usize> =
        std::collections::BTreeMap::new();

    for fx in &fixtures {
        let obj = fullobj::compile_full_object(
            &fx.name,
            FULLOBJ_RATE_HZ,
            fx.channels,
            fx.frames,
            fx.semantics,
            &fx.samples,
            budget(),
        )?;
        if obj.complete_bytes() != obj.bytes.len() as u64 {
            return fail("container complete_bytes is not its serialized length");
        }
        let mat = fullobj::materialize_full_object(&obj.bytes)?;
        if mat.samples() != &fx.samples[..] {
            return fail(&format!("{}: reconstruction is not exact", fx.name));
        }
        if mat.decoded.semantics != fx.semantics {
            return fail(&format!("{}: semantics were not preserved", fx.name));
        }
        if mat.decoded.channels != fx.channels || mat.decoded.total_frames != fx.frames {
            return fail(&format!("{}: root geometry was not preserved", fx.name));
        }
        let (checked, _) = boundary_observations(&mat, &fx.samples)?;
        total_observations += checked;
        hostile_rejections += hostile_decode_checks(&obj.bytes)?;

        if obj.all_literal() {
            literal_objects += 1;
        } else if obj.none_literal() {
            procedural_objects += 1;
        } else {
            mixed_objects += 1;
        }
        let mut seg_cells = Vec::with_capacity(obj.segments.len());
        for s in &obj.segments {
            *selected_kinds.entry(s.kind.name()).or_insert(0) += 1;
            seg_cells.push(serde_json::json!({
                "start_frame": s.plan.start_frame,
                "frame_count": s.plan.frame_count,
                "kind": s.kind.name(),
                "candidate_tag": s.kind.tag(),
                "representation": s.representation.name(),
                "representation_tag": s.representation.tag(),
                "content_id": s.content_id.to_string(),
                "objective_bytes": s.objective_bytes,
                "stored_bytes": s.stored_bytes,
                "proposed": s.proposed,
                "accepted": s.accepted,
            }));
        }

        cells.push(serde_json::json!({
            "name": fx.name,
            "channels": fx.channels,
            "frames": fx.frames,
            "semantics": if fx.semantics.is_loop() { "loop" } else { "one_shot" },
            "semantics_code": fx.semantics.kind_code(),
            "loop_period_frames": fx.semantics.period_frames(),
            "segment_count": obj.segment_count(),
            "segments": seg_cells,
            "header_bytes": obj.header_bytes,
            "index_bytes": obj.index_bytes,
            "payload_bytes": obj.payload_bytes,
            "integrity_bytes": obj.integrity_bytes,
            "complete_bytes": obj.complete_bytes(),
            "objective_bytes": obj.objective_bytes(),
            "literal_equivalent_bytes":
                crate::inverse::cost::canonical_u1_literal_bytes(fx.frames, fx.channels),
            "exact_reconstruction": true,
            "semantics_preserved": true,
            "boundary_observations": checked,
            "container_sha256": crate::hash::sha256::hex(&obj.sha256()),
            "compile_ns": obj.compile_ns,
        }));
    }

    let total_ns = sw.elapsed_ns().max(0) as u64;
    let object_count = cells.len();
    let result_hex = crate::hash::sha256::hex(&Sha256::digest(&static_projection(&cells)));
    if FULLOBJ_RESULT_SHA256.is_empty() {
        eprintln!("court fullobj: frozen result hash is unset; observed {result_hex}");
    } else if result_hex != FULLOBJ_RESULT_SHA256 {
        return fail(&format!(
            "static result hash changed: frozen {FULLOBJ_RESULT_SHA256}, observed {result_hex}"
        ));
    }

    let verdict = Verdict::Supported;
    let params = CourtParams {
        universe: Some("vole.audio.u1".into()),
        profile: Some("u1/v1 + fullobj.v1".into()),
        backend: Some("full-object container (exact U1 segments)".into()),
        sample_rate_hz: Some(FULLOBJ_RATE_HZ),
        quantum_frames: Some(crate::limits::DEFAULT_QUANTUM_FRAMES),
        content_kind: Some("hostile non-flagship mechanism fixtures".into()),
        ..Default::default()
    };
    let mut builder = ReceiptBuilder::new("fullobj");
    builder
        .result(verdict)
        .result_detail(format!(
            "full-object container over {} non-flagship fixtures; format v{}; frozen segment \
             ceiling {MAX_SEGMENT_FRAMES} frames (the Phase-K observation ceiling); every extent \
             reconstructed sample-for-sample; {total_observations} boundary observations exact; \
             {hostile_rejections} hostile containers rejected; result sha256 {result_hex}",
            object_count,
            fullobj::FORMAT_VERSION,
        ))
        .params(params)
        .provenance(Provenance {
            reference_hash: Some(result_hex.clone()),
            ..Default::default()
        })
        .extra(
            "container",
            serde_json::json!({
                "format_tag": String::from_utf8_lossy(fullobj::FORMAT_TAG),
                "format_version": fullobj::FORMAT_VERSION,
                "header_bytes": fullobj::HEADER_BYTES,
                "index_record_bytes": fullobj::INDEX_RECORD_BYTES,
                "integrity_bytes": fullobj::INTEGRITY_BYTES,
                "max_segment_frames": MAX_SEGMENT_FRAMES,
                "segmentation_rule": "start = 0; while remaining: len = min(65,536, remaining); \
                                      consecutive intrinsic ranges; frozen before any result",
                "segmentation_origin": "inherited verbatim from the frozen Phase-K observation \
                                        ceiling (MAX_INVERSE_FRAMES), not tuned",
                "selection_rule": "accepted exact candidates -> minimum Phase-K complete_bytes -> \
                                   deterministic proposal-order tie break; no weighted score",
                "reference_library": "empty: every object is priced standalone (no corpus-level \
                                      deduplication in a full-object comparison)",
                "not_a_representation": "the container is an object-above-objects archive, not a \
                                         new U1 Representation tag; Phase K is unchanged",
            }),
        )
        .extra(
            "boundary_lengths_frames",
            serde_json::json!(BOUNDARY_LENGTHS),
        )
        .extra(
            "aggregate",
            serde_json::json!({
                "objects": cells.len(),
                "boundary_observations": total_observations,
                "hostile_containers_rejected": hostile_rejections,
                "objects_all_literal": literal_objects,
                "objects_all_procedural": procedural_objects,
                "objects_mixed": mixed_objects,
                "selected_candidate_kinds": selected_kinds,
                "total_ns": total_ns,
            }),
        )
        .extra("cells", serde_json::Value::Array(cells))
        .limitation(
            "this court pins the container mechanism and its exactness; it makes no flagship \
             performance claim and does not touch the frozen flagship corpus",
        )
        .limitation(
            "the container stores real serialized bytes: complete_bytes = header + segment index \
             + Σ actual payload bytes + integrity digest. Per-segment objective_bytes is the \
             Phase-K selection objective (the candidate's complete_bytes); both are recorded and \
             the two are allowed to differ only by container framing, never hidden",
        )
        .limitation(
            "B1-vs-VOLE comparison and the flagship population measurement are a later increment \
             (Seal 6); this receipt is the mechanism gate",
        );
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court fullobj: {verdict}");
    println!(
        "  fixtures: {} | boundary observations: {total_observations} | hostile rejected: {hostile_rejections}",
        object_count
    );
    println!(
        "  objects all-literal: {literal_objects} | all-procedural: {procedural_objects} | mixed: {mixed_objects}"
    );
    println!("  selected kinds: {selected_kinds:?}");
    println!("  result sha256: {result_hex}");
    println!("  receipt: {}", path.display());
    // Referenced so the helper name is exercised in this build even when the
    // candidate tag is never surfaced elsewhere.
    debug_assert_eq!(candidate_tag_name(CandidateKind::Literal.tag()), "literal");
    Ok(verdict)
}
