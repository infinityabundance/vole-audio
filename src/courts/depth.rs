//! `court depth` — minimum stable observation depth (Phase M, contract §48/§57).
//!
//! The runtime court measures one quantum (512 frames). The interesting
//! question for a real-time source is different: **how much buffered lookahead
//! does a backend need so that it never underruns?** As the quantum shrinks the
//! per-window deadline shrinks with it, and a backend whose tail latency exceeds
//! the deadline needs depth to absorb the deficit.
//!
//! This court runs the frozen sequential trace at several quanta over
//! B2/B3-warm/B4/B5 and, from the measured per-window latencies, computes the
//! minimum prefill depth at which no window underruns:
//!
//! ```text
//! completion[i] = latency_0 + ... + latency_i          (producer finish time)
//! consumer needs quantum i at  completion[k-1] + i*deadline   (k = prefill)
//! underrun  <=>  completion[i] - i*deadline > completion[k-1]
//! min depth  =  smallest k with completion[k-1] >= max_i(completion[i] - i*deadline)
//! no finite k satisfies it  =>  UNSTABLE at that quantum
//! ```
//!
//! Depth is a derived measured quantity, so it is deliberately **outside** the
//! frozen static result; the frozen projection covers the protocol, the request
//! geometry and the exactness of every window.

use crate::baseline::{B1_LEVEL_PRIMARY, FlacArtifact, b1_flac_artifact};
use crate::corpus;
use crate::corpus::generate::{self, Spec};
use crate::error::{Error, Result};
use crate::evidence::TailSummary;
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder};
use crate::fullobj::{self, FullSemantics, VerifiedFullObject};
use crate::hash::sha256::{Sha256, hex};
use crate::inverse::SearchBudget;
use crate::runtime::cache::CacheState;
use crate::runtime::{
    DiskPcmArtifact, DiskPcmSource, FlacPreloadSource, ResidentPcmSource, RuntimeSource,
    VoleBoundedSource, WindowPlan, deadline_ns, frozen_trace,
};
use crate::status::Verdict;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use super::measure::{Traversal, min_stable_depth_quanta, run_traversal, timer_overhead_ns};

/// Frozen static-result hash (empty means "not yet frozen").
pub const DEPTH_RESULT_SHA256: &str =
    "3e616cde6a63942dc127b16382305f0cff2b0fca598b45c85a4c75daea5abb65";

/// The frozen depth-sweep protocol identity.
pub const PROTOCOL_SCHEMA: &str = "vole.audio.depth.protocol.v1";

/// The frozen quantum ladder (frames per read).
pub const QUANTA: [u32; 5] = [64, 128, 256, 512, 1024];

const SOURCES: [&str; 4] = ["B2", "B3-warm", "B4", "B5"];

fn budget() -> SearchBudget {
    SearchBudget::default()
}

fn semantics_of(spec: &Spec) -> FullSemantics {
    match spec.semantics.period_frames() {
        Some(period_frames) => FullSemantics::Loop { period_frames },
        None => FullSemantics::OneShot,
    }
}

/// Outcome of the depth computation at one quantum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Depth {
    Stable { depth_quanta: u64 },
    Unstable,
}

impl Depth {
    fn label(&self) -> &'static str {
        match self {
            Depth::Stable { .. } => "STABLE",
            Depth::Unstable => "UNSTABLE",
        }
    }

    fn depth_quanta(&self) -> Option<u64> {
        match self {
            Depth::Stable { depth_quanta } => Some(*depth_quanta),
            Depth::Unstable => None,
        }
    }
}

/// Minimum prefill depth at which a producer with these per-window latencies
/// never underruns.
fn min_stable_depth(latencies: &[u64], deadline: u64) -> Depth {
    match min_stable_depth_quanta(latencies, deadline) {
        Some(depth_quanta) => Depth::Stable { depth_quanta },
        None => Depth::Unstable,
    }
}

fn static_projection(manifest_sha: &str, corpus_sha: &str, cells: &[serde_json::Value]) -> Vec<u8> {
    let mut out = Vec::new();
    for head in [PROTOCOL_SCHEMA, manifest_sha, corpus_sha] {
        out.extend_from_slice(head.as_bytes());
        out.push(0);
    }
    for q in QUANTA {
        out.extend_from_slice(&q.to_le_bytes());
    }
    for c in cells {
        out.extend_from_slice(c["id"].as_str().unwrap_or("").as_bytes());
        out.push(0);
        out.extend_from_slice(c["canonical_i32_sha256"].as_str().unwrap_or("").as_bytes());
        out.push(0);
        out.extend_from_slice(&c["channels"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&c["frames"].as_u64().unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&c["sample_rate_hz"].as_u64().unwrap_or(0).to_le_bytes());
        if let Some(sources) = c["sources"].as_object() {
            for (name, s) in sources {
                out.extend_from_slice(name.as_bytes());
                out.push(0);
                if let Some(quanta) = s["quanta"].as_object() {
                    for (q, v) in quanta {
                        out.extend_from_slice(q.as_bytes());
                        out.push(0);
                        out.push(v["exact"].as_bool().unwrap_or(false) as u8);
                        out.extend_from_slice(
                            &v["deadline_ns"].as_u64().unwrap_or(0).to_le_bytes(),
                        );
                        out.extend_from_slice(&v["windows"].as_u64().unwrap_or(0).to_le_bytes());
                    }
                }
                out.push(0);
            }
        }
        out.push(0xEE);
    }
    out
}

/// Run the court; writes an immutable receipt under `receipts/depth/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let fail = |why: &str| -> Result<Verdict> {
        let mut b = ReceiptBuilder::new("depth");
        b.result(Verdict::FailedCorrectness)
            .result_detail(format!("depth sweep failed: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court depth: FAILED_CORRECTNESS ({why})");
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

    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("depth-b3-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;

    let mut cells: Vec<serde_json::Value> = Vec::with_capacity(manifest.objects.len());
    let mut all_exact = true;
    // source -> quantum -> per-object depths
    let mut pooled: BTreeMap<&'static str, BTreeMap<u32, Vec<u64>>> = BTreeMap::new();
    let mut misses: BTreeMap<&'static str, BTreeMap<u32, u64>> = BTreeMap::new();

    for o in &manifest.objects {
        let spec = spec_by_id
            .get(o.id.as_str())
            .ok_or_else(|| Error::internal(format!("{}: not in the frozen membership", o.id)))?;
        let samples = generate::generate(spec)?;
        let canonical = hex(&generate::canonical_sha256(&samples));
        if canonical != o.canonical_i32_sha256 {
            return fail(&format!("{}: regenerated content changed", o.id));
        }
        let ch = usize::from(o.channels);

        let mut b2 = ResidentPcmSource::open("B2", o.channels, o.sample_rate_hz, samples.clone())?;
        let path = dir.join(format!("{}.pcm", o.id));
        let b3_art = DiskPcmArtifact::create(&path, o.channels, o.sample_rate_hz, &samples)?;
        let mut b3 = DiskPcmSource::open("B3-warm", &b3_art)?;
        let b4_artifact: Option<FlacArtifact> = if generate::b1_comparable(o.channels) {
            Some(
                b1_flac_artifact(&samples, o.channels, o.sample_rate_hz, B1_LEVEL_PRIMARY)
                    .map_err(|e| Error::internal(format!("{}: B1 artifact: {e}", o.id)))?,
            )
        } else {
            None
        };
        let mut b4 = match &b4_artifact {
            Some(a) => Some(FlacPreloadSource::open(
                "B4",
                o.channels,
                o.sample_rate_hz,
                a,
            )?),
            None => None,
        };
        let obj = fullobj::compile_full_object(
            &o.id,
            o.sample_rate_hz,
            o.channels,
            o.frames,
            semantics_of(spec),
            &samples,
            budget(),
        )?;
        let verified = Arc::new(VerifiedFullObject::verify(obj.bytes.clone())?);
        let mut b5 = VoleBoundedSource::open("B5", verified.clone(), false)?;

        let mut source_json = serde_json::Map::new();
        for name in SOURCES {
            let mut quanta_json = serde_json::Map::new();
            for q in QUANTA {
                if name == "B4" && b4.is_none() {
                    quanta_json.insert(
                        q.to_string(),
                        serde_json::json!({
                            "status": "NOT_APPLICABLE_BY_FORMAT_DOMAIN",
                            "exact": serde_json::Value::Null,
                        }),
                    );
                    continue;
                }
                let plan = WindowPlan::from_spans(
                    &samples,
                    ch,
                    frozen_trace(o.frames, q).map_err(|e| {
                        Error::internal(format!("{}: trace at {q} frames: {e}", o.id))
                    })?,
                )?;
                let max_frames = q as usize * ch;
                let mut dst = vec![0i32; max_frames];
                let mut scratch: Vec<u8> = Vec::new();
                let t: Traversal = match name {
                    "B2" => {
                        run_traversal(&mut b2, &plan, ch, o.sample_rate_hz, &mut dst, &mut scratch)?
                    }
                    "B3-warm" => {
                        let ev = b3.prime_and_verify()?;
                        if ev.state != CacheState::Warm {
                            return fail(&format!("{}: B3 warm state not verified", o.id));
                        }
                        run_traversal(&mut b3, &plan, ch, o.sample_rate_hz, &mut dst, &mut scratch)?
                    }
                    "B4" => {
                        let s = b4.as_mut().expect("B4 present");
                        run_traversal(s, &plan, ch, o.sample_rate_hz, &mut dst, &mut scratch)?
                    }
                    "B5" => {
                        b5.reset()?;
                        run_traversal(&mut b5, &plan, ch, o.sample_rate_hz, &mut dst, &mut scratch)?
                    }
                    other => return Err(Error::internal(format!("unknown source '{other}'"))),
                };
                all_exact &= t.exact;
                let deadline = deadline_ns(q, o.sample_rate_hz);
                let depth = min_stable_depth(&t.latencies, deadline);
                if let Some(d) = depth.depth_quanta() {
                    pooled
                        .entry(name)
                        .or_default()
                        .entry(q)
                        .or_default()
                        .push(d);
                }
                *misses.entry(name).or_default().entry(q).or_default() += t.deadline_misses;
                quanta_json.insert(
                    q.to_string(),
                    serde_json::json!({
                        "status": "MEASURED",
                        "deadline_ns": deadline,
                        "windows": t.windows,
                        "exact": t.exact,
                        "deadline_misses": t.deadline_misses,
                        "min_stable_depth_quanta": depth.depth_quanta(),
                        "stability": depth.label(),
                        "latency": TailSummary::summarize(&t.latencies),
                    }),
                );
            }
            source_json.insert(
                name.to_string(),
                serde_json::json!({
                    "artifact_storage_bytes": match name {
                        "B2" => 0,
                        "B3-warm" => b3.info().artifact_storage_bytes,
                        "B4" => b4.as_ref().map(|s| s.info().artifact_storage_bytes).unwrap_or(0),
                        _ => obj.complete_bytes(),
                    },
                    "quanta": serde_json::Value::Object(quanta_json),
                }),
            );
        }
        let _ = std::fs::remove_file(&path);

        cells.push(serde_json::json!({
            "id": o.id,
            "source_structure_class": o.source_structure_class,
            "entropy_class": o.entropy_class,
            "sample_rate_hz": o.sample_rate_hz,
            "channels": o.channels,
            "frames": o.frames,
            "canonical_i32_sha256": canonical,
            "sources": serde_json::Value::Object(source_json),
        }));
    }

    let _ = std::fs::remove_dir_all(&dir);
    if !all_exact {
        return fail("a depth-sweep window was not reproduced exactly");
    }

    let result_hex = hex(&Sha256::digest(&static_projection(
        &report.manifest_sha256,
        &report.corpus_sha256,
        &cells,
    )));
    if DEPTH_RESULT_SHA256.is_empty() {
        eprintln!("court depth: frozen result hash is unset; observed {result_hex}");
    } else if result_hex != DEPTH_RESULT_SHA256 {
        return fail(&format!(
            "static result hash changed: frozen {DEPTH_RESULT_SHA256}, observed {result_hex}"
        ));
    }

    // Corpus-level stable depth is the worst over objects (stability must hold
    // for every object independently).
    let mut surface = serde_json::Map::new();
    for name in SOURCES {
        let mut per_q = serde_json::Map::new();
        for q in QUANTA {
            let depths = pooled
                .get(name)
                .and_then(|m| m.get(&q))
                .cloned()
                .unwrap_or_default();
            let worst = depths.iter().copied().max();
            per_q.insert(
                q.to_string(),
                serde_json::json!({
                    "objects_with_finite_depth": depths.len(),
                    "worst_min_stable_depth_quanta": worst,
                    "median_min_stable_depth_quanta": median(&depths),
                    "deadline_misses": misses.get(name).and_then(|m| m.get(&q)).copied().unwrap_or(0),
                }),
            );
        }
        surface.insert(name.to_string(), serde_json::Value::Object(per_q));
    }
    let (timer_min_ns, timer_median_ns) = timer_overhead_ns();
    let object_count = cells.len();

    let verdict = Verdict::Supported;
    let params = CourtParams {
        universe: Some(manifest.universe.clone()),
        profile: Some(format!("{} + depth.v1", manifest.profile)),
        backend: Some("B2 / B3-warm / B4 / B5".into()),
        sample_rate_hz: None,
        quantum_frames: None,
        content_kind: Some("flagship corpus (frozen before results): depth sweep".into()),
        ..Default::default()
    };
    let mut builder = ReceiptBuilder::new("depth");
    builder
        .result(verdict)
        .result_detail(format!(
            "minimum stable observation depth over {} frozen objects at quanta {:?}; every window \
             reproduced exactly; result sha256 {result_hex}",
            cells.len(),
            QUANTA,
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
                "quanta_frames": QUANTA,
                "method": "completion[i] - i*deadline vs completion[k-1]; smallest stable prefill k",
                "stability_scope": "per object; the corpus-level depth is the worst over objects",
                "latency_boundary": "harness-owned",
                "outside_the_frozen_hash": "depth is derived from measured latencies and is evidence, \
                                            not a frozen constant",
            }),
        )
        .extra(
            "measurement",
            serde_json::json!({
                "timer_overhead_ns": { "min": timer_min_ns, "median": timer_median_ns },
                "surface": surface,
            }),
        )
        .extra("cells", serde_json::Value::Array(cells))
        .limitation(
            "depth is computed from this host's measured latencies under idle conditions. It is not a \
             guarantee for other machines or under load; `court interference` measures the load case",
        );
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court depth: {verdict}");
    println!(
        "  objects: {} | quanta: {QUANTA:?} | all exact: {all_exact}",
        object_count
    );
    println!("  result sha256: {result_hex}");
    println!("  receipt: {}", path.display());
    Ok(verdict)
}

fn median(values: &[u64]) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_unstable();
    Some(v[v.len() / 2])
}
