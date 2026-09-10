//! `court interference` — adversarial real-time load (Phase M, contract §49).
//!
//! The runtime and depth courts measure an idle host. This court measures the
//! same frozen workload under controllable contention and reports the difference
//! honestly:
//!
//! ```text
//! idle              no added load (baseline)
//! cpu_burn          one CPU-bound spinner per logical CPU
//! memory_bandwidth  streaming access over a large resident buffer
//! storage_io        write + fsync + read cycles on a block-backed file
//! ```
//!
//! Conditions this host cannot actually manipulate — compositor/display load,
//! competing GPU compute, GPU context contention, DVFS, thermal steady state and
//! PCIe power saving — are reported `NOT_CONTROLLED`, never claimed. Energy is
//! Energy is measured only through real instruments: cumulative joules from a
//! powercap `energy_uj` counter when one exists, otherwise `NOT_AVAILABLE` (a
//! hwmon instantaneous power reading, if present, is recorded as a spot reading
//! and is not treated as workload energy).
//!
//! A bounded **CPU-contention soak** follows the matrix: it repeats the workload
//! under `cpu_burn` for a fixed wall-clock budget and reports tail drift and
//! accumulated deadline misses (the contract's "long run"/xrun question). It is
//! deliberately *not* run under memory bandwidth, which this court's own matrix
//! shows to be the more disruptive load; the soak workload is not chosen after
//! seeing the result.

use crate::baseline::{B1_LEVEL_PRIMARY, FlacArtifact, b1_flac_artifact};
use crate::corpus;
use crate::corpus::generate::{self, Spec};
use crate::error::{Error, Result};
use crate::evidence::TailSummary;
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder};
use crate::evidence::timing::Stopwatch;
use crate::fullobj::{self, FullSemantics, VerifiedFullObject};
use crate::hash::sha256::{Sha256, hex};
use crate::inverse::SearchBudget;
use crate::runtime::cache::CacheState;
use crate::runtime::load::{LoadGuard, LoadKind};
use crate::runtime::{
    DiskPcmArtifact, DiskPcmSource, FlacPreloadSource, QUANTUM_FRAMES, ResidentPcmSource,
    VoleBoundedSource, WindowPlan, deadline_ns, probe_energy_counter, probe_power,
};
use crate::status::Verdict;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use super::measure::{Traversal, min_stable_depth_quanta, run_traversal, timer_overhead_ns};

/// Frozen static-result hash (empty means "not yet frozen").
pub const INTERFERENCE_RESULT_SHA256: &str =
    "f23c70c15016814ce3efcb7454712dbcc9517226b8e187122d39156c38fd5088";

/// The frozen interference protocol identity.
pub const PROTOCOL_SCHEMA: &str = "vole.audio.interference.protocol.v1";

/// Controllable conditions, in measurement order (idle first).
pub const CONDITIONS: [Option<LoadKind>; 4] = [
    None,
    Some(LoadKind::CpuBurn),
    Some(LoadKind::MemoryBandwidth),
    Some(LoadKind::StorageIo),
];

/// Conditions the contract names that this host/architecture cannot control.
pub const NOT_CONTROLLED: [&str; 7] = [
    "compositor_display_load",
    "graphics_load",
    "competing_gpu_compute",
    "gpu_context_contention",
    "dvfs",
    "thermal_steady_state",
    "pcie_power_saving",
];

/// Bounded soak budget (seconds).
pub const SOAK_SECS: u64 = 10;

/// How long to let a load settle before measuring under it.
const SETTLE_MS: u64 = 250;

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

fn condition_name(c: Option<LoadKind>) -> &'static str {
    match c {
        None => "idle",
        Some(k) => k.name(),
    }
}

/// Prepared per-object authoring artifacts, built once and reused across
/// conditions so the conditions differ only in the load.
struct Prepared {
    id: String,
    channels: u8,
    sample_rate_hz: u32,
    frames: u64,
    canonical: String,
    samples: Vec<i32>,
    b3: DiskPcmArtifact,
    b4: Option<FlacArtifact>,
    b5: Arc<VerifiedFullObject>,
}

/// One object's deterministic traversal geometry for one (source, condition).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ObjectCond {
    exact: bool,
    windows: usize,
    deadline_ns: u64,
}

/// Accumulated per-condition, per-source evidence.
#[derive(Clone, Default)]
struct Acc {
    started: bool,
    exact: bool,
    windows: usize,
    latencies: Vec<u64>,
    deadline_misses: u64,
    worst_depth: Option<u64>,
    unstable_objects: u64,
    physical_storage_bytes_read: u64,
}

impl Acc {
    fn finish(&self) -> serde_json::Value {
        serde_json::json!({
            "exact": self.exact,
            "windows_per_traversal": self.windows,
            "latency": TailSummary::summarize(&self.latencies),
            "deadline_misses": self.deadline_misses,
            "worst_min_stable_depth_quanta": self.worst_depth,
            "unstable_objects": self.unstable_objects,
            "physical_storage_bytes_read": self.physical_storage_bytes_read,
        })
    }
}

fn static_projection(manifest_sha: &str, corpus_sha: &str, cells: &[serde_json::Value]) -> Vec<u8> {
    let mut out = Vec::new();
    for head in [PROTOCOL_SCHEMA, manifest_sha, corpus_sha] {
        out.extend_from_slice(head.as_bytes());
        out.push(0);
    }
    out.extend_from_slice(&QUANTUM_FRAMES.to_le_bytes());
    for c in CONDITIONS {
        out.extend_from_slice(condition_name(c).as_bytes());
        out.push(0);
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
                if let Some(conditions) = s["conditions"].as_object() {
                    for (cond, v) in conditions {
                        out.extend_from_slice(cond.as_bytes());
                        out.push(0);
                        out.push(v["exact"].as_bool().unwrap_or(false) as u8);
                        out.extend_from_slice(&v["windows"].as_u64().unwrap_or(0).to_le_bytes());
                        out.extend_from_slice(
                            &v["deadline_ns"].as_u64().unwrap_or(0).to_le_bytes(),
                        );
                    }
                }
                out.push(0);
            }
        }
        out.push(0xEE);
    }
    out
}

/// Run the court; writes an immutable receipt under `receipts/interference/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let fail = |why: &str| -> Result<Verdict> {
        let mut b = ReceiptBuilder::new("interference");
        b.result(Verdict::FailedCorrectness)
            .result_detail(format!("interference court failed: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court interference: FAILED_CORRECTNESS ({why})");
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
        .join(format!("interference-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let pcm_dir = dir.join("pcm");
    let load_dir = dir.join("load");
    std::fs::create_dir_all(&pcm_dir)?;
    std::fs::create_dir_all(&load_dir)?;

    // ---- authoring: build every artifact once ----
    let mut prepared: Vec<Prepared> = Vec::with_capacity(manifest.objects.len());
    for o in &manifest.objects {
        let spec = spec_by_id
            .get(o.id.as_str())
            .ok_or_else(|| Error::internal(format!("{}: not in the frozen membership", o.id)))?;
        let samples = generate::generate(spec)?;
        let canonical = hex(&generate::canonical_sha256(&samples));
        if canonical != o.canonical_i32_sha256 {
            return fail(&format!("{}: regenerated content changed", o.id));
        }
        let path = pcm_dir.join(format!("{}.pcm", o.id));
        let b3 = DiskPcmArtifact::create(&path, o.channels, o.sample_rate_hz, &samples)?;
        let b4 = if generate::b1_comparable(o.channels) {
            Some(
                b1_flac_artifact(&samples, o.channels, o.sample_rate_hz, B1_LEVEL_PRIMARY)
                    .map_err(|e| Error::internal(format!("{}: B1 artifact: {e}", o.id)))?,
            )
        } else {
            None
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
        let b5 = Arc::new(VerifiedFullObject::verify(obj.bytes.clone())?);
        prepared.push(Prepared {
            id: o.id.clone(),
            channels: o.channels,
            sample_rate_hz: o.sample_rate_hz,
            frames: o.frames,
            canonical,
            samples,
            b3,
            b4,
            b5,
        });
    }

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let power = probe_power();
    let counter = probe_energy_counter();
    let power_source_probe = match &power {
        Some(p) => serde_json::json!({ "status": "AVAILABLE", "source": p.describe() }),
        None => serde_json::json!({
            "status": "NOT_AVAILABLE",
            "detail": "no supported hwmon power source found",
        }),
    };
    let energy_counter_probe = match &counter {
        Some(c) => serde_json::json!({
            "status": "AVAILABLE",
            "source": c.describe(),
            "method": "cumulative energy_uj read at each condition boundary",
        }),
        None => serde_json::json!({
            "status": "NOT_AVAILABLE",
            "detail": "no readable powercap energy_uj counter found (one may exist but be \
                       root-only); no energy figure is invented",
        }),
    };

    let sw = Stopwatch::start();
    let mut all_exact = true;
    // condition -> source -> accumulated
    let mut matrix: BTreeMap<&'static str, BTreeMap<&'static str, Acc>> = BTreeMap::new();
    // True per-object deterministic records: (object -> source -> condition).
    // The pooled `matrix` above is measured evidence; the frozen cells are built
    // from *these*, so each object's cell binds its own traversal geometry.
    let mut per_object: BTreeMap<
        String,
        BTreeMap<&'static str, BTreeMap<&'static str, ObjectCond>>,
    > = BTreeMap::new();
    let mut energy_samples: Vec<serde_json::Value> = Vec::new();

    for condition in CONDITIONS {
        let cname = condition_name(condition);
        let guard = match condition {
            Some(kind) => Some(LoadGuard::start(kind, threads, &load_dir)?),
            None => None,
        };
        if guard.is_some() {
            std::thread::sleep(std::time::Duration::from_millis(SETTLE_MS));
        }
        let watts_before = power.as_ref().and_then(|p| p.read_watts());
        let energy_start_uj = counter.as_ref().and_then(|c| c.read_uj());
        let entry = matrix.entry(cname).or_default();
        for p in &prepared {
            let ch = usize::from(p.channels);
            let plan = WindowPlan::build(&p.samples, ch)?;
            let deadline = deadline_ns(QUANTUM_FRAMES, p.sample_rate_hz);
            let mut dst = vec![0i32; QUANTUM_FRAMES as usize * ch];
            let mut scratch: Vec<u8> = Vec::new();

            for name in SOURCES {
                let acc = entry.entry(name).or_default();
                let t: Traversal = match name {
                    "B2" => {
                        let mut s = ResidentPcmSource::open(
                            "B2",
                            p.channels,
                            p.sample_rate_hz,
                            p.samples.clone(),
                        )?;
                        run_traversal(&mut s, &plan, ch, p.sample_rate_hz, &mut dst, &mut scratch)?
                    }
                    "B3-warm" => {
                        let mut s = DiskPcmSource::open("B3-warm", &p.b3)?;
                        let ev = s.prime_and_verify()?;
                        if ev.state != CacheState::Warm {
                            return fail(&format!("{}: B3 warm state not verified", p.id));
                        }
                        run_traversal(&mut s, &plan, ch, p.sample_rate_hz, &mut dst, &mut scratch)?
                    }
                    "B4" => match &p.b4 {
                        Some(a) => {
                            let mut s =
                                FlacPreloadSource::open("B4", p.channels, p.sample_rate_hz, a)?;
                            run_traversal(
                                &mut s,
                                &plan,
                                ch,
                                p.sample_rate_hz,
                                &mut dst,
                                &mut scratch,
                            )?
                        }
                        None => continue,
                    },
                    "B5" => {
                        let mut s = VoleBoundedSource::open("B5", p.b5.clone(), false)?;
                        run_traversal(&mut s, &plan, ch, p.sample_rate_hz, &mut dst, &mut scratch)?
                    }
                    other => return Err(Error::internal(format!("unknown source '{other}'"))),
                };
                all_exact &= t.exact;
                per_object
                    .entry(p.id.clone())
                    .or_default()
                    .entry(name)
                    .or_default()
                    .insert(
                        cname,
                        ObjectCond {
                            exact: t.exact,
                            windows: t.windows,
                            deadline_ns: deadline,
                        },
                    );
                if !acc.started {
                    acc.started = true;
                    acc.windows = t.windows;
                    acc.exact = true;
                }
                acc.exact &= t.exact;
                acc.deadline_misses += t.deadline_misses;
                acc.physical_storage_bytes_read += t.physical_storage_bytes_read;
                match min_stable_depth_quanta(&t.latencies, deadline) {
                    Some(d) => {
                        acc.worst_depth = Some(acc.worst_depth.map_or(d, |w| w.max(d)));
                    }
                    None => acc.unstable_objects += 1,
                }
                acc.latencies.extend_from_slice(&t.latencies);
            }
        }
        let watts_after = power.as_ref().and_then(|p| p.read_watts());
        let energy_end_uj = counter.as_ref().and_then(|c| c.read_uj());
        if let (Some(c), Some(s), Some(e)) = (counter.as_ref(), energy_start_uj, energy_end_uj) {
            // A determinable interval is evidence; an undeterminable one (wrapped
            // counter with no declared range) is reported, never zeroed.
            match c.joules_between(s, e) {
                Some(joules) => energy_samples.push(serde_json::json!({
                    "condition": cname,
                    "joules": joules,
                    "source": c.id(),
                })),
                None => energy_samples.push(serde_json::json!({
                    "condition": cname,
                    "joules": serde_json::Value::Null,
                    "source": c.id(),
                    "detail": "counter wrapped with no declared range; interval unavailable",
                })),
            }
        } else if let (Some(a), Some(b)) = (watts_before, watts_after) {
            energy_samples.push(serde_json::json!({
                "condition": cname,
                "instantaneous_power_watts_before": a,
                "instantaneous_power_watts_after": b,
                "note": "spot power readings only; not a workload-energy measurement",
            }));
        }
        drop(guard);
    }

    // ---- bounded CPU-contention soak ----
    let soak = soak(&prepared, threads, &load_dir)?;
    all_exact &= soak.exact;

    if !all_exact {
        return fail("an interference window was not reproduced exactly");
    }

    // cells: per object, per source, per condition (exact + geometry only)
    let mut cells: Vec<serde_json::Value> = Vec::new();
    for (o, p) in manifest.objects.iter().zip(&prepared) {
        let mut source_json = serde_json::Map::new();
        for name in SOURCES {
            let mut cond_json = serde_json::Map::new();
            for condition in CONDITIONS {
                let cname = condition_name(condition);
                let hit = per_object
                    .get(&p.id)
                    .and_then(|m| m.get(name))
                    .and_then(|m| m.get(cname));
                cond_json.insert(
                    cname.to_string(),
                    match hit {
                        Some(r) => serde_json::json!({
                            "status": "MEASURED",
                            "exact": r.exact,
                            "windows": r.windows,
                            "deadline_ns": r.deadline_ns,
                        }),
                        None => serde_json::json!({
                            "status": "NOT_APPLICABLE_BY_FORMAT_DOMAIN",
                            "exact": serde_json::Value::Null,
                        }),
                    },
                );
            }
            source_json.insert(
                name.to_string(),
                serde_json::json!({ "conditions": serde_json::Value::Object(cond_json) }),
            );
        }
        cells.push(serde_json::json!({
            "id": o.id,
            "sample_rate_hz": o.sample_rate_hz,
            "channels": o.channels,
            "frames": p.frames,
            "canonical_i32_sha256": p.canonical,
            "sources": serde_json::Value::Object(source_json),
        }));
    }

    let _ = std::fs::remove_dir_all(&dir);

    let result_hex = hex(&Sha256::digest(&static_projection(
        &report.manifest_sha256,
        &report.corpus_sha256,
        &cells,
    )));
    if INTERFERENCE_RESULT_SHA256.is_empty() {
        eprintln!("court interference: frozen result hash is unset; observed {result_hex}");
    } else if result_hex != INTERFERENCE_RESULT_SHA256 {
        return fail(&format!(
            "static result hash changed: frozen {INTERFERENCE_RESULT_SHA256}, observed {result_hex}"
        ));
    }

    // matrix surface, with idle-relative tails
    let mut surface = serde_json::Map::new();
    for condition in CONDITIONS {
        let cname = condition_name(condition);
        let mut per_source = serde_json::Map::new();
        for name in SOURCES {
            if let Some(a) = matrix.get(cname).and_then(|m| m.get(name)) {
                let mut v = a.finish();
                let idle_p50 = matrix
                    .get("idle")
                    .and_then(|m| m.get(name))
                    .and_then(|i| TailSummary::summarize(&i.latencies).p50_ns);
                let here = TailSummary::summarize(&a.latencies).p50_ns;
                if let Some(obj) = v.as_object_mut() {
                    obj.insert(
                        "p50_over_idle".into(),
                        match (here, idle_p50) {
                            (Some(h), Some(i)) if i > 0 => {
                                serde_json::json!(h as f64 / i as f64)
                            }
                            _ => serde_json::Value::Null,
                        },
                    );
                }
                per_source.insert(name.to_string(), v);
            }
        }
        surface.insert(cname.to_string(), serde_json::Value::Object(per_source));
    }

    let (timer_min_ns, timer_median_ns) = timer_overhead_ns();
    let total_ns = sw.elapsed_ns().max(0) as u64;
    let object_count = cells.len();

    let verdict = Verdict::Supported;
    let params = CourtParams {
        universe: Some(manifest.universe.clone()),
        profile: Some(format!("{} + interference.v1", manifest.profile)),
        backend: Some("B2 / B3-warm / B4 / B5 under controllable load".into()),
        sample_rate_hz: None,
        quantum_frames: Some(QUANTUM_FRAMES),
        content_kind: Some(
            "flagship corpus (frozen before results): adversarial real-time load".into(),
        ),
        ..Default::default()
    };
    let mut builder = ReceiptBuilder::new("interference");
    builder
        .result(verdict)
        .result_detail(format!(
            "adversarial real-time load over {} frozen objects at {QUANTUM_FRAMES}-frame quanta; \
             conditions idle/cpu_burn/memory_bandwidth/storage_io on {threads} workers; every window \
             reproduced exactly; every controllable condition measured, {SOAK_SECS}s CPU-contention \
             soak; result sha256 {result_hex}",
            object_count,
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
                "quantum_frames": QUANTUM_FRAMES,
                "conditions": CONDITIONS.iter().map(|c| condition_name(*c)).collect::<Vec<_>>(),
                "workers": threads,
                "settle_ms": SETTLE_MS,
                "soak_secs": SOAK_SECS,
                "not_controlled": NOT_CONTROLLED,
                "latency_boundary": "harness-owned",
                "energy": "cumulative powercap energy_uj when available; a hwmon spot reading is \
                           evidence only, never workload energy. Never estimated",
            }),
        )
        .extra(
            "measurement",
            serde_json::json!({
                "timer_overhead_ns": { "min": timer_min_ns, "median": timer_median_ns },
                "matrix": surface,
                "soak": soak.json,
                "power_source_probe": power_source_probe,
                "energy_counter_probe": energy_counter_probe,
                "energy_samples": energy_samples,
                "total_ns": total_ns,
            }),
        )
        .extra("cells", serde_json::Value::Array(cells))
        .limitation(
            "only conditions this host can actually create are measured; compositor/display load, \
             competing GPU compute, GPU context contention, DVFS, thermal steady state and PCIe power \
             saving are reported NOT_CONTROLLED rather than claimed. CUDA/ROCm contention requires \
             those courts",
        )
        .limitation(
            "energy is NOT_AVAILABLE without a cumulative energy counter (powercap energy_uj): a \
             hwmon spot power reading is recorded but not converted into a workload-energy figure, \
             and nothing is derived from TDP or a model. The soak is a bounded CPU-contention \
             budget, not an unbounded endurance run",
        );
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court interference: {verdict}");
    println!("  objects: {object_count} | conditions: idle/cpu_burn/memory_bandwidth/storage_io");
    println!(
        "  soak: {} passes, {} deadline misses, drift {}",
        soak.passes, soak.deadline_misses, soak.drift_ratio
    );
    println!("  result sha256: {result_hex}");
    println!("  receipt: {}", path.display());
    Ok(verdict)
}

struct Soak {
    exact: bool,
    passes: u64,
    deadline_misses: u64,
    drift_ratio: f64,
    json: serde_json::Value,
}

/// Bounded soak under CPU contention: repeat the workload for a fixed budget and
/// report tail drift (second half vs first half) and accumulated misses.
fn soak(prepared: &[Prepared], threads: usize, load_dir: &Path) -> Result<Soak> {
    // A fixed subset keeps each pass short so several passes fit the budget.
    let subset: Vec<&Prepared> = prepared.iter().take(24).collect();
    let guard = LoadGuard::start(LoadKind::CpuBurn, threads, load_dir)?;
    std::thread::sleep(std::time::Duration::from_millis(SETTLE_MS));

    let soak_sw = Stopwatch::start();

    let mut head: Vec<u64> = Vec::new();
    let mut tail: Vec<u64> = Vec::new();
    let mut passes = 0u64;
    let mut deadline_misses = 0u64;
    let mut exact = true;
    let mut per_pass: Vec<u64> = Vec::new();
    let budget_ns = SOAK_SECS * 1_000_000_000;
    let mut elapsed = 0i64;
    while elapsed < budget_ns as i64 && passes < 1000 {
        let mut pass_misses = 0u64;
        for p in &subset {
            let ch = usize::from(p.channels);
            let plan = WindowPlan::build(&p.samples, ch)?;
            let mut dst = vec![0i32; QUANTUM_FRAMES as usize * ch];
            let mut scratch: Vec<u8> = Vec::new();
            let mut s = VoleBoundedSource::open("B5", p.b5.clone(), false)?;
            let t = run_traversal(&mut s, &plan, ch, p.sample_rate_hz, &mut dst, &mut scratch)?;
            exact &= t.exact;
            pass_misses += t.deadline_misses;
            if passes.is_multiple_of(2) {
                head.extend_from_slice(&t.latencies);
            } else {
                tail.extend_from_slice(&t.latencies);
            }
        }
        deadline_misses += pass_misses;
        per_pass.push(pass_misses);
        passes += 1;
        elapsed = soak_sw.elapsed_ns();
    }
    drop(guard);

    let head_p50 = TailSummary::summarize(&head).p50_ns;
    let tail_p50 = TailSummary::summarize(&tail).p50_ns;
    let drift_ratio = match (head_p50, tail_p50) {
        (Some(h), Some(t)) if h > 0 => t as f64 / h as f64,
        _ => 1.0,
    };
    let json = serde_json::json!({
        "condition": "cpu_burn",
        "objects": subset.len(),
        "passes": passes,
        "deadline_misses": deadline_misses,
        "per_pass_misses": per_pass,
        "head_latency": TailSummary::summarize(&head),
        "tail_latency": TailSummary::summarize(&tail),
        "drift_ratio_tail_over_head_p50": drift_ratio,
        "note": "a bounded soak, not an endurance run; head/tail alternate passes over the subset",
    });
    Ok(Soak {
        exact,
        passes,
        deadline_misses,
        drift_ratio,
        json,
    })
}
