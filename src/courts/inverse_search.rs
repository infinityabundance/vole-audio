//! `court inverse-search` — Phase L parallel/device inverse search.
//!
//! The inverse compiler's dominant search cost is the bounded period scan: for
//! every candidate period `p`, the exact periodic residual record count. That
//! whole vector is embarrassingly parallel (one independent period per worker),
//! so Phase L places it on four surfaces and requires them to agree:
//!
//! ```text
//! scalar     sequential host scan (the reference)
//! parallel   host threads over disjoint period ranges
//! CUDA       vole_period_scan, one device thread per period
//! ROCm       the same kernel from the AMDGPU code object
//! ```
//!
//! Two properties are asserted, and they are the whole point of the phase:
//!
//! 1. **placement has no semantics** — every surface produces identical
//!    per-period counts (they all call the shared
//!    `device::search_shared::period_records`), so the ranked period list is
//!    identical;
//! 2. **search output is only a proposal** — feeding the device-ranked period
//!    list into the compiler produces exactly the same accepted candidate set
//!    as the sequential scan, and every accepted candidate is re-verified by
//!    the normative exact evaluator (intrinsic closure + scalar observation +
//!    bounded seek).
//!
//! Honest measurement: the court reports the measured wall time of each
//! surface and the implied ratio, but makes no claim that the GPU wins. Work
//! per period is `O(frames / p)`, so the balance depends on the period bound;
//! the policy is "keep the faster implementation for the family", and the
//! receipt carries the numbers either way.

use crate::backend::cuda::SearchWorld;
use crate::device::search_shared::PERIOD_NOT_CLOSEABLE;
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder};
use crate::evidence::timing::Stopwatch;
use crate::hash::sha256::{Sha256, hex};
use crate::inverse::{self, PeriodScan, ReferenceLibrary, ScanSurface, SearchBudget};
use crate::status::Verdict;
use std::path::{Path, PathBuf};

/// Candidate periods scanned per fixture (the compiler's default bound).
pub const SEARCH_PERIODS: u32 = 512;
/// Periods proposed per fixture (the compiler's default keep count).
pub const SEARCH_KEEP: usize = 4;

const DEFAULT_PTX: &str = "scripts/out/vole_audio.ptx";
const DEFAULT_ROCM_ARTIFACT: &str = "scripts/out/vole_audio.amdgcn.elf";

/// Read the PTX artifact if present.
fn ptx_bytes() -> Option<(Vec<u8>, String)> {
    let path = std::env::var("VOLE_CUDA_PTX")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_PTX));
    let bytes = std::fs::read(path).ok()?;
    let sha = hex(&Sha256::digest(&bytes));
    Some((bytes, sha))
}

/// Hash of a scan's per-period counts (the cross-surface equality anchor).
fn scan_hash(scan: &PeriodScan) -> String {
    let mut h = Sha256::new();
    for c in &scan.counts {
        h.update(&c.to_le_bytes());
    }
    hex(&h.finalize())
}

/// Static identity of a search report (kinds, labels, costs, ops, frontier).
fn static_projection(r: &inverse::SearchReport) -> Vec<u8> {
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

/// Run the court; writes an immutable receipt under `receipts/inverse-search/`.
pub fn run(receipts_root: &Path) -> crate::error::Result<Verdict> {
    let fail = |why: &str| -> crate::error::Result<Verdict> {
        let mut b = ReceiptBuilder::new("inverse-search");
        b.result(Verdict::FailedCorrectness)
            .result_detail(format!("inverse search failed: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court inverse-search: FAILED_CORRECTNESS ({why})");
        eprintln!("  receipt: {}", path.display());
        Ok(Verdict::FailedCorrectness)
    };

    let fixtures = crate::courts::inverse::fixtures()?;
    if fixtures.is_empty() {
        return fail("no fixtures");
    }
    let library = ReferenceLibrary::new();
    let budget = SearchBudget::default();
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    // --- CUDA availability (typed, never assumed) -------------------------
    let ptx = ptx_bytes();
    let mut cuda_world = None;
    let mut cuda_row = serde_json::json!({
        "available": false,
        "state": "NOT_ATTEMPTED",
        "reason": "PTX artifact absent (run scripts/build-cuda-device.sh)",
    });
    if let Some((bytes, sha)) = &ptx {
        match SearchWorld::open(0, bytes, 4096, SEARCH_PERIODS) {
            Ok(world) => {
                cuda_row = serde_json::json!({
                    "available": true,
                    "artifact_sha256": sha,
                    "entry": crate::backend::cuda::search::SEARCH_KERNEL_ENTRY,
                    "threads_per_block": 128,
                });
                cuda_world = Some(world);
            }
            Err(e) => {
                cuda_row = serde_json::json!({
                    "available": false,
                    "state": "UNAVAILABLE",
                    "reason": format!("{e}"),
                    "artifact_sha256": sha,
                });
            }
        }
    }

    // --- ROCm capability (typed gate; compile evidence only) --------------
    let rocm_probe = crate::backend::rocm::RocmProbe::capture();
    let rocm_row = match &rocm_probe {
        Ok(probe) => {
            let (verdict, reason) = crate::courts::rocm_d0::phase_j_d0_gate(probe);
            serde_json::json!({
                "gate": verdict.label(),
                "reason": reason,
                "entry": crate::backend::rocm::search::SEARCH_ENTRY,
                "artifact": std::fs::metadata(DEFAULT_ROCM_ARTIFACT).map(|m| m.len()).ok(),
            })
        }
        Err(e) => serde_json::json!({
            "gate": "INCONCLUSIVE",
            "reason": format!("{e}"),
        }),
    };

    // --- Scan every fixture on every surface ------------------------------
    let mut cells: Vec<serde_json::Value> = Vec::new();
    let mut scalar_ns_total = 0u64;
    let mut parallel_ns_total = 0u64;
    let mut cuda_ns_total = 0u64;
    let mut cuda_compared = 0u64;
    let mut accepted_candidates_checked = 0u64;
    let mut result_hash = Sha256::new();

    for fx in &fixtures {
        // The periodic residual family is defined on a mono window, so the
        // scan runs over channel 0 (deinterleaved) for every fixture.
        let channels = usize::from(fx.channels);
        let frames = fx.frames as u32;
        let window: Vec<i32> = (0..fx.frames as usize)
            .map(|f| fx.samples[f * channels])
            .collect();
        inverse::search::validate_request(&window, frames, SEARCH_PERIODS)?;

        let sw = Stopwatch::start();
        let scalar = inverse::search::scan_scalar(&window, frames, SEARCH_PERIODS);
        let scalar_ns = sw.elapsed_ns().max(0) as u64;

        let sw = Stopwatch::start();
        let parallel = inverse::search::scan_parallel(&window, frames, SEARCH_PERIODS, threads)?;
        let parallel_ns = sw.elapsed_ns().max(0) as u64;

        if !inverse::search::scans_agree(&scalar, &parallel) {
            return fail(&format!(
                "{}: the parallel host scan diverged from the sequential scan",
                fx.name
            ));
        }

        // Device scan (identical counts required).
        let mut cuda_cell = serde_json::json!({ "ran": false });
        let mut cuda_periods: Option<Vec<u32>> = None;
        if let Some(world) = &cuda_world {
            let sw = Stopwatch::start();
            let counts = match world.scan(&window) {
                Ok(c) => c,
                Err(e) => return fail(&format!("{}: CUDA scan failed: {e}", fx.name)),
            };
            let cuda_ns = sw.elapsed_ns().max(0) as u64;
            let cuda_scan = PeriodScan { counts };
            if !inverse::search::scans_agree(&scalar, &cuda_scan) {
                return fail(&format!(
                    "{}: the CUDA period scan diverged from the sequential scan",
                    fx.name
                ));
            }
            cuda_ns_total += cuda_ns;
            cuda_compared += 1;
            cuda_periods = Some(cuda_scan.periods(SEARCH_KEEP));
            cuda_cell = serde_json::json!({
                "ran": true,
                "counts_sha256": scan_hash(&cuda_scan),
                "nanos": cuda_ns,
                "ratio_vs_scalar": cuda_ns as f64 / scalar_ns.max(1) as f64,
            });
        }

        let scalar_periods = scalar.periods(SEARCH_KEEP);
        if let Some(cp) = &cuda_periods
            && *cp != scalar_periods
        {
            return fail(&format!(
                "{}: device-ranked periods differ from the scalar ranking",
                fx.name
            ));
        }

        // Search output is only a proposal: the device-ranked periods must
        // produce exactly the sequential-scan candidate set, and every
        // accepted candidate must pass the full exactness battery.
        let baseline = inverse::compile_with(fx, &library, budget, None)?;
        let fed = inverse::compile_with(fx, &library, budget, Some(&scalar_periods))?;
        if static_projection(&baseline) != static_projection(&fed) {
            return fail(&format!(
                "{}: an externally ranked period list changed the accepted set",
                fx.name
            ));
        }
        for a in &fed.accepted {
            if !(a.intrinsic_exact && a.evaluator_exact && a.seek_exact) {
                return fail(&format!(
                    "{}: accepted candidate '{}' is not exact",
                    fx.name, a.label
                ));
            }
            accepted_candidates_checked += 1;
            result_hash.update(&[a.kind.tag()]);
            result_hash.update(a.label.as_bytes());
            result_hash.update(&[0]);
            result_hash.update(&a.cost.complete_bytes.to_le_bytes());
        }
        if !fed.literal_accepted() {
            return fail(&format!(
                "{}: literal fallback missing after a ranked search",
                fx.name
            ));
        }

        scalar_ns_total += scalar_ns;
        parallel_ns_total += parallel_ns;

        let rank: Vec<serde_json::Value> = scalar
            .ranked(SEARCH_KEEP)
            .into_iter()
            .map(|(p, records)| {
                serde_json::json!({
                    "period": p,
                    "records": if records == PERIOD_NOT_CLOSEABLE { serde_json::Value::Null } else { serde_json::json!(records) },
                })
            })
            .collect();
        cells.push(serde_json::json!({
            "fixture": fx.name,
            "channels": fx.channels,
            "frames": frames,
            "scan_bound": scalar.limit(),
            "counts_sha256": scan_hash(&scalar),
            "ranked": rank,
            "periodic_candidates_proposed": fed
                .accepted
                .iter()
                .filter(|a| a.kind == crate::inverse::CandidateKind::ResidualPeriodic)
                .count(),
            "accepted": fed.accepted.len(),
            "frontier": fed.frontier.len(),
            "scalar_nanos": scalar_ns,
            "parallel_nanos": parallel_ns,
            "parallel_ratio_vs_scalar": parallel_ns as f64 / scalar_ns.max(1) as f64,
            "cuda": cuda_cell,
        }));
    }

    // --- ROCm execution row (only when the gate authorizes it) ------------
    let mut rocm_executed = false;
    let mut rocm_cells: Vec<serde_json::Value> = Vec::new();
    if let Ok(probe) = &rocm_probe {
        let (verdict, _) = crate::courts::rocm_d0::phase_j_d0_gate(probe);
        if verdict == Verdict::Supported {
            let artifact = std::fs::read(
                std::env::var("VOLE_ROCM_ARTIFACT")
                    .unwrap_or_else(|_| DEFAULT_ROCM_ARTIFACT.to_string()),
            );
            match artifact {
                Ok(bytes) => {
                    let fx = &fixtures[0];
                    let channels = usize::from(fx.channels);
                    let window: Vec<i32> = (0..fx.frames as usize)
                        .map(|f| fx.samples[f * channels])
                        .collect();
                    let scanner = crate::backend::rocm::SearchWorldRocm::open(
                        0,
                        &bytes,
                        fx.frames as u32,
                        SEARCH_PERIODS,
                    )?;
                    let counts = scanner.scan(&window)?;
                    let scan = PeriodScan { counts };
                    let scalar =
                        inverse::search::scan_scalar(&window, fx.frames as u32, SEARCH_PERIODS);
                    if !inverse::search::scans_agree(&scalar, &scan) {
                        return fail("the ROCm period scan diverged from the sequential scan");
                    }
                    rocm_executed = true;
                    rocm_cells.push(serde_json::json!({
                        "fixture": fx.name,
                        "counts_sha256": scan_hash(&scan),
                        "surface": ScanSurface::Rocm.label(),
                    }));
                }
                Err(e) => {
                    rocm_cells.push(serde_json::json!({
                        "available": false,
                        "reason": format!("AMDGPU artifact unreadable: {e}"),
                    }));
                }
            }
        }
    }

    let result_hex = hex(&result_hash.finalize());
    let aggregate = serde_json::json!({
        "fixtures": fixtures.len(),
        "cuda_compared": cuda_compared,
        "cuda_requested": cuda_world.is_some(),
        "host_threads": threads,
        "scalar_nanos_total": scalar_ns_total,
        "parallel_nanos_total": parallel_ns_total,
        "parallel_ratio_vs_scalar": parallel_ns_total as f64 / scalar_ns_total.max(1) as f64,
        "cuda_nanos_total": if cuda_compared > 0 { serde_json::json!(cuda_ns_total) } else { serde_json::Value::Null },
        "cuda_ratio_vs_scalar": if cuda_compared > 0 {
            serde_json::json!(cuda_ns_total as f64 / scalar_ns_total.max(1) as f64)
        } else {
            serde_json::Value::Null
        },
        "accepted_candidates_checked": accepted_candidates_checked,
        "rocm_executed": rocm_executed,
    });

    let verdict = if cuda_world.is_none() {
        // CPU surfaces ran; the device comparison could not.
        if ptx.is_none() {
            Verdict::Inconclusive
        } else {
            Verdict::UnsupportedByHardware
        }
    } else {
        Verdict::Supported
    };

    let params = CourtParams {
        universe: Some("vole.audio.u1".into()),
        profile: Some("u1/v1 + vole.inverse.k1 + vole.inverse.l1".into()),
        backend: Some("scalar|parallel|cuda|rocm".into()),
        sample_rate_hz: Some(crate::courts::inverse::INVERSE_RATE_HZ),
        quantum_frames: Some(SEARCH_PERIODS),
        content_kind: Some("entropy-corpus-v1 (mono channel-0 search windows".into()),
        ..Default::default()
    };
    let mut builder = ReceiptBuilder::new("inverse-search");
    builder
        .result(verdict)
        .result_detail(format!(
            "period scan over {} fixtures x {} periods on {} surfaces; \
             device compared on {}; {} accepted candidates re-verified; result sha256 {result_hex}",
            fixtures.len(),
            SEARCH_PERIODS,
            if cuda_world.is_some() { 4 } else { 3 },
            cuda_compared,
            accepted_candidates_checked,
        ))
        .params(params)
        .provenance(Provenance {
            reference_hash: Some(result_hex),
            exact_equality: Some(cuda_world.is_some()),
            ..Default::default()
        })
        .extra(
            "budget",
            serde_json::json!({
                "max_period_scan": budget.max_period_scan,
                "max_residual_period_candidates": budget.max_residual_period_candidates,
            }),
        )
        .extra("aggregate", aggregate)
        .extra("cuda", cuda_row)
        .extra("rocm", rocm_row)
        .extra("rocm_cells", serde_json::Value::Array(rocm_cells))
        .extra("cells", serde_json::Value::Array(cells))
        .limitation(
            "period-scan placement only: the scan ranks candidate periods; every accepted \
             representation is re-verified by the normative exact evaluator",
        )
        .limitation(
            "no claim that the GPU wins: work per period is O(frames / p), so the balance \
             depends on the period bound; measured ratios are reported as-is",
        );
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court inverse-search: {verdict}");
    println!(
        "  fixtures: {} x {} periods; cuda compared on {}",
        fixtures.len(),
        SEARCH_PERIODS,
        cuda_compared
    );
    println!(
        "  parallel/scalar time ratio: {:.3}",
        parallel_ns_total as f64 / scalar_ns_total.max(1) as f64
    );
    println!("  receipt: {}", path.display());
    Ok(verdict)
}
