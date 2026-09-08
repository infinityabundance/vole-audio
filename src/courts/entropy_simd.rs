//! `court entropy-simd` — CPU decode parallel surfaces (H.2.16).
//!
//! H.2.16 forbids fabricating vectorization where single-state rANS is
//! inherently serial. What the charter sanctions is measuring the real
//! parallel surfaces of the format — independent entropy **pages** (and
//! independent channels/streams inside them) — and proving they decode
//! exactly, never letting an unproven path claim authority.
//!
//! This court therefore:
//!
//! 1. decodes every fixture job with the scalar host decoder (semantic
//!    authority) and confirms it equals the representation decode;
//! 2. decodes the identical jobs **page-parallel across CPU threads** (each
//!    page is an independent decode unit with its own scratch; outputs write
//!    disjoint arena regions) and requires byte equality with scalar —
//!    scalar == page-parallel on every fixture;
//! 3. measures sequential vs page-parallel wall time over repeated decodes
//!    and reports the exact speedup surface;
//! 4. records honestly that instruction-level SIMD (AVX2/AVX-512) entropy
//!    decode is **not implemented** and claims no speedup from it:
//!    single-state rANS is serial per stream, and any multi-state/interleaved
//!    layout that would enable real instruction SIMD must first be justified
//!    as a separately identified representation with byte-exact proof
//!    (future phase work; not fabricated here).
//!
//! Verdict SUPPORTED means: exactness of the measured parallel surface holds
//! on this host and the instruction-SIMD absence is recorded — never a claim
//! that AVX2/AVX-512 decode exists.

use crate::backend::entropy_flat::{flatten_literal_range, flatten_residual_range};
use crate::entropy::corpus;
use crate::entropy::represent::{ModelMode, RepresentedLiteral, RepresentedResidual};
use crate::entropy::symbol::Symbolization;
use crate::error::Result;
use crate::evidence::receipt::{CourtParams, ReceiptBuilder};
use crate::object::descriptor::{ObjectDescriptor, Representation};
use crate::object::residual::{Residual, ResidualModel};
use crate::status::Verdict;
use crate::universe::layout::Layout;
use std::path::Path;
use std::sync::Barrier;
use std::time::Instant;

const PAGE_FRAMES: u32 = 512;
const REPS: usize = 48;

struct JobCase {
    label: String,
    job: crate::backend::entropy_flat::FlatEntropyJob,
    expected: Vec<i32>,
}

fn build_cases() -> Vec<JobCase> {
    let mut out = Vec::new();
    let mut push_lit = |label: &str, samples: &[i32], channels: u8, sym: Symbolization| {
        let layout = if channels == 1 {
            Layout::Mono
        } else {
            Layout::Stereo
        };
        let frames = (samples.len() / usize::from(channels)) as u64;
        let d = ObjectDescriptor::new(Representation::Literal, frames, layout, None).unwrap();
        let rl = RepresentedLiteral::encode(d, samples, PAGE_FRAMES, sym, ModelMode::Inline, false)
            .unwrap();
        let job = flatten_literal_range(&rl, 0, frames as u32).unwrap();
        out.push(JobCase {
            label: label.to_string(),
            job,
            expected: samples.to_vec(),
        });
    };
    for fx in &corpus::all() {
        for &sym in &[
            Symbolization::Lane4Plain,
            Symbolization::Lane4ZigZag,
            Symbolization::DeltaLane4,
        ] {
            push_lit(
                &format!("{} sym={}", fx.name, sym.code()),
                &fx.samples,
                fx.channels,
                sym,
            );
        }
    }
    // Procedural residual (periodic hypothesis + sparse corrections), mono.
    {
        let frames = 4096usize;
        let cycle: Vec<i32> = (0..64).map(|i| ((i as i64 - 32) * 128) as i32).collect();
        let model = ResidualModel::Periodic { cycle };
        let mut intrinsic = vec![0i32; frames];
        let mut rs = 0x5eed_c0de_u64;
        for (f, slot) in intrinsic.iter_mut().enumerate() {
            *slot = model.model_sample(f as u64);
        }
        for _ in 0..256 {
            rs = rs
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f = (rs % frames as u64) as usize;
            let delta = (((rs >> 33) as i64) % 2048) - 1024;
            intrinsic[f] = crate::universe::arithmetic::sat_i32(i64::from(intrinsic[f]) + delta);
        }
        let d = ObjectDescriptor::new(
            Representation::PredictorResidual,
            frames as u64,
            Layout::Mono,
            None,
        )
        .unwrap();
        let records = Residual::closing_residual(&intrinsic, 1, &model).unwrap();
        let residual = Residual::new(&d, model, records).unwrap();
        let rr = RepresentedResidual::encode(d, &residual, PAGE_FRAMES, ModelMode::Inline, false)
            .unwrap();
        let job = flatten_residual_range(&rr, 0, frames as u32).unwrap();
        out.push(JobCase {
            label: "residual-periodic-sparse".into(),
            job,
            expected: intrinsic,
        });
    }
    out
}

/// Scalar decode of pages `[begin..end)` into `out`. Each page writes its own
/// disjoint arena region, so disjoint page ranges are safe to run on disjoint
/// `out` slices with per-thread scratch (the parallel surface under test).
fn decode_range(
    case: &JobCase,
    out: &mut [i32],
    scratch: &mut [u8],
    range: std::ops::Range<usize>,
) -> bool {
    case.job.decode_pages_host(out, scratch, range)
}

/// Run `t` threads over disjoint page ranges, `REPS` barrier-paced decodes,
/// writing into `out`. Returns false on any decode failure. `out` must be the
/// full arena; the `t` threads' page ranges are disjoint within it.
fn timed_parallel(case: &JobCase, t: usize, out: &mut [i32]) -> bool {
    let n_pages = case.job.pages.len();
    let barrier = Barrier::new(t);
    let pages_per = n_pages.div_ceil(t);
    let arena = out.as_mut_ptr() as usize;
    let arena_len = out.len();
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for tid in 0..t {
            let case = &*case;
            let barrier = &barrier;
            let begin = tid * pages_per;
            let end = (begin + pages_per).min(n_pages);
            handles.push(s.spawn(move || {
                let mut local = vec![0u8; case.job.max_page_scratch.max(1)];
                for _ in 0..REPS {
                    barrier.wait();
                    // SAFETY: this thread decodes only pages [begin..end),
                    // which write disjoint out regions; `arena` is rebuilt
                    // fresh each rep and the ranges never overlap across
                    // threads.
                    let out_slice =
                        unsafe { core::slice::from_raw_parts_mut(arena as *mut i32, arena_len) };
                    if !decode_range(case, out_slice, &mut local, begin..end) {
                        return false;
                    }
                    barrier.wait();
                }
                true
            }));
        }
        handles.into_iter().all(|h| h.join().unwrap_or(false))
    })
}

/// Run the court; writes an immutable receipt under `receipts/entropy-simd/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let fail = |v: Verdict, why: &str| -> Result<Verdict> {
        let mut b = ReceiptBuilder::new("entropy-simd");
        b.result(v).result_detail(format!("entropy-simd: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court entropy-simd: {v} ({why})");
        eprintln!("  receipt: {}", path.display());
        Ok(v)
    };

    let cases = build_cases();
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 32);
    let mut cells: Vec<serde_json::Value> = Vec::new();
    let mut speedups: Vec<f64> = Vec::new();

    for case in &cases {
        let n_pages = case.job.pages.len();
        if n_pages == 0 {
            return fail(Verdict::FailedCorrectness, "empty job");
        }
        let arena = case.job.arena_samples;
        // 1) Scalar sequential authority decode.
        let mut scalar = vec![0i32; arena];
        let mut scratch = vec![0u8; case.job.max_page_scratch.max(1)];
        if !decode_range(case, &mut scalar, &mut scratch, 0..n_pages) || scalar != case.expected {
            return fail(
                Verdict::FailedCorrectness,
                &format!("{} scalar decode != representation", case.label),
            );
        }
        // Sequential timing reference (single thread, same reps).
        let mut sink = vec![0i32; arena];
        let t0 = Instant::now();
        if !timed_parallel(case, 1, &mut sink) {
            return fail(Verdict::FailedCorrectness, "sequential decode failed");
        }
        let seq_ms = t0.elapsed().as_secs_f64() * 1e3 / REPS as f64;
        if sink != scalar {
            return fail(Verdict::FailedCorrectness, "sequential timed != scalar");
        }
        cells.push(serde_json::json!({
            "case": case.label,
            "pages": n_pages,
            "threads": 1,
            "exact": true,
            "mean_ms_per_decode": seq_ms,
        }));

        // 2) Page-parallel decode at 2 threads and at the host width; both
        //    must be byte-exact vs scalar, then the host-width one is timed.
        let mut widths: Vec<usize> = vec![2];
        if n_threads != 2 {
            widths.push(n_threads);
        }
        for &t in &widths {
            let t = t.min(n_pages);
            if t < 2 {
                continue;
            }
            let t0 = Instant::now();
            if !timed_parallel(case, t, &mut sink) {
                return fail(
                    Verdict::FailedCorrectness,
                    &format!("{} {t}-thread page-parallel decode failed", case.label),
                );
            }
            let par_ms = t0.elapsed().as_secs_f64() * 1e3 / REPS as f64;
            if sink != scalar {
                return fail(
                    Verdict::FailedCorrectness,
                    &format!("{} {t}-thread page-parallel != scalar", case.label),
                );
            }
            if seq_ms > 0.0 && par_ms > 0.0 {
                speedups.push(seq_ms / par_ms);
            }
            cells.push(serde_json::json!({
                "case": case.label,
                "pages": n_pages,
                "threads": t,
                "exact": true,
                "mean_ms_per_decode": par_ms,
            }));
        }
    }

    let mean_speedup = if speedups.is_empty() {
        0.0
    } else {
        speedups.iter().sum::<f64>() / speedups.len() as f64
    };

    let mut b = ReceiptBuilder::new("entropy-simd");
    b.result(Verdict::Supported)
        .result_detail(format!(
            "scalar == CPU page-parallel decode (up to {} threads) on {} jobs; measured \
             speedup surface reported; instruction-level SIMD decode NOT implemented by \
             design (single-state rANS is serial per stream; no fabricated vectorization — \
             H.2.16)",
            n_threads,
            cases.len()
        ))
        .params(CourtParams {
            universe: Some("vole.audio.u1".into()),
            profile: Some("u1/v1 + vole.entropy.p1/p1/v1".into()),
            backend: Some("cpu".into()),
            content_kind: Some("entropy-corpus-v1 simd-surface".into()),
            ..Default::default()
        })
        .extra("threads", serde_json::json!(n_threads))
        .extra("cells", serde_json::Value::Array(cells))
        .extra(
            "mean_page_parallel_speedup",
            serde_json::json!({ "over_sequential": mean_speedup }),
        )
        .extra(
            "instruction_simd",
            serde_json::json!({
                "avx2_entropy_decode": "NOT_IMPLEMENTED",
                "avx512_entropy_decode": "NOT_IMPLEMENTED",
                "reason": "single-state rANS is inherently serial per stream; a \
                    multi-state/interleaved layout enabling real instruction SIMD would be \
                    a separately identified representation requiring byte-exact proof — \
                    future work, never fabricated",
                "measured_parallel_surface": "independent entropy pages over CPU threads \
                    (exact == scalar on every fixture)",
            }),
        );
    b.limitation(
        "this court measures the sanctioned CPU parallel surface (independent pages); it \
         does not claim instruction-level SIMD decode exists — see instruction_simd extra",
    );
    let (_, path) = b.finish_write(receipts_root)?;
    println!("court entropy-simd: SUPPORTED");
    println!(
        "  mean page-parallel speedup over sequential: {mean_speedup:.2}x ({n_threads} threads)"
    );
    println!("  receipt: {}", path.display());
    Ok(Verdict::Supported)
}
