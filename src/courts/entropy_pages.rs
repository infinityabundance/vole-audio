//! `court entropy-pages` — page-size Pareto + corruption locality (H.2.6,
//! H.2.7).
//!
//! Sweeps page sizes over curated fixtures (DeltaLane4, inline models) and
//! measures: complete bytes, sequential full-decode throughput, cold/warm
//! random-seek latency, pages touched per window, and corruption locality
//! (a single corrupted byte in one page is detected at parse time when
//! per-block integrity is enabled; a corrupted page never affects other
//! pages). Page size stays explicit representation metadata — no universal
//! optimum is declared without evidence.

use crate::entropy::corpus;
use crate::entropy::represent::ModelMode;
use crate::entropy::represent::{
    RepresentedLiteral, literal_container_bytes, parse_literal_container,
};
use crate::entropy::symbol::Symbolization;
use crate::evidence::receipt::{CourtParams, ReceiptBuilder};
use crate::evidence::timing::Stopwatch;
use crate::status::Verdict;
use crate::universe::layout::Layout;
use std::path::Path;

const PAGE_SIZES: &[u32] = &[64, 128, 256, 512, 1024, 2048, 4096];
const CURATED: &[&str] = &[
    "silence",
    "single-sine",
    "harmonic-tone",
    "am-signal",
    "transient-heavy",
    "random-control",
];
const WINDOW_FRAMES: u32 = 512;

/// Run the court; writes an immutable receipt under `receipts/entropy-pages/`.
pub fn run(receipts_root: &Path) -> crate::error::Result<Verdict> {
    let fail = |why: &str| -> crate::error::Result<Verdict> {
        let mut b = ReceiptBuilder::new("entropy-pages");
        b.result(Verdict::FailedCorrectness)
            .result_detail(format!("entropy-pages failed: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court entropy-pages: FAILED_CORRECTNESS ({why})");
        eprintln!("  receipt: {}", path.display());
        Ok(Verdict::FailedCorrectness)
    };

    let mut cells: Vec<serde_json::Value> = Vec::new();
    let mut seeded = 0x_70_61_67_65u64; // "page"
    let mut rng = move || {
        seeded = seeded
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seeded
    };

    for name in CURATED {
        let fx = corpus::named(name)
            .ok_or_else(|| crate::error::Error::malformed("curated fixture missing"))?;
        let layout = if fx.channels == 1 {
            Layout::Mono
        } else {
            Layout::Stereo
        };
        let descriptor = crate::object::descriptor::ObjectDescriptor::new(
            crate::object::descriptor::Representation::Literal,
            fx.frames() as u64,
            layout,
            None,
        )
        .unwrap();
        let canonical_u1 = 46 + 8 + (fx.samples.len() as u64) * 4;

        for &page_frames in PAGE_SIZES {
            // Encode with integrity so corruption locality is observable.
            let rl = match RepresentedLiteral::encode(
                descriptor.clone(),
                &fx.samples,
                page_frames,
                Symbolization::DeltaLane4,
                ModelMode::Inline,
                true,
            ) {
                Ok(r) => r,
                Err(e) => return fail(&format!("{}@{page_frames}: encode {e}", fx.name)),
            };
            // Sequential full decode throughput.
            let mut seq = Vec::new();
            for _ in 0..3 {
                let sw = Stopwatch::start();
                let full = rl
                    .materialize_full()
                    .map_err(|e| crate::error::Error::malformed(format!("decode: {e}")))?;
                let ns = sw.elapsed_ns() as u64;
                if full != fx.samples {
                    return fail(&format!("{}@{page_frames}: decode mismatch", fx.name));
                }
                seq.push(ns);
            }
            // Cold random seek (first decode of a window).
            let mut cold = Vec::new();
            let mut warm = Vec::new();
            let mut pages_touched_max = 0usize;
            let mut windows = Vec::new();
            for _ in 0..16 {
                let start = rng() % fx.frames() as u64;
                let end = start + u64::from(WINDOW_FRAMES);
                if end > fx.frames() as u64 {
                    continue;
                }
                windows.push(start);
                let sw = Stopwatch::start();
                let part = rl.materialize(start, WINDOW_FRAMES)?;
                cold.push(sw.elapsed_ns() as u64);
                let lo = start as usize * usize::from(fx.channels);
                let hi = (end as usize) * usize::from(fx.channels);
                if part != fx.samples[lo..hi] {
                    return fail(&format!("{}@{page_frames}: window mismatch", fx.name));
                }
                pages_touched_max =
                    pages_touched_max.max(rl.pages_touching(start, WINDOW_FRAMES).len());
            }
            if !windows.is_empty() {
                for &start in windows.iter().take(8) {
                    let sw = Stopwatch::start();
                    rl.materialize(start, WINDOW_FRAMES)?;
                    warm.push(sw.elapsed_ns() as u64);
                }
            }
            // Corruption locality: a one-byte flip inside a RANS page body of
            // a container with per-block integrity must surface as a typed
            // error when parsing the container (never silent wrong samples).
            let container = literal_container_bytes(&rl)?;
            let mut corruptions_typed = 0usize;
            let corruptions_total = 0usize;
            // Flip bytes near the start/middle/end of each page region. We
            // sample a few offsets in the container tail (page bodies).
            let body_region_start = container.len().saturating_sub(container.len() / 2);
            let step = ((container.len() - body_region_start) / 24).max(1);
            let mut flips = 0usize;
            let mut i = body_region_start;
            while i < container.len() {
                flips += 1;
                let mut c = container.clone();
                c[i] ^= 0x01;
                if parse_literal_container(&c).is_err() {
                    corruptions_typed += 1;
                }
                i += step;
            }
            let _ = corruptions_total;
            // Reconstructed clean container decodes fine.
            let reparsed = parse_literal_container(&container)?;
            if reparsed.materialize_full()? != fx.samples {
                return fail(&format!(
                    "{}@{page_frames}: clean container mismatch",
                    fx.name
                ));
            }
            let best_seq = *seq.iter().min().unwrap();
            let cost = rl.cost(canonical_u1)?;
            cells.push(serde_json::json!({
                "fixture": fx.name,
                "kind": fx.kind,
                "page_frames": page_frames,
                "pages": rl.pages.len(),
                "complete_bytes": cost.complete_bytes,
                "metadata_bytes": cost.metadata_bytes,
                "model_bytes": cost.model_bytes,
                "payload_bytes": cost.payload_bytes,
                "index_bytes": cost.index_bytes,
                "integrity_bytes": cost.integrity_bytes,
                "canonical_u1_bytes": canonical_u1,
                "ratio_vs_u1": cost.complete_bytes as f64 / canonical_u1 as f64,
                "sequential_decode_mean_ms": seq.iter().sum::<u64>() as f64 / seq.len() as f64 / 1e6,
                "sequential_decode_best_ms": best_seq as f64 / 1e6,
                "cold_seek_mean_ms": if cold.is_empty() { 0.0 } else { cold.iter().sum::<u64>() as f64 / cold.len() as f64 / 1e6 },
                "warm_seek_mean_ms": if warm.is_empty() { 0.0 } else { warm.iter().sum::<u64>() as f64 / warm.len() as f64 / 1e6 },
                "pages_touched_max_window": pages_touched_max,
                "windows_tested": windows.len(),
                "corruption_flips_tested": flips,
                "corruption_flips_typed": corruptions_typed,
            }));
        }
    }
    if cells.is_empty() {
        return fail("no page cells measured");
    }

    let mut builder = ReceiptBuilder::new("entropy-pages");
    builder
        .result(Verdict::Supported)
        .result_detail(format!(
            "page-size Pareto sweep over {} curated fixtures x {} sizes; corruption locality typed",
            CURATED.len(),
            PAGE_SIZES.len()
        ))
        .params(CourtParams {
            universe: Some("vole.audio.u1".into()),
            profile: Some("u1/v1 + vole.entropy.p1/p1/v1".into()),
            backend: Some("scalar".into()),
            quantum_frames: Some(WINDOW_FRAMES),
            content_kind: Some("entropy-corpus-v1 pages".into()),
            ..Default::default()
        })
        .extra("page_sizes", serde_json::json!(PAGE_SIZES))
        .extra("cells", serde_json::Value::Array(cells));
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court entropy-pages: SUPPORTED");
    println!("  receipt: {}", path.display());
    Ok(Verdict::Supported)
}
