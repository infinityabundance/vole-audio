//! `court runtime-advanced` — Report 3 whole-repository runtime surfaces.
//!
//! Optional, explicitly-reported host surfaces beside the frozen B2–B5
//! architectures (nothing here replaces a baseline or changes a representation):
//!
//! * **hybrid deadline entry** — sleep to `deadline − δ`, spin for the final δ,
//!   over a frozen δ ladder; the court records the wake-error tail.
//! * **huge-page execution arenas** — an `mmap` + `MADV_HUGEPAGE` buffer, with
//!   the kernel setting and the process `AnonHugePages` delta.
//! * **real-time scheduling + memory locking** — `SCHED_FIFO`, `SCHED_DEADLINE`
//!   and `mlockall(MCL_CURRENT|MCL_FUTURE)`, attempted and reported honestly
//!   (they usually require `CAP_SYS_NICE`/`CAP_IPC_LOCK`).
//!
//! Only deterministic facts enter the projection; wall-clock tail statistics are
//! recorded as extras, never hashed.

use crate::evidence::counters::Counters;
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder};
use crate::hash::sha256::{Sha256, hex};
use crate::runtime::advanced::{
    HugeArena, anon_huge_pages_kib, lateness_ns, transparent_hugepage_setting, try_mlockall,
    try_sched_deadline, try_sched_fifo, unlock_all, wait_until,
};
use crate::status::Verdict;
use std::collections::BTreeMap;
use std::path::Path;
use std::time::{Duration, Instant};

/// The frozen hybrid spin ladder (ns).
const SPIN_LADDER: [u64; 5] = [0, 10_000, 25_000, 50_000, 100_000];

/// Deadline spacing (ns) and iterations per policy.
const DEADLINE_NS: u64 = 500_000;
const ITERS: u32 = 250;

/// Huge-page arena size (bytes).
const ARENA_BYTES: usize = 64 << 20;

fn outcome_json(o: &crate::runtime::advanced::ScheduleOutcome) -> serde_json::Value {
    serde_json::json!({
        "policy": o.policy,
        "applied": o.applied,
        "errno": o.errno,
        "detail": o.detail,
    })
}

/// Run the court; writes `receipts/runtime-advanced/`.
pub fn run(receipts_root: &Path) -> crate::error::Result<Verdict> {
    let mut extras: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut projection = Vec::new();

    // --- 1. Hybrid deadline entry ---
    let mut hybrid_rows = Vec::new();
    for &spin in &SPIN_LADDER {
        let mut late: Vec<i64> = Vec::with_capacity(ITERS as usize);
        for i in 0..ITERS {
            // Stagger start instants so deadlines do not align to one tick.
            let stagger = Duration::from_nanos(u64::from((i * 37) % 300) * 1000);
            let deadline = Instant::now() + stagger + Duration::from_nanos(DEADLINE_NS);
            wait_until(deadline, spin);
            late.push(lateness_ns(deadline));
        }
        late.sort_unstable();
        let n = late.len();
        let p50 = late[n / 2];
        let p99 = late[(n * 99 / 100).min(n - 1)];
        let max = late[n - 1];
        let late_count = late.iter().filter(|&&v| v > 1_000_000).count();
        projection.extend_from_slice(&spin.to_le_bytes());
        hybrid_rows.push(serde_json::json!({
            "spin_ns": spin,
            "p50_ns": p50,
            "p99_ns": p99,
            "max_ns": max,
            "late_over_1ms": late_count,
        }));
    }
    extras.insert(
        "hybrid_deadline_entry".into(),
        serde_json::json!({
            "deadline_ns": DEADLINE_NS,
            "iterations": ITERS,
            "policies": hybrid_rows,
        }),
    );

    // --- 2. Huge-page execution arena ---
    let thp_setting = transparent_hugepage_setting();
    let before = anon_huge_pages_kib().unwrap_or(0);
    let mut mapped = false;
    let mut hint_accepted = false;
    let mut after = before;
    if let Some(mut arena) = HugeArena::new(ARENA_BYTES) {
        mapped = true;
        hint_accepted = arena.huge_page_hint();
        arena.touch_all();
        after = anon_huge_pages_kib().unwrap_or(before);
    }
    projection.extend_from_slice(thp_setting.as_deref().unwrap_or("").as_bytes());
    projection.push(u8::from(mapped));
    projection.push(u8::from(hint_accepted));
    extras.insert(
        "huge_pages".into(),
        serde_json::json!({
            "arena_bytes": ARENA_BYTES,
            "thp_setting": thp_setting,
            "mapped": mapped,
            "hint_accepted": hint_accepted,
            "anon_huge_kib_before": before,
            "anon_huge_kib_after": after,
            "anon_huge_kib_delta": after.saturating_sub(before),
        }),
    );

    // --- 3. Real-time scheduling + memory locking ---
    let fifo = try_sched_fifo(1);
    let sched_deadline = try_sched_deadline(1_000_000, 5_000_000, 5_000_000);
    let mlock = try_mlockall();
    if mlock.applied {
        unlock_all();
    }
    // Restore the default policy so later courts are not measured under a
    // real-time scheduler the user did not ask for.
    let restored = crate::runtime::advanced::try_sched_other();
    projection.push(u8::from(fifo.applied));
    projection.push(fifo.errno.unwrap_or(0) as u8);
    projection.push(u8::from(sched_deadline.applied));
    projection.push(sched_deadline.errno.unwrap_or(0) as u8);
    projection.push(u8::from(mlock.applied));
    projection.push(mlock.errno.unwrap_or(0) as u8);
    projection.push(u8::from(restored.applied));
    extras.insert(
        "scheduler".into(),
        serde_json::json!({
            "sched_fifo": outcome_json(&fifo),
            "sched_deadline": outcome_json(&sched_deadline),
            "mlockall": outcome_json(&mlock),
            "restored_default": outcome_json(&restored),
            "note": "real-time policies require CAP_SYS_NICE / CAP_IPC_LOCK; a refusal is an \
                     honest host limitation, not a failure; the default policy is restored \
                     before the court returns",
        }),
    );

    let reference_hash = hex(&Sha256::digest(&projection));
    extras.insert(
        "method".into(),
        serde_json::json!({
            "hybrid": "sleep to deadline - spin_ns, then spin_loop; lateness = now - deadline",
            "huge_pages": "mmap(PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS) + madvise(MADV_HUGEPAGE)",
            "scheduling": "sched_setscheduler(SCHED_FIFO), sched_setattr(SCHED_DEADLINE), mlockall",
            "projection": "deterministic facts only; wall-clock tails are extras, never hashed",
        }),
    );

    let mut counters = Counters::new();
    counters.quanta_submitted = iters_total();
    let mut builder = ReceiptBuilder::new("runtime-advanced");
    builder
        .result(Verdict::Supported)
        .result_detail(format!(
            "hybrid deadline entry over {} spin budgets, a {} MiB huge-page arena, and \
             SCHED_FIFO/SCHED_DEADLINE/mlockall attempts; deterministic projection {reference_hash}",
            SPIN_LADDER.len(),
            ARENA_BYTES >> 20,
        ))
        .params(CourtParams {
            universe: Some("vole.audio.u1".into()),
            profile: Some("u1/v1".into()),
            backend: Some("host runtime surfaces".into()),
            content_kind: Some("report-3 whole-repository runtime program".into()),
            ..Default::default()
        })
        .counters(counters)
        .provenance(Provenance {
            reference_hash: Some(reference_hash.clone()),
            benchmark_order: SPIN_LADDER.iter().map(|s| format!("spin-{s}ns")).collect(),
            ..Default::default()
        });
    for (k, v) in extras {
        builder.extra(k, v);
    }
    builder.limitation(
        "the advanced runtime surfaces are optional and additive: they never replace B2-B5, and a \
         host that refuses a real-time policy or huge pages is reported, not worked around",
    );
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court runtime-advanced: SUPPORTED");
    println!("  result sha256: {reference_hash}");
    println!("  receipt: {}", path.display());
    Ok(Verdict::Supported)
}

const fn iters_total() -> u64 {
    (SPIN_LADDER.len() as u64) * (ITERS as u64)
}
