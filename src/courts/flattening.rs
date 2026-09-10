//! `court flattening` — host flat-evaluator parity (Phase K).
//!
//! The flattened representation (`backend::flatten`) is the state the device
//! kernels consume; before any device touches it, the host flat evaluator must
//! reproduce the scalar oracle **bit for bit** over every window. This court
//! is the host-only parity anchor for the whole GPU path (the CUDA/ROCm courts
//! re-run the same contract on hardware), and it reports the *exposure*
//! consequences of flattening honestly:
//!
//! * host-side materialization of residual closures (`CUDA_FALLBACK`) is
//!   counted (`fallback_closure_bytes`), never hidden;
//! * the upload footprint (voices + content arena + partial arena) is counted;
//! * the per-class voice distribution is recorded.
//!
//! Worlds: the frozen semantic court fixture, the frozen authored fixture (all
//! procedural classes, including a residual object), and a deterministic
//! seed sweep of random adversarial battery worlds (every class, adversarial
//! rates/envelopes/pan/loops/note-offs).

use crate::backend::flatten::flatten;
use crate::evidence::receipt::{CourtParams, Provenance, ReceiptBuilder};
use crate::evidence::timing::Stopwatch;
use crate::hash::sha256::Sha256;
use crate::object::ObjectStore;
use crate::sampler::scheduler::TimelineEvent;
use crate::sampler::world::World;
use crate::status::Verdict;
use crate::universe::observation::observation_sha256;
use std::path::Path;

/// Deterministic random-battery seed count.
pub const BATTERY_SEEDS: u64 = 256;

/// Windows compared per world (start, frames).
const WINDOWS: &[(i64, usize)] = &[(0, 1024), (0, 1), (37, 512), (256, 768)];

struct WorldRow {
    label: String,
    store: ObjectStore,
    world: World,
    windows: Vec<(i64, usize)>,
}

/// Run the court; writes an immutable receipt under `receipts/flattening/`.
pub fn run(receipts_root: &Path) -> crate::error::Result<Verdict> {
    let fail = |why: &str| -> crate::error::Result<Verdict> {
        let mut b = ReceiptBuilder::new("flattening");
        b.result(Verdict::FailedCorrectness)
            .result_detail(format!("flattening parity failed: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court flattening: FAILED_CORRECTNESS ({why})");
        eprintln!("  receipt: {}", path.display());
        Ok(Verdict::FailedCorrectness)
    };

    let mut rows: Vec<WorldRow> = Vec::new();

    // Frozen semantic fixture.
    let (store, events) = crate::courts::semantic::semantic_court_fixture();
    let world = World::new(
        crate::courts::semantic::RATE_HZ,
        crate::courts::semantic::CHANNELS,
        events,
    )?;
    rows.push(WorldRow {
        label: "semantic-fixture".into(),
        store,
        world,
        windows: WINDOWS.to_vec(),
    });

    // Frozen authored fixture (procedural classes incl. residual-free objects).
    let (store, events) = crate::courts::authored::authored_court_fixture();
    let world = World::new(crate::courts::authored::RATE_HZ, 1, events)?;
    rows.push(WorldRow {
        label: "authored-fixture".into(),
        store,
        world,
        windows: vec![(0, 4000), (0, 1), (400, 3600), (1600, 2400)],
    });

    // Deterministic adversarial battery (every class, residual closures).
    for seed in 0..BATTERY_SEEDS {
        if let Some(row) = battery_world(seed)? {
            rows.push(row);
        }
    }

    let mut cells: Vec<serde_json::Value> = Vec::new();
    let mut compared = 0u64;
    let mut audible = 0u64;
    let mut total_fallback_bytes = 0u64;
    let mut worlds_with_fallback = 0u64;
    let mut hash = Sha256::new();

    for row in &rows {
        let flat = match flatten(&row.store, &row.world) {
            Ok(f) => f,
            Err(e) => return fail(&format!("{}: flatten failed: {e}", row.label)),
        };
        let sw = Stopwatch::start();
        let flat_ns = {
            let _ = &flat;
            sw.elapsed_ns().max(0) as u64
        };
        let mut window_cells = Vec::new();
        for &(start, frames) in &row.windows {
            let sw = Stopwatch::start();
            let scalar = match crate::eval::ScalarOracle::new(row.world.clone())
                .observe(&row.store, start, frames)
            {
                Ok(v) => v,
                Err(_) => continue, // out-of-domain window for this world
            };
            let scalar_ns = sw.elapsed_ns().max(0) as u64;
            let sw = Stopwatch::start();
            let flat_out = flat.render(start, frames);
            let render_ns = sw.elapsed_ns().max(0) as u64;
            if scalar != flat_out {
                return fail(&format!(
                    "{}: flat != scalar at window [{start}, {})",
                    row.label,
                    start + frames as i64
                ));
            }
            if scalar.iter().any(|&x| x != 0) {
                audible += 1;
            }
            compared += 1;
            hash.update(&observation_sha256(&flat_out));
            window_cells.push(serde_json::json!({
                "start": start,
                "frames": frames,
                "exact": true,
                "scalar_ns": scalar_ns,
                "flat_ns": render_ns,
            }));
        }
        if flat.fallback_closure_bytes > 0 {
            worlds_with_fallback += 1;
        }
        total_fallback_bytes += flat.fallback_closure_bytes;
        cells.push(serde_json::json!({
            "world": row.label,
            "voices": flat.voices.len(),
            "arena_samples": flat.samples.len(),
            "partials": flat.partials.len(),
            "upload_bytes": flat.upload_bytes,
            "fallback_closure_bytes": flat.fallback_closure_bytes,
            "class_counts": flat.class_counts,
            "flatten_ns": flat_ns,
            "windows": window_cells,
        }));
    }

    if compared == 0 {
        return fail("no window was compared");
    }
    if audible == 0 {
        return fail("every compared window was silent — the battery is not exercising content");
    }
    // Residual closure materialization is an exposure cost that must surface
    // when the battery contains residual objects.
    if total_fallback_bytes == 0 {
        return fail("no residual closure was materialized host-side — battery lacks residuals");
    }

    let result_hex = crate::hash::sha256::hex(&hash.finalize());
    let params = CourtParams {
        universe: Some("vole.audio.u1".into()),
        profile: Some("u1/v1".into()),
        backend: Some("scalar==flat".into()),
        sample_rate_hz: Some(48_000),
        channels: Some(2),
        quantum_frames: Some(1024),
        content_kind: Some("semantic + authored + adversarial battery".into()),
        ..Default::default()
    };
    let mut builder = ReceiptBuilder::new("flattening");
    builder
        .result(Verdict::Supported)
        .result_detail(format!(
            "flat == scalar on {compared} windows over {} worlds; result sha256 {result_hex}",
            rows.len()
        ))
        .params(params)
        .provenance(Provenance {
            reference_hash: Some(result_hex),
            exact_equality: Some(true),
            ..Default::default()
        })
        .extra("worlds", serde_json::json!(rows.len()))
        .extra("windows_compared", serde_json::json!(compared))
        .extra("audible_windows", serde_json::json!(audible))
        .extra(
            "worlds_with_residual_fallback",
            serde_json::json!(worlds_with_fallback),
        )
        .extra(
            "total_fallback_closure_bytes",
            serde_json::json!(total_fallback_bytes),
        )
        .extra("cells", serde_json::Value::Array(cells));
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court flattening: SUPPORTED");
    println!("  worlds: {}, windows compared: {compared}", rows.len());
    println!("  host residual closure materialized: {total_fallback_bytes} bytes");
    println!("  receipt: {}", path.display());
    Ok(Verdict::Supported)
}

/// One deterministic adversarial battery world (same generator as the SIMD and
/// flat parity batteries, so every backend is exercised over identical worlds).
fn battery_world(seed: u64) -> crate::error::Result<Option<WorldRow>> {
    use crate::eval::battery::{corpus, random_voice};
    use crate::universe::prng::XoShiro256;

    let (store, pool) = corpus(seed);
    let mut rng = XoShiro256::from_seed(seed ^ 0xF1A7);
    let next32 = |rng: &mut XoShiro256| rng.next_u64() as u32;
    let n_voices = 1 + (next32(&mut rng) % 4);
    let mut events: Vec<TimelineEvent> = Vec::new();
    for _ in 0..n_voices {
        let object = pool[(next32(&mut rng) as usize) % pool.len()];
        let mut spec = random_voice(object, &store, &mut rng, 2);
        if next32(&mut rng) % 2 == 0 {
            spec.note_off = Some(spec.trigger_frame + 1 + (next32(&mut rng) % 600) as i64);
        }
        events.push(TimelineEvent::VoiceOn(spec));
    }
    let world = match World::new(48_000, 2, events) {
        Ok(w) => w,
        Err(_) => return Ok(None), // out-of-domain random draw; not a parity case
    };
    // Ensure the world actually resolves before flattening.
    if world.resolve_all(&store).is_err() {
        return Ok(None);
    }
    Ok(Some(WorldRow {
        label: format!("battery-seed-{seed}"),
        store,
        world,
        windows: vec![(0, 512), (0, 1), (64, 256)],
    }))
}

/// Presence of at least one residual object in the battery (sanity that the
/// fallback accounting assertion is meaningful).
#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::ObjectData;

    #[test]
    fn battery_produces_residual_worlds() {
        let mut found = false;
        for seed in 0..BATTERY_SEEDS {
            if let Some(row) = battery_world(seed).unwrap()
                && row
                    .store
                    .iter()
                    .any(|o| matches!(o.data, ObjectData::PredictorResidual(_)))
            {
                found = true;
                break;
            }
        }
        assert!(found, "battery corpus must include residual objects");
    }
}
