//! `court entropy-cuda` — CUDA entropy decode parity (H.2.17/H.2.18).
//!
//! Loads the PTX artifact built from this same package (which contains
//! `vole_entropy_decode`: one thread per entropy page over the shared
//! `device::entropy_shared` decoder), uploads flat entropy jobs, launches the
//! decode kernel, and requires:
//!
//! * every page's kernel status == 1 (bounded, exact decode);
//! * the downloaded arena equals the host flat decoder byte-for-byte on the
//!   identical job;
//! * the host flat decoder equals the representation decode (scalar
//!   authority) — so scalar == CUDA decoded symbols/observations.
//!
//! Coverage: entropy literal pages (DeltaLane4 over a stereo tone; noise
//! forcing RAW-fallback pages) and a procedural residual object with sparse
//! corrections (mask + delta lanes; closure samples on the device).
//!
//! GPU absent / artifact absent -> honest `UNSUPPORTED_BY_HARDWARE` /
//! `INCONCLUSIVE` receipts; never fabricated.

use crate::backend::cuda::entropy::EntropyWorld;
use crate::backend::entropy_flat::{flatten_literal_range, flatten_residual_range};
use crate::entropy::represent::ModelMode;
use crate::entropy::represent::{RepresentedLiteral, RepresentedResidual};
use crate::entropy::symbol::Symbolization;
use crate::evidence::receipt::{CourtParams, ReceiptBuilder};
use crate::evidence::timing::Stopwatch;
use crate::object::descriptor::{ObjectDescriptor, Representation};
use crate::object::residual::{Residual, ResidualModel};
use crate::status::Verdict;
use crate::universe::layout::Layout;
use std::path::Path;

/// Default PTX artifact produced by scripts/build-cuda-device.sh.
const DEFAULT_PTX: &str = "scripts/out/vole_audio.ptx";

fn ptx_bytes() -> crate::error::Result<Option<(Vec<u8>, String)>> {
    use crate::error::{Error, Kind};
    let path = std::env::var("VOLE_CUDA_PTX").unwrap_or_else(|_| DEFAULT_PTX.to_string());
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(Error::new(Kind::Io, format!("read {path}: {e}"))),
    };
    let sha = crate::hash::sha256::hex(&crate::hash::sha256::Sha256::digest(&bytes));
    Ok(Some((bytes, sha)))
}

fn tone(frames: usize, ch: usize) -> Vec<i32> {
    let mut s = 0x7777_aaaa_u64;
    (0..frames * ch)
        .map(|k| {
            let f = k / ch;
            let mut v: i64 = (((f as i64) * 7) % 2047) - 1023;
            v *= 8192;
            let wobble = (((f as i64) * 31) % 127) - 63;
            v += wobble * 33;
            if k % ch == 1 {
                v = -v;
            }
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            v + ((s >> 40) as i64 % 65) - 32
        })
        .collect::<Vec<i64>>()
        .into_iter()
        .map(|x| x as i32)
        .collect()
}

fn noise(frames: usize, ch: usize) -> Vec<i32> {
    let mut s = 0x1234_5678_9abc_def0_u64;
    (0..frames * ch)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s as u32) as i32
        })
        .collect()
}

/// Run the court; writes an immutable receipt under `receipts/entropy-cuda/`.
pub fn run(receipts_root: &Path) -> crate::error::Result<Verdict> {
    let fail = |verdict: Verdict, why: &str| -> crate::error::Result<Verdict> {
        let mut b = ReceiptBuilder::new("entropy-cuda");
        b.result(verdict)
            .result_detail(format!("entropy-cuda: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court entropy-cuda: {verdict} ({why})");
        eprintln!("  receipt: {}", path.display());
        Ok(verdict)
    };

    // PTX artifact.
    let ptx = match ptx_bytes()? {
        Some(p) => p,
        None => return fail(Verdict::Inconclusive, "PTX artifact absent"),
    };

    // Build jobs.
    let mut cells: Vec<serde_json::Value> = Vec::new();

    // Job 1: entropy literal (delta_lane4 stereo tone) with RAW fallback
    // pages (add a noise object as job 2).
    let frames = 4096usize;
    let samples = tone(frames, 2);
    let d_lit = ObjectDescriptor::new(Representation::Literal, frames as u64, Layout::Stereo, None)
        .unwrap();
    let rl = RepresentedLiteral::encode(
        d_lit,
        &samples,
        512,
        Symbolization::DeltaLane4,
        ModelMode::Inline,
        false,
    )
    .unwrap();
    let lit_job = flatten_literal_range(&rl, 0, frames as u32).unwrap();

    let noise_samples = noise(2048, 2);
    let d_noise =
        ObjectDescriptor::new(Representation::Literal, 2048, Layout::Stereo, None).unwrap();
    let rn = RepresentedLiteral::encode(
        d_noise,
        &noise_samples,
        512,
        Symbolization::DeltaLane4,
        ModelMode::Inline,
        false,
    )
    .unwrap();
    let noise_job = flatten_literal_range(&rn, 0, 2048).unwrap();

    // Job 3: procedural residual object (periodic hypothesis + sparse exact
    // corrections).
    let rframes = 4096usize;
    let cycle: Vec<i32> = (0..64).map(|i| i << 20).collect();
    let model = ResidualModel::Periodic { cycle };
    let mut intrinsic = vec![0i32; rframes];
    for (f, slot) in intrinsic.iter_mut().enumerate() {
        *slot = model.model_sample(f as u64);
    }
    let mut rs = 0xabcdef12u64;
    for _ in 0..300 {
        rs = rs
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let f = (rs % rframes as u64) as usize;
        let delta = (((rs >> 33) as i64) % 20000) as i32 - 10000;
        intrinsic[f] =
            crate::universe::arithmetic::sat_i32(i64::from(intrinsic[f]) + i64::from(delta));
    }
    let d_res = ObjectDescriptor::new(
        Representation::PredictorResidual,
        rframes as u64,
        Layout::Mono,
        None,
    )
    .unwrap();
    let records = Residual::closing_residual(&intrinsic, 1, &model).unwrap();
    let residual = Residual::new(&d_res, model, records).unwrap();
    let rr = RepresentedResidual::encode(d_res, &residual, 512, ModelMode::Inline, false).unwrap();
    let res_job = flatten_residual_range(&rr, 0, rframes as u32).unwrap();

    // Host reference decodes.
    let mut host_out = vec![0i32; lit_job.arena_samples];
    let mut scratch = vec![0u8; lit_job.max_page_scratch.max(1)];
    if !lit_job.decode_pages_host(&mut host_out, &mut scratch, 0..lit_job.pages.len()) {
        return fail(
            Verdict::FailedCorrectness,
            "host decode of literal job failed",
        );
    }
    let mut host_noise = vec![0i32; noise_job.arena_samples];
    let mut scratch_n = vec![0u8; noise_job.max_page_scratch.max(1)];
    if !noise_job.decode_pages_host(&mut host_noise, &mut scratch_n, 0..noise_job.pages.len()) {
        return fail(
            Verdict::FailedCorrectness,
            "host decode of noise job failed",
        );
    }
    let mut host_res = vec![0i32; res_job.arena_samples];
    let mut scratch_r = vec![0u8; res_job.max_page_scratch.max(1)];
    if !res_job.decode_pages_host(&mut host_res, &mut scratch_r, 0..res_job.pages.len()) {
        return fail(
            Verdict::FailedCorrectness,
            "host decode of residual job failed",
        );
    }
    // Scalar authority checks.
    if host_out != samples || host_noise != noise_samples || host_res != intrinsic {
        return fail(
            Verdict::FailedCorrectness,
            "host flat decode != scalar representation decode",
        );
    }

    // Device decode.
    let world = match EntropyWorld::open(0, &ptx.0, &lit_job) {
        Ok(w) => w,
        Err(e) if e.kind() == crate::error::Kind::Unavailable => {
            return fail(
                Verdict::UnsupportedByHardware,
                &format!("CUDA unavailable: {e}"),
            );
        }
        Err(e) => return fail(Verdict::Inconclusive, &format!("CUDA setup: {e}")),
    };
    // Run job 1.
    let mut dev_out = match world.decode() {
        Ok(o) => o,
        Err(e) => return fail(Verdict::FailedCorrectness, &format!("kernel decode: {e}")),
    };
    let sw = Stopwatch::start();
    for _ in 0..10 {
        dev_out = world.decode()?;
    }
    let decode_ms = sw.elapsed().as_millis_f64();
    if dev_out != host_out {
        return fail(Verdict::FailedCorrectness, "CUDA arena != host flat decode");
    }
    cells.push(serde_json::json!({
        "job": "literal-delta-lane4-stereo",
        "frames": frames,
        "channels": 2,
        "pages": lit_job.pages.len(),
        "arena_samples": lit_job.arena_samples,
        "payload_bytes": lit_job.payload.len(),
        "models": lit_job.model_ranges.len() / 2,
        "scalar_equal_cuda": true,
        "decode_mean_ms": decode_ms / 10.0,
    }));

    // Job 2 (noise, RAW fallback) — separate world (arena geometry differs).
    let world2 = match EntropyWorld::open(0, &ptx.0, &noise_job) {
        Ok(w) => w,
        Err(e) => return fail(Verdict::Inconclusive, &format!("CUDA setup 2: {e}")),
    };
    let dev_noise = world2.decode()?;
    if dev_noise != host_noise {
        return fail(
            Verdict::FailedCorrectness,
            "CUDA noise arena != host decode",
        );
    }
    cells.push(serde_json::json!({
        "job": "literal-noise-raw-fallback",
        "frames": 2048,
        "channels": 2,
        "pages": noise_job.pages.len(),
        "scalar_equal_cuda": true,
    }));

    // Job 3 (residual closure).
    let world3 = match EntropyWorld::open(0, &ptx.0, &res_job) {
        Ok(w) => w,
        Err(e) => return fail(Verdict::Inconclusive, &format!("CUDA setup 3: {e}")),
    };
    let dev_res = world3.decode()?;
    if dev_res != host_res {
        return fail(
            Verdict::FailedCorrectness,
            "CUDA residual arena != host decode",
        );
    }
    cells.push(serde_json::json!({
        "job": "procedural-residual-closure",
        "frames": rframes,
        "channels": 1,
        "corrections": residual.records.len(),
        "pages": res_job.pages.len(),
        "scalar_equal_cuda": true,
    }));

    let mut builder = ReceiptBuilder::new("entropy-cuda");
    builder
        .result(Verdict::Supported)
        .result_detail(format!(
            "scalar == CUDA entropy decode on {} jobs (literal delta-lane4, RAW-fallback \
             noise, procedural residual closure); all page statuses 1",
            cells.len()
        ))
        .params(CourtParams {
            universe: Some("vole.audio.u1".into()),
            profile: Some("u1/v1 + vole.entropy.p1/p1/v1".into()),
            backend: Some("cuda".into()),
            content_kind: Some("entropy-corpus-v1 cuda".into()),
            ..Default::default()
        })
        .extra("cells", serde_json::Value::Array(cells))
        .extra(
            "artifact",
            serde_json::json!({ "path": DEFAULT_PTX, "sha256": ptx.1 }),
        );
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court entropy-cuda: SUPPORTED");
    println!("  receipt: {}", path.display());
    Ok(Verdict::Supported)
}
