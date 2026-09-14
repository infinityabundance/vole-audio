//! Deterministic trainer for the voice LSF split multi-stage vector quantiser.
//!
//! ```text
//! cargo run --release --example voice_train -- \
//!     --root <dev-clean dir> --out assets/voice/lsf_msvq_v1.bin \
//!     --files 400 [--report <path>]
//! ```
//!
//! The trainer walks the LibriSpeech `dev-clean` speaker directories, **skips
//! the twelve `effectiveness` evaluation speakers as whole directories**, then
//! trains on the first `--files` remaining `.flac` paths in sorted order. Each
//! utterance is decoded to 16 kHz mono, order-16 LSF vectors are extracted with
//! the same block geometry the codec uses, and a split (two halves) two-stage
//! codebook with sizes `[256, 64]` per split is fitted by LBG. Everything is
//! deterministic: fixed-seed walk order, fixed LBG splitting, fixed Lloyd
//! iteration counts, and deterministic nearest-centroid tie-breaking.
//!
//! The resulting asset is a **frozen decoder table**: mean and every centroid
//! are stored as `i16` (Q = 32767/pi) and every runtime encode/decode uses those
//! quantised values, so the encoder and decoder agree bit-for-bit.

use std::path::{Path, PathBuf};
use std::process::Command;

use vole_audio::hash::sha256::{Sha256, hex};
use vole_audio::learned::train::lpc;
use vole_audio::voice::lsf;
use vole_audio::voice::predict::predictor_to_reflections;

/// Analysis frame length in samples (20 ms at 16 kHz).
const FRAME: usize = 320;
/// Frame hop in samples (10 ms).
const HOP: usize = 160;
/// Left context reused by the short-term analysis.
const LEFT: usize = 304;
/// LSF order.
const ORDER: usize = 16;
/// Coefficients per split.
const SPLIT: usize = 8;
/// Number of splits.
const SPLITS: usize = 2;
/// Rows per `(split, stage)`.
const STAGE_SIZES: [[usize; 2]; 2] = [[256, 64], [256, 64]];
/// Maximum number of training vectors (stop early, deterministically).
const CAP: usize = 300_000;
/// LBG split perturbation, in radians.
const EPS: f64 = 0.01;
/// Lloyd iterations per LBG level.
const LLOYD_ITERS: usize = 25;
/// Quantised-domain refinement iterations.
const REFINE_ITERS: usize = 5;

/// i16 quantisation scale for the asset (radians -> Q).
const Q_SCALE: f64 = 32767.0 / std::f64::consts::PI;

/// The `effectiveness` evaluation speakers, excluded as whole directories.
const EXCLUDED: [&str; 12] = [
    "1272", "1462", "1673", "174", "1919", "1988", "1993", "2035", "2078", "2086", "2277", "2412",
];

/// A fixed-width split vector.
type V8 = [f64; SPLIT];

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    root: PathBuf,
    out: PathBuf,
    files: usize,
    report: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut root: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut files = 400usize;
    let mut report: Option<PathBuf> = None;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--root" => root = Some(PathBuf::from(it.next().ok_or("--root needs a value")?)),
            "--out" => out = Some(PathBuf::from(it.next().ok_or("--out needs a value")?)),
            "--files" => {
                files = it
                    .next()
                    .ok_or("--files needs a value")?
                    .parse()
                    .map_err(|_| "--files must be a positive integer".to_string())?;
                if files == 0 {
                    return Err("--files must be at least 1".into());
                }
            }
            "--report" => report = Some(PathBuf::from(it.next().ok_or("--report needs a value")?)),
            other => return Err(format!("unexpected argument `{other}`")),
        }
    }
    Ok(Args {
        root: root.ok_or("--root is required")?,
        out: out.ok_or("--out is required")?,
        files,
        report,
    })
}

// ---------------------------------------------------------------------------
// Corpus walk
// ---------------------------------------------------------------------------

fn collect_flac_paths(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut all = Vec::new();
    walk(root, &mut all)?;
    all.retain(|p| !speaker_is_excluded(root, p));
    all.sort();
    Ok(all)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            walk(&path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("flac") {
            out.push(path);
        }
    }
    Ok(())
}

/// True iff the first path component under `root` is one of the excluded
/// evaluation speakers.
fn speaker_is_excluded(root: &Path, path: &Path) -> bool {
    let Ok(rel) = path.strip_prefix(root) else {
        return false;
    };
    match rel.components().next() {
        Some(c) => {
            let name = c.as_os_str().to_str().unwrap_or("");
            EXCLUDED.contains(&name)
        }
        None => false,
    }
}

// ---------------------------------------------------------------------------
// FLAC decoding
// ---------------------------------------------------------------------------

/// Decode one FLAC file to 16 kHz mono i16 via the `flac` CLI (stdout WAV).
fn decode_flac(path: &Path) -> Result<Vec<i16>, String> {
    let out = Command::new("flac")
        .arg("-d")
        .arg("-c")
        .arg("-s")
        .arg(path)
        .output()
        .map_err(|e| format!("cannot run flac: {e}"))?;
    if !out.status.success() {
        return Err(format!("flac failed to decode {}", path.display()));
    }
    let (rate, channels, pcm) = read_wav_i16(&out.stdout)?;
    if rate != 16_000 || channels != 1 {
        return Err(format!(
            "{}: expected 16 kHz mono, got {rate} Hz / {channels} channel(s)",
            path.display()
        ));
    }
    Ok(pcm)
}

/// Minimal RIFF/WAVE parser for 16-bit PCM.
fn read_wav_i16(bytes: &[u8]) -> Result<(u32, u8, Vec<i16>), String> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("flac output is not RIFF/WAVE".into());
    }
    let mut pos = 12usize;
    let mut channels = 0u8;
    let mut rate = 0u32;
    let mut bits = 0u16;
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(
            bytes[pos + 4..pos + 8]
                .try_into()
                .map_err(|_| "bad chunk length")?,
        ) as usize;
        let body = bytes
            .get(pos + 8..pos + 8 + len)
            .ok_or("truncated WAV chunk")?;
        if id == b"fmt " && len >= 16 {
            let fmt = u16::from_le_bytes(body[0..2].try_into().map_err(|_| "fmt")?);
            if fmt != 1 {
                return Err("WAV is not PCM".into());
            }
            channels = u16::from_le_bytes(body[2..4].try_into().map_err(|_| "ch")?) as u8;
            rate = u32::from_le_bytes(body[4..8].try_into().map_err(|_| "rate")?);
            bits = u16::from_le_bytes(body[14..16].try_into().map_err(|_| "bits")?);
        } else if id == b"data" {
            data = Some(body);
        }
        pos = pos + 8 + len + (len & 1);
    }
    if bits != 16 {
        return Err(format!("expected 16-bit PCM, got {bits}"));
    }
    let data = data.ok_or("WAV has no data chunk")?;
    if !data.len().is_multiple_of(2) {
        return Err("WAV data is not 16-bit aligned".into());
    }
    let mut pcm = Vec::with_capacity(data.len() / 2);
    let mut i = 0usize;
    while i + 2 <= data.len() {
        pcm.push(i16::from_le_bytes(
            data[i..i + 2].try_into().map_err(|_| "sample")?,
        ));
        i += 2;
    }
    Ok((rate, channels, pcm))
}

// ---------------------------------------------------------------------------
// LSF extraction
// ---------------------------------------------------------------------------

/// Convert a reflection vector to an order-16 LSF vector in the codec's
/// convention: negate the estimator's reflections before the conversion.
fn lsf_from_reflections(k: &[f64]) -> Option<[f64; ORDER]> {
    if k.len() != ORDER {
        return None;
    }
    let neg: Vec<f64> = k.iter().map(|v| -*v).collect();
    let l = lsf::reflections_to_lsf(&neg)?;
    let mut out = [0.0f64; ORDER];
    out.copy_from_slice(&l);
    Some(out)
}

/// Extract the three estimator variants for one analysis block into `vectors`.
fn push_variants(block: &[i32], vectors: &mut Vec<[f64; ORDER]>) {
    if vectors.len() >= CAP {
        return;
    }
    let r = lpc::autocorrelation(block, ORDER, 0.25);
    if let Some(v) = lsf_from_reflections(&lpc::levinson_reflections(&r, ORDER)) {
        vectors.push(v);
    }
    if vectors.len() >= CAP {
        return;
    }
    if let Some(v) = lsf_from_reflections(&lpc::burg_reflections(block, ORDER)) {
        vectors.push(v);
    }
    if vectors.len() >= CAP {
        return;
    }
    if let Some(c) = lpc::lsq_coefficients(block, ORDER)
        && let Some(k) = predictor_to_reflections(&c)
        && let Some(v) = lsf_from_reflections(&k)
    {
        vectors.push(v);
    }
}

fn extract_vectors(files: &[PathBuf]) -> Result<Vec<[f64; ORDER]>, String> {
    let mut vectors: Vec<[f64; ORDER]> = Vec::new();
    'files: for (i, path) in files.iter().enumerate() {
        let pcm = decode_flac(path)?;
        let mut t = 0usize;
        while t + FRAME <= pcm.len() {
            let lo = t.saturating_sub(LEFT);
            let hi = t + FRAME;
            let block: Vec<i32> = pcm[lo..hi].iter().map(|&s| i32::from(s)).collect();
            push_variants(&block, &mut vectors);
            if vectors.len() >= CAP {
                break 'files;
            }
            t += HOP;
        }
        if (i + 1) % 50 == 0 {
            eprintln!(
                "  decoded {}/{} files, {} vectors",
                i + 1,
                files.len(),
                vectors.len()
            );
        }
    }
    Ok(vectors)
}

// ---------------------------------------------------------------------------
// Vector-quantiser fitting
// ---------------------------------------------------------------------------

fn mean8(data: &[V8]) -> V8 {
    let mut s = [0.0f64; SPLIT];
    for v in data {
        for (slot, x) in s.iter_mut().zip(v.iter()) {
            *slot += x;
        }
    }
    let n = data.len().max(1) as f64;
    for slot in &mut s {
        *slot /= n;
    }
    s
}

/// Index of the nearest centroid; ties go to the lowest index.
fn nearest(centroids: &[V8], v: &V8) -> usize {
    let mut best = 0usize;
    let mut best_d = f64::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let mut e = 0.0f64;
        for (a, b) in v.iter().zip(c.iter()) {
            let d = a - b;
            e += d * d;
        }
        if e < best_d {
            best_d = e;
            best = i;
        }
    }
    best
}

fn sub8(a: &V8, b: &V8) -> V8 {
    let mut out = [0.0f64; SPLIT];
    for (slot, (x, y)) in out.iter_mut().zip(a.iter().zip(b.iter())) {
        *slot = x - y;
    }
    out
}

fn add_into(sum: &mut V8, v: &V8) {
    for (slot, x) in sum.iter_mut().zip(v.iter()) {
        *slot += x;
    }
}

fn scaled(sum: &V8, count: usize) -> V8 {
    let inv = 1.0 / count.max(1) as f64;
    let mut out = [0.0f64; SPLIT];
    for (slot, x) in out.iter_mut().zip(sum.iter()) {
        *slot = x * inv;
    }
    out
}

/// One Lloyd assignment/update pass set over float centroids.
fn lloyd(data: &[V8], centroids: &mut [V8], iters: usize) {
    let k = centroids.len();
    for _ in 0..iters {
        let mut sums = vec![[0.0f64; SPLIT]; k];
        let mut counts = vec![0usize; k];
        for v in data {
            let a = nearest(centroids, v);
            add_into(&mut sums[a], v);
            counts[a] += 1;
        }
        for (i, cent) in centroids.iter_mut().enumerate() {
            if counts[i] > 0 {
                *cent = scaled(&sums[i], counts[i]);
            }
        }
    }
}

/// LBG: start from the mean, repeatedly split each centroid by `+-eps`, then
/// refine. Sizes are powers of two, so the doubling never over/under-shoots.
fn lbg(data: &[V8], size: usize, eps: f64) -> Vec<V8> {
    let mut centroids = vec![mean8(data)];
    while centroids.len() < size {
        let mut next = Vec::with_capacity(centroids.len() * 2);
        for c in &centroids {
            let mut a = *c;
            let mut b = *c;
            for slot in &mut a {
                *slot -= eps;
            }
            for slot in &mut b {
                *slot += eps;
            }
            next.push(a);
            next.push(b);
        }
        next.truncate(size);
        centroids = next;
        lloyd(data, &mut centroids, LLOYD_ITERS);
    }
    centroids
}

fn quantise(v: &V8) -> [i16; SPLIT] {
    let mut q = [0i16; SPLIT];
    for (slot, x) in q.iter_mut().zip(v.iter()) {
        *slot = (x * Q_SCALE).round().clamp(-32768.0, 32767.0) as i16;
    }
    q
}

fn dequantise(q: &[i16; SPLIT]) -> V8 {
    let mut out = [0.0f64; SPLIT];
    for (slot, &x) in out.iter_mut().zip(q.iter()) {
        *slot = f64::from(x) / Q_SCALE;
    }
    out
}

/// A fitted, quantised split codebook.
struct SplitCodebook {
    stage0: Vec<[i16; SPLIT]>,
    stage1: Vec<[i16; SPLIT]>,
}

/// One quantised-domain Lloyd step: assign against the dequantised table and
/// re-quantise the mean of each cell (an empty cell keeps its previous row).
fn refine_quantised(data: &[V8], rows: &mut [[i16; SPLIT]], offsets: &[V8], iters: usize) {
    let k = rows.len();
    for _ in 0..iters {
        let table: Vec<V8> = rows.iter().map(dequantise).collect();
        let mut sums = vec![[0.0f64; SPLIT]; k];
        let mut counts = vec![0usize; k];
        for d in data {
            // Assign against the sum of quantised centroids: for stage 1,
            // `offsets` holds the dequantised stage-0 row that `d` follows.
            let residual = if offsets.is_empty() {
                *d
            } else {
                let nearest_offset = nearest(offsets, d);
                sub8(d, &offsets[nearest_offset])
            };
            let a = nearest(&table, &residual);
            add_into(&mut sums[a], &residual);
            counts[a] += 1;
        }
        for i in 0..k {
            if counts[i] > 0 {
                rows[i] = quantise(&scaled(&sums[i], counts[i]));
            }
        }
    }
}

/// Fit one split of `data` (already mean-removed) and refine in the quantised
/// domain so the frozen table is self-consistent with the runtime search.
fn fit_split(data: &[V8]) -> SplitCodebook {
    let size0 = STAGE_SIZES[0][0];
    let size1 = STAGE_SIZES[0][1];

    let c0 = lbg(data, size0, EPS);
    let resid: Vec<V8> = data
        .iter()
        .map(|d| {
            let a = nearest(&c0, d);
            sub8(d, &c0[a])
        })
        .collect();
    let c1 = lbg(&resid, size1, EPS);

    // Quantised-domain refinement of stage 0.
    let mut stage0: Vec<[i16; SPLIT]> = c0.iter().map(quantise).collect();
    refine_quantised(data, &mut stage0, &[], REFINE_ITERS);

    // Quantised-domain refinement of stage 1 against stage-0 + stage-1.
    let stage0_rad: Vec<V8> = stage0.iter().map(dequantise).collect();
    let mut stage1: Vec<[i16; SPLIT]> = c1.iter().map(quantise).collect();
    refine_quantised(data, &mut stage1, &stage0_rad, REFINE_ITERS);

    SplitCodebook { stage0, stage1 }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    if let Err(e) = run() {
        eprintln!("voice_train: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    let all = collect_flac_paths(&args.root)?;
    if all.is_empty() {
        return Err(format!(
            "no non-excluded .flac files found under {}",
            args.root.display()
        ));
    }
    let files: Vec<PathBuf> = all.into_iter().take(args.files).collect();
    println!(
        "training files: {} (speakers excluded as whole directories: {})",
        files.len(),
        EXCLUDED.join(", ")
    );

    let vectors = extract_vectors(&files)?;
    if vectors.is_empty() {
        return Err("no training vectors were extracted".into());
    }
    println!("training vectors: {}", vectors.len());

    // Global mean, and its frozen quantised table entry.
    let mut mean = [0.0f64; ORDER];
    for v in &vectors {
        for (slot, x) in mean.iter_mut().zip(v.iter()) {
            *slot += x;
        }
    }
    let n = vectors.len() as f64;
    for slot in &mut mean {
        *slot /= n;
    }
    let mut mean_q = [0i16; ORDER];
    for (slot, x) in mean_q.iter_mut().zip(mean.iter()) {
        *slot = (x * Q_SCALE).round().clamp(-32768.0, 32767.0) as i16;
    }
    let mut mean_rad = [0.0f64; ORDER];
    for (slot, &x) in mean_rad.iter_mut().zip(mean_q.iter()) {
        *slot = f64::from(x) / Q_SCALE;
    }

    // Fit each split on mean-removed data, using the frozen quantised mean so
    // the training target matches the runtime reconstruction.
    let mut books: Vec<SplitCodebook> = Vec::with_capacity(SPLITS);
    for s in 0..SPLITS {
        let m = half(&mean_rad, s);
        let data: Vec<V8> = vectors.iter().map(|v| sub8(&half(v, s), &m)).collect();
        books.push(fit_split(&data));
    }

    // Final training distortion against the frozen quantised table, one pass
    // with the same joint search the runtime encoder uses.
    let (mse, mae) = training_error(&vectors, &mean_rad, &books);

    // Serialise the asset.
    let payload = serialise(&mean_q, &books);
    let digest = Sha256::digest(&payload);
    let sha_hex = hex(&digest);
    let payload_len = payload.len();
    let mut asset = payload;
    asset.extend_from_slice(&digest);

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    std::fs::write(&args.out, &asset).map_err(|e| format!("write {}: {e}", args.out.display()))?;

    println!("stage sizes per split: [256, 64] x {}", SPLITS);
    println!(
        "final training LSF MSE: {mse:.10} rad^2 (mean abs {mae:.10} rad over {} coeffs)",
        vectors.len() * ORDER
    );
    println!(
        "payload {payload_len} bytes, asset {} bytes, payload SHA256 {sha_hex}",
        asset.len()
    );
    println!("wrote {}", args.out.display());

    if let Some(report) = &args.report {
        let text = format!(
            "voice LSF MSVQ trainer\n\
             training files: {}\n\
             training vectors: {}\n\
             excluded speakers: {}\n\
             stage sizes per split: [256, 64] x {}\n\
             final training LSF MSE: {mse:.10} rad^2\n\
             final training LSF MAE: {mae:.10} rad\n\
             payload bytes: {payload_len}\n\
             asset bytes: {}\n\
             payload SHA256: {sha_hex}\n",
            files.len(),
            vectors.len(),
            EXCLUDED.join(", "),
            SPLITS,
            asset.len(),
        );
        std::fs::write(report, text).map_err(|e| format!("write {}: {e}", report.display()))?;
        println!("wrote report {}", report.display());
    }

    Ok(())
}

/// One split half of an order-16 vector.
fn half(v: &[f64; ORDER], s: usize) -> V8 {
    let mut h = [0.0f64; SPLIT];
    h.copy_from_slice(&v[s * SPLIT..s * SPLIT + SPLIT]);
    h
}

/// Reconstruct with the frozen table (mean + stage0 + stage1 per split).
fn reconstruct(
    mean_rad: &[f64; ORDER],
    books: &[SplitCodebook],
    pick: &[[usize; 2]],
) -> [f64; ORDER] {
    let mut out = [0.0f64; ORDER];
    for s in 0..SPLITS {
        let c0 = dequantise(&books[s].stage0[pick[s][0]]);
        let c1 = dequantise(&books[s].stage1[pick[s][1]]);
        for j in 0..SPLIT {
            out[s * SPLIT + j] = mean_rad[s * SPLIT + j] + c0[j] + c1[j];
        }
    }
    out
}

/// Best sub-index pair per split for the quantised table (the same joint search
/// as `vq::encode_lsf`), given pre-dequantised split tables.
fn best_pick(
    target: &[f64; ORDER],
    mean_rad: &[f64; ORDER],
    tables0: &[Vec<V8>],
    tables1: &[Vec<V8>],
) -> [[usize; 2]; SPLITS] {
    let mut pick = [[0usize; 2]; SPLITS];
    for s in 0..SPLITS {
        let mut t = [0.0f64; SPLIT];
        for j in 0..SPLIT {
            t[j] = target[s * SPLIT + j] - mean_rad[s * SPLIT + j];
        }
        let mut best = (f64::INFINITY, 0usize, 0usize);
        for (i0, c0) in tables0[s].iter().enumerate() {
            let d = sub8(&t, c0);
            for (i1, c1) in tables1[s].iter().enumerate() {
                let mut e = 0.0f64;
                for (a, b) in d.iter().zip(c1.iter()) {
                    let x = a - b;
                    e += x * x;
                }
                if e < best.0 {
                    best = (e, i0, i1);
                }
            }
        }
        pick[s] = [best.1, best.2];
    }
    pick
}

fn training_error(
    vectors: &[[f64; ORDER]],
    mean_rad: &[f64; ORDER],
    books: &[SplitCodebook],
) -> (f64, f64) {
    let tables0: Vec<Vec<V8>> = books
        .iter()
        .map(|b| b.stage0.iter().map(dequantise).collect())
        .collect();
    let tables1: Vec<Vec<V8>> = books
        .iter()
        .map(|b| b.stage1.iter().map(dequantise).collect())
        .collect();
    let mut sse = 0.0f64;
    let mut sae = 0.0f64;
    for v in vectors {
        let pick = best_pick(v, mean_rad, &tables0, &tables1);
        let recon = reconstruct(mean_rad, books, &pick);
        for (a, b) in v.iter().zip(recon.iter()) {
            let d = a - b;
            sse += d * d;
            sae += d.abs();
        }
    }
    let coeffs = vectors.len() as f64 * ORDER as f64;
    (sse / coeffs, sae / coeffs)
}

/// Serialise the normative asset payload (everything before the trailing SHA).
fn serialise(mean_q: &[i16; ORDER], books: &[SplitCodebook]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"VOLEVQ01");
    out.extend_from_slice(&(ORDER as u16).to_le_bytes());
    out.extend_from_slice(&(SPLITS as u16).to_le_bytes());
    out.push(2u8);
    out.push(0u8);
    for row in &STAGE_SIZES {
        for &size in row {
            out.extend_from_slice(&(size as u32).to_le_bytes());
        }
    }
    for q in mean_q {
        out.extend_from_slice(&q.to_le_bytes());
    }
    for book in books {
        for q in &book.stage0 {
            for x in q {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        for q in &book.stage1 {
            for x in q {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    out
}
