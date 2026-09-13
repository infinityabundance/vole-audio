//! `court learned-lossy` — Phase 7B: `vole.audio.lossy.exp1` vs Opus and Lyra.
//!
//! The court measures VOLE's lossy profile against **external** competitors at
//! **matched actual bitrate**, which is the Phase-7B gate:
//!
//! * VOLE (`vole.audio.lossy.exp1`) — in process, its real emitted bytes;
//! * Opus (`opusenc`/`opusdec`) — external subprocess, a real VBR rate ladder;
//! * Lyra (`encoder_main`/`decoder_main`, 16 kHz mono) — external subprocess;
//! * ViSQOL — external perceptual metric (MOS-LQO) where the pinned binary runs.
//!
//! Because the competitors' encoders overshoot or undershoot their requested
//! bitrate, comparing "VOLE at 24 kbps" with "Opus asked for 24 kbps" would be
//! comparing unequal cells. Instead each competitor's measured
//! (actual bitrate, quality) curve is **interpolated at VOLE's actual bitrate**
//! and the difference is reported. Cells outside the competitor's measured
//! range are `null`, never extrapolated.
//!
//! ViSQOL in speech mode is a *speech* metric, so it is applied to the speech
//! clips and reported for them; the synthetic waveform and control fixtures are
//! compared by delay-aligned SNR, spectral distortion and worst-block error.
//! (Running a speech-intelligibility metric over full-scale noise controls would
//! be a category error dressed up as evidence.)
//!
//! No competitor code is imported, linked or wrapped; they are separate
//! processes and their real emitted sizes are measured. Only VOLE-derived
//! integers and text enter the frozen projection, so the frozen hash is stable
//! across host tool versions.
//!
//! Outputs are delay-aligned by a bounded cross-correlation (`ALIGN_LAG`
//! samples, wide enough for every codec's algorithmic delay and narrow enough
//! that it cannot silently realign different content) before distortion is
//! measured, so a codec's delay is not counted as distortion.

use crate::courts::learned_common as common;
use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::lossy::{LossyCodec, LossyConfig};
use crate::status::Verdict;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Frozen static-result hash over the VOLE-derived projection.
pub const LEARNED_LOSSY_SHA256: &str =
    "426a6fab1e2b76eb8444e3ffed252b2203d2b9659be1df91b157d45ab116650f";

const VOLE_RATES: [u32; 8] = [
    8_000, 12_000, 16_000, 24_000, 32_000, 48_000, 64_000, 96_000,
];
const OPUS_RATES: [u32; 10] = [6, 8, 12, 16, 24, 32, 48, 64, 96, 128];
const LYRA_RATES: [u32; 3] = [3_200, 6_000, 9_200];
const MAX_FRAMES: usize = 16_000;
/// Alignment search bound in samples.
const ALIGN_LAG: i64 = 512;

fn research_root() -> PathBuf {
    PathBuf::from("research")
}

fn visqol_model() -> PathBuf {
    research_root().join(
        "visqol-master/model/lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.\
         005_ep2400_train1_7_raw.tflite",
    )
}

fn tool_available(path: &Path) -> bool {
    path.is_file()
}

/// Run an external tool with its chatter suppressed (it is a measuring stick,
/// not part of the measurement).
fn quiet(cmd: &mut Command) -> &mut Command {
    cmd.stdout(Stdio::null()).stderr(Stdio::null())
}

struct Tools {
    opusenc: Option<PathBuf>,
    opusdec: Option<PathBuf>,
    lyra_enc: Option<PathBuf>,
    lyra_dec: Option<PathBuf>,
    lyra_model: PathBuf,
    visqol: Option<PathBuf>,
    visqol_model: PathBuf,
}

impl Tools {
    fn discover() -> Tools {
        let which = |name: &str| -> Option<PathBuf> {
            let p = PathBuf::from(format!("/usr/bin/{name}"));
            if p.is_file() { Some(p) } else { None }
        };
        let r = research_root();
        let lyra_enc = r.join("lyra-main/bazel-bin/lyra/cli_example/encoder_main");
        let lyra_dec = r.join("lyra-main/bazel-bin/lyra/cli_example/decoder_main");
        let visqol = r.join("visqol-master/bazel-bin/visqol");
        Tools {
            opusenc: which("opusenc"),
            opusdec: which("opusdec"),
            lyra_enc: tool_available(&lyra_enc).then_some(lyra_enc),
            lyra_dec: tool_available(&lyra_dec).then_some(lyra_dec),
            lyra_model: r.join("lyra-main/lyra/model_coeffs"),
            visqol: tool_available(&visqol).then_some(visqol),
            visqol_model: visqol_model(),
        }
    }
}

// ---------------------------------------------------------------------------
// Minimal 16-bit WAV I/O (no dependency; competitors read/write plain PCM).
// ---------------------------------------------------------------------------

fn write_wav(path: &Path, samples: &[i16], channels: u16, rate: u32) -> Result<()> {
    let data_len = samples.len() * 2;
    let byte_rate = rate * u32::from(channels) * 2;
    let block_align = channels * 2;
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, out).map_err(Error::io)
}

/// Read the first `fmt ` and `data` chunks of a PCM WAV.
fn read_wav(path: &Path) -> Result<(Vec<i16>, u16, u32)> {
    let b = std::fs::read(path).map_err(Error::io)?;
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err(Error::malformed("not a RIFF/WAVE file"));
    }
    let mut pos = 12usize;
    let mut channels = 0u16;
    let mut rate = 0u32;
    let mut bits = 0u16;
    let mut data: Option<(usize, usize)> = None;
    while pos + 8 <= b.len() {
        let id = &b[pos..pos + 4];
        let len = u32::from_le_bytes(b[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = pos + 8;
        if id == b"fmt " && body + 16 <= b.len() {
            channels = u16::from_le_bytes(b[body + 2..body + 4].try_into().unwrap());
            rate = u32::from_le_bytes(b[body + 4..body + 8].try_into().unwrap());
            bits = u16::from_le_bytes(b[body + 14..body + 16].try_into().unwrap());
        } else if id == b"data" {
            data = Some((body, (body + len).min(b.len())));
        }
        pos = body + len + (len & 1);
    }
    let (start, end) = data.ok_or_else(|| Error::malformed("WAV has no data chunk"))?;
    if bits != 16 {
        return Err(Error::new(
            crate::error::Kind::Unsupported,
            "only 16-bit PCM WAV is read",
        ));
    }
    let mut samples = Vec::with_capacity((end - start) / 2);
    for c in b[start..end].as_chunks::<2>().0 {
        samples.push(i16::from_le_bytes([c[0], c[1]]));
    }
    Ok((samples, channels, rate))
}

// ---------------------------------------------------------------------------
// Alignment and distortion
// ---------------------------------------------------------------------------

/// Best integer lag of `test` relative to `reference`, searched over a bounded
/// window around the origin, maximising normalised cross-correlation.
fn best_lag(reference: &[i32], test: &[i32], max_lag: i64) -> i64 {
    let n = reference.len().min(test.len());
    if n == 0 {
        return 0;
    }
    let mut best_lag = 0i64;
    let mut best = f64::NEG_INFINITY;
    for lag in -max_lag..=max_lag {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        let start = (n / 4) as i64;
        let end = (3 * n / 4) as i64;
        let mut t = start;
        while t < end {
            let s = t + lag;
            if t >= 0 && (t as usize) < reference.len() && s >= 0 && (s as usize) < test.len() {
                let rv = f64::from(reference[t as usize]);
                let sv = f64::from(test[s as usize]);
                num += rv * sv;
                den += sv * sv;
            }
            t += 1;
        }
        let score = if den > 0.0 { num / den.sqrt() } else { 0.0 };
        if score > best {
            best = score;
            best_lag = lag;
        }
    }
    best_lag
}

/// Delay-aligned reference/test pair.
fn aligned_pair(reference: &[i32], test: &[i32], lag: i64) -> (Vec<i32>, Vec<i32>) {
    let n = reference.len().min(test.len());
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for (i, &r) in reference.iter().enumerate().take(n) {
        let s = i as i64 + lag;
        if s >= 0 && (s as usize) < test.len() {
            a.push(r);
            b.push(test[s as usize]);
        }
    }
    (a, b)
}

fn snr_db(reference: &[i32], test: &[i32]) -> f64 {
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for (&r, &t) in reference.iter().zip(test.iter()) {
        sig += f64::from(r) * f64::from(r);
        let d = f64::from(r) - f64::from(t);
        err += d * d;
    }
    if err <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (sig / err).log10()
}

/// `Some(x)` when `x` is finite. Non-finite metrics become `null` in the
/// receipt rather than an invalid JSON number, and are never invented.
fn finite(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}

/// Spectral (log-magnitude) distortion via a naive DFT on a bounded frame.
fn spectral_distortion_db(reference: &[i32], test: &[i32]) -> f64 {
    let n = reference.len().min(test.len()).min(2048);
    if n < 32 {
        return 0.0;
    }
    let step = (reference.len() / n).max(1);
    let mut acc = 0.0f64;
    let mut count = 0.0f64;
    for k in 1..n / 2 {
        let w = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
        let (mut rr, mut ri, mut tr, mut ti) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut i = 0usize;
        let mut t = 0usize;
        while i < n && t < test.len() {
            let a = w * i as f64;
            rr += f64::from(reference[t]) * a.cos();
            ri += f64::from(reference[t]) * a.sin();
            tr += f64::from(test[t]) * a.cos();
            ti += f64::from(test[t]) * a.sin();
            i += 1;
            t += step;
        }
        let rm = (rr * rr + ri * ri).sqrt().max(1.0);
        let tm = (tr * tr + ti * ti).sqrt().max(1.0);
        acc += 20.0 * (rm / tm).log10().abs();
        count += 1.0;
    }
    if count > 0.0 { acc / count } else { 0.0 }
}

/// Worst per-block error (a proxy for transient damage), in dB relative to the
/// reference block energy.
fn transient_error_db(reference: &[i32], test: &[i32]) -> f64 {
    let block = 512usize;
    let n = reference.len().min(test.len());
    let mut worst = f64::NEG_INFINITY;
    let mut off = 0usize;
    while off + block <= n {
        let (mut sig, mut err) = (0.0f64, 0.0f64);
        for i in off..off + block {
            sig += f64::from(reference[i]).powi(2);
            let d = f64::from(reference[i]) - f64::from(test[i]);
            err += d * d;
        }
        if sig > 0.0 {
            worst = worst.max(10.0 * (sig / err.max(1e-9)).log10());
        }
        off += block;
    }
    if worst.is_finite() { worst } else { 0.0 }
}

fn visqol_mos(tools: &Tools, reference: &Path, degraded: &Path, speech: bool) -> Option<f64> {
    let visqol = tools.visqol.as_ref()?;
    let csv = degraded.with_extension("visqol.csv");
    // ViSQOL appends to the results CSV, so a stale row would be read back as a
    // fresh measurement. Always start from an empty file.
    let _ = std::fs::remove_file(&csv);
    let mut cmd = Command::new(visqol);
    cmd.arg(format!("--reference_file={}", reference.display()))
        .arg(format!("--degraded_file={}", degraded.display()))
        .arg(format!(
            "--similarity_to_quality_model={}",
            tools.visqol_model.display()
        ))
        .arg(format!("--results_csv={}", csv.display()));
    if speech {
        cmd.arg("--use_speech_mode=true");
    }
    let out = quiet(&mut cmd).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = std::fs::read_to_string(&csv).ok()?;
    let mut rows: Vec<&str> = text.lines().collect();
    if rows.len() < 2 {
        return None;
    }
    let header = rows.remove(0);
    let col = header
        .split(',')
        .position(|c| c.trim().eq_ignore_ascii_case("moslqo"))?;
    rows.last()?.split(',').nth(col)?.trim().parse::<f64>().ok()
}

/// One competitor operating point.
#[derive(Clone)]
struct Point {
    bps: f64,
    snr: f64,
    visqol: Option<f64>,
}

/// Linear interpolation of a quality field at a target bitrate. `None` outside
/// the measured range, so unequal cells are never pretended to be equal.
fn interp(points: &[Point], target: f64, field: fn(&Point) -> Option<f64>) -> Option<f64> {
    let mut sorted: Vec<(f64, f64)> = points
        .iter()
        .filter_map(|p| field(p).map(|v| (p.bps, v)))
        .collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    if sorted.is_empty() || target < sorted[0].0 || target > sorted[sorted.len() - 1].0 {
        return None;
    }
    if sorted.len() == 1 {
        return Some(sorted[0].1);
    }
    for w in sorted.windows(2) {
        let (a, b) = (w[0], w[1]);
        if target >= a.0 && target <= b.0 {
            let t = if (b.0 - a.0).abs() < 1e-9 {
                0.0
            } else {
                (target - a.0) / (b.0 - a.0)
            };
            return Some(a.1 + t * (b.1 - a.1));
        }
    }
    None
}

fn opus_points(
    tools: &Tools,
    wav: &Path,
    work: &Path,
    seconds: f64,
    reference: &[i32],
    with_visqol: bool,
) -> Vec<Point> {
    let (Some(enc), Some(dec)) = (tools.opusenc.as_ref(), tools.opusdec.as_ref()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for &kbps in &OPUS_RATES {
        let opus = work.join("opus.opus");
        let dec_wav = work.join("opus_dec.wav");
        let _ = std::fs::remove_file(&opus);
        let ok = quiet(
            Command::new(enc)
                .arg("--quiet")
                .arg(format!("--bitrate={kbps}"))
                .arg("--vbr")
                .arg(wav)
                .arg(&opus),
        )
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
        if !ok {
            continue;
        }
        let Some(bytes) = std::fs::metadata(&opus).ok().map(|m| m.len()) else {
            continue;
        };
        let ok = quiet(Command::new(dec).arg("--quiet").arg(&opus).arg(&dec_wav))
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            continue;
        }
        let Ok((samples, och, rate)) = read_wav(&dec_wav) else {
            continue;
        };
        let mono: Vec<i32> = samples
            .iter()
            .step_by(usize::from(och).max(1))
            .map(|&v| i32::from(v))
            .collect();
        let lag = best_lag(reference, &mono, ALIGN_LAG);
        let (a, b) = aligned_pair(reference, &mono, lag);
        let snr = snr_db(&a, &b);
        let visqol = if with_visqol {
            let rw = work.join("opus_ref.wav");
            let dw = work.join("opus_aligned.wav");
            let _ = write_wav(&rw, &to_i16(&a), 1, rate);
            let _ = write_wav(&dw, &to_i16(&b), 1, rate);
            visqol_mos(tools, &rw, &dw, true)
        } else {
            None
        };
        out.push(Point {
            bps: bytes as f64 * 8.0 / seconds,
            snr,
            visqol,
        });
    }
    out
}

fn lyra_points(
    tools: &Tools,
    wav: &Path,
    work: &Path,
    seconds: f64,
    reference: &[i32],
) -> Vec<Point> {
    let (Some(enc), Some(dec)) = (tools.lyra_enc.as_ref(), tools.lyra_dec.as_ref()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for &bps in &LYRA_RATES {
        let dir = work.join("lyra");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).map_err(Error::io).ok();
        let model = tools.lyra_model.to_string_lossy().to_string();
        let ok = quiet(
            Command::new(enc)
                .arg(format!("--input_path={}", wav.display()))
                .arg(format!("--output_dir={}", dir.display()))
                .arg(format!("--bitrate={bps}"))
                .arg(format!("--model_path={model}")),
        )
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
        if !ok {
            continue;
        }
        let Some(encoded) = std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.extension().map(|x| x == "lyra").unwrap_or(false))
        else {
            continue;
        };
        let Some(bytes) = std::fs::metadata(&encoded).ok().map(|m| m.len()) else {
            continue;
        };
        let ok = quiet(
            Command::new(dec)
                .arg(format!("--encoded_path={}", encoded.display()))
                .arg(format!("--output_dir={}", dir.display()))
                .arg(format!("--bitrate={bps}"))
                .arg(format!("--model_path={model}")),
        )
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
        if !ok {
            continue;
        }
        let Some(dec_wav) = std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| {
                p.file_name()
                    .map(|n| n.to_string_lossy().ends_with("_decoded.wav"))
                    .unwrap_or(false)
            })
        else {
            continue;
        };
        let Ok((samples, _ch, _rate)) = read_wav(&dec_wav) else {
            continue;
        };
        let mono: Vec<i32> = samples.iter().map(|&v| i32::from(v)).collect();
        let lag = best_lag(reference, &mono, ALIGN_LAG);
        let (a, b) = aligned_pair(reference, &mono, lag);
        out.push(Point {
            bps: bytes as f64 * 8.0 / seconds,
            snr: snr_db(&a, &b),
            visqol: None,
        });
    }
    out
}

/// One corpus case.
struct Case {
    id: String,
    kind: &'static str,
    rate: u32,
    channels: u16,
    samples: Vec<i32>,
}

fn cases() -> Result<Vec<Case>> {
    let mut out = Vec::new();
    for f in crate::entropy::corpus::all() {
        let frames = f.frames().min(MAX_FRAMES);
        let ch = usize::from(f.channels);
        // The lossy court is driven from the canonical 16-bit range, matching
        // what the external competitors can actually carry.
        let samples: Vec<i32> = f.samples[..frames * ch]
            .iter()
            .map(|&v| v.clamp(-32_768, 32_767))
            .collect();
        out.push(Case {
            id: format!("corpus-{}", f.name),
            kind: f.kind,
            rate: 16_000,
            channels: u16::from(f.channels),
            samples,
        });
    }
    if crate::learned::corpus_real::available() {
        let clips = crate::learned::corpus_real::effectiveness_clips();
        let loaded = crate::learned::corpus_real::load_cases(
            &clips,
            3,
            &PathBuf::from("target/real-corpus/scratch"),
        )?;
        for c in loaded {
            let ch = usize::from(c.clip.channels);
            let frames = (c.samples.len() / ch).min(MAX_FRAMES);
            out.push(Case {
                id: format!("speech-{}", c.clip.id),
                kind: "speech",
                rate: c.clip.sample_rate_hz,
                channels: c.clip.channels.into(),
                samples: c.samples[..frames * ch].to_vec(),
            });
        }
    }
    Ok(out)
}

fn to_i16(samples: &[i32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&v| v.clamp(-32_768, 32_767) as i16)
        .collect()
}

/// Extract channel `c` of an interleaved buffer.
fn channel_of<T: Copy>(samples: &[T], ch: usize, c: usize) -> Vec<T> {
    if ch <= 1 {
        samples.to_vec()
    } else {
        samples.iter().skip(c).step_by(ch).copied().collect()
    }
}

/// Run the court; writes `receipts/learned-lossy/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.lossy.court.v2");
    common::push_label(&mut projection, crate::lossy::LOSSY_PROFILE);

    let tools = Tools::discover();
    let work = PathBuf::from("target/lossy-court");
    std::fs::create_dir_all(&work).map_err(Error::io)?;

    let corpus = cases()?;
    let mut rows = Vec::new();
    let mut all_ok = true;
    let mut opus_deltas: Vec<f64> = Vec::new();
    let mut lyra_deltas: Vec<f64> = Vec::new();
    let mut visqol_deltas: Vec<f64> = Vec::new();

    for case in &corpus {
        let ch = usize::from(case.channels);
        let ref_wav = work.join(format!("ref-{}.wav", case.id));
        write_wav(&ref_wav, &to_i16(&case.samples), case.channels, case.rate)?;
        let seconds = case.samples.len() as f64 / f64::from(case.rate) / ch as f64;

        // ViSQOL speech mode is a speech metric; apply it where it is valid.
        let use_visqol = tools.visqol.is_some() && case.kind == "speech";
        let mut opus = opus_points(&tools, &ref_wav, &work, seconds, &case.samples, use_visqol);
        opus.sort_by(|a, b| a.bps.partial_cmp(&b.bps).unwrap());
        let lyra = if case.rate == 16_000 && case.channels == 1 {
            lyra_points(&tools, &ref_wav, &work, seconds, &case.samples)
        } else {
            Vec::new()
        };

        let mut vole_cells = Vec::new();
        for &bps in &VOLE_RATES {
            let config = LossyConfig {
                sample_rate_hz: case.rate,
                channels: case.channels as u8,
                frame_len: LossyConfig::default_frame_len(case.rate),
                target_bits_per_second: bps,
            };
            let codec = LossyCodec::new(config)?;
            let sw = Stopwatch::start();
            let bytes = codec.encode(&case.samples)?;
            let encode_ns = sw.elapsed_ns().max(0) as u64;
            let sw = Stopwatch::start();
            let decoded = codec.decode(&bytes)?;
            let decode_ns = sw.elapsed_ns().max(0) as u64;
            let actual_bps = bytes.len() as f64 * 8.0 / seconds;

            // Per-channel metrics, averaged over channels.
            let mut snr = 0.0f64;
            let mut spec = 0.0f64;
            let mut trans = 0.0f64;
            let mut mos_sum = 0.0f64;
            let mut mos_n = 0u32;
            for c in 0..ch {
                let r = channel_of(&case.samples, ch, c);
                let d = channel_of(&decoded, ch, c);
                let lag = best_lag(&r, &d, ALIGN_LAG);
                let (a, b) = aligned_pair(&r, &d, lag);
                snr += snr_db(&a, &b);
                spec += spectral_distortion_db(&a, &b);
                trans += transient_error_db(&a, &b);
                if use_visqol {
                    let rw = work.join("vole_ref.wav");
                    let dw = work.join("vole_aligned.wav");
                    let _ = write_wav(&rw, &to_i16(&a), 1, case.rate);
                    let _ = write_wav(&dw, &to_i16(&b), 1, case.rate);
                    if let Some(v) = visqol_mos(&tools, &rw, &dw, case.kind == "speech") {
                        mos_sum += v;
                        mos_n += 1;
                    }
                }
            }
            let n = ch as f64;
            let snr = snr / n;
            let spec = spec / n;
            let trans = trans / n;
            let mos = if mos_n > 0 {
                Some(mos_sum / f64::from(mos_n))
            } else {
                None
            };

            let opus_snr_at = interp(&opus, actual_bps, |p| Some(p.snr)).and_then(finite);
            let lyra_snr_at = interp(&lyra, actual_bps, |p| Some(p.snr)).and_then(finite);
            let opus_mos_at = interp(&opus, actual_bps, |p| p.visqol).and_then(finite);
            let d_snr_opus = match (finite(snr), opus_snr_at) {
                (Some(v), Some(o)) => Some(v - o),
                _ => None,
            };
            let d_snr_lyra = match (finite(snr), lyra_snr_at) {
                (Some(v), Some(l)) => Some(v - l),
                _ => None,
            };
            let d_mos_opus = match (mos, opus_mos_at) {
                (Some(v), Some(o)) => Some(v - o),
                _ => None,
            };
            if let Some(d) = d_snr_opus {
                opus_deltas.push(d);
            }
            if let Some(d) = d_snr_lyra {
                lyra_deltas.push(d);
            }
            if let Some(d) = d_mos_opus {
                visqol_deltas.push(d);
            }

            if bytes.is_empty() || snr.is_nan() {
                all_ok = false;
            }
            common::push_label(&mut projection, &case.id);
            common::push_u64(&mut projection, u64::from(bps));
            common::push_u64(&mut projection, bytes.len() as u64);
            // An exact reconstruction has infinite SNR; freeze a finite cap.
            common::push_u64(&mut projection, (snr.clamp(0.0, 200.0) * 100.0) as u64);

            vole_cells.push(serde_json::json!({
                "target_bps": bps,
                "bytes": bytes.len(),
                "actual_bps": actual_bps,
                "snr_db": finite(snr),
                "spectral_db": finite(spec),
                "transient_db": finite(trans),
                "visqol_mos": mos.and_then(finite),
                "opus_snr_at_actual": opus_snr_at,
                "opus_visqol_at_actual": opus_mos_at,
                "lyra_snr_at_actual": lyra_snr_at,
                "delta_snr_vs_opus": d_snr_opus,
                "delta_snr_vs_lyra": d_snr_lyra,
                "delta_visqol_vs_opus": d_mos_opus,
                "encode_ns": encode_ns,
                "decode_ns": decode_ns,
            }));
        }

        rows.push(serde_json::json!({
            "id": case.id,
            "kind": case.kind,
            "rate": case.rate,
            "channels": case.channels,
            "frames": case.samples.len() / ch,
            "vole": vole_cells,
            "opus": opus.iter().map(|p| serde_json::json!({
                "actual_bps": p.bps,
                "snr_db": p.snr,
                "visqol_mos": p.visqol,
            })).collect::<Vec<_>>(),
            "lyra": lyra.iter().map(|p| serde_json::json!({
                "actual_bps": p.bps,
                "snr_db": p.snr,
            })).collect::<Vec<_>>(),
        }));
    }

    let mean = |v: &[f64]| {
        if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    let wins = |v: &[f64]| v.iter().filter(|&&d| d > 0.0).count();

    let verdict = if all_ok {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };

    common::finish_with_profile(
        "learned-lossy",
        crate::learned::profile::LearnedProfile::Exp3,
        receipts_root,
        LEARNED_LOSSY_SHA256,
        &projection,
        verdict,
        format!(
            "lossy codec vole.audio.lossy.exp1 over {} cases vs external Opus{} and Lyra{} at \
             matched actual bitrate: VOLE wins {} of {} Opus SNR cells (mean {:+.2} dB), {} of {} \
             Opus ViSQOL cells (mean {:+.3} MOS), and {} of {} Lyra SNR cells (mean {:+.2} dB); \
             only VOLE-derived integers freeze the result",
            rows.len(),
            if tools.opusenc.is_some() {
                ""
            } else {
                " (NOT_AVAILABLE)"
            },
            if tools.lyra_enc.is_some() {
                ""
            } else {
                " (NOT_AVAILABLE)"
            },
            wins(&opus_deltas),
            opus_deltas.len(),
            mean(&opus_deltas),
            wins(&visqol_deltas),
            visqol_deltas.len(),
            mean(&visqol_deltas),
            wins(&lyra_deltas),
            lyra_deltas.len(),
            mean(&lyra_deltas),
        ),
        vec![
            (
                "harness",
                serde_json::json!({
                    "profile": crate::lossy::LOSSY_PROFILE,
                    "opus": tools.opusenc.as_ref().map(|p| p.to_string_lossy().to_string()),
                    "lyra": tools.lyra_enc.as_ref().map(|p| p.to_string_lossy().to_string()),
                    "visqol": tools.visqol.as_ref().map(|p| p.to_string_lossy().to_string()),
                    "align_lag_samples": ALIGN_LAG,
                    "opus_rates": OPUS_RATES,
                    "lyra_rates": LYRA_RATES,
                    "note": "competitors are external processes; no code is imported, linked or wrapped",
                }),
            ),
            (
                "matched_bitrate_summary",
                serde_json::json!({
                    "opus_snr_delta_mean_db": mean(&opus_deltas),
                    "opus_snr_cells": opus_deltas.len(),
                    "opus_snr_wins": wins(&opus_deltas),
                    "opus_visqol_delta_mean": mean(&visqol_deltas),
                    "opus_visqol_cells": visqol_deltas.len(),
                    "opus_visqol_wins": wins(&visqol_deltas),
                    "lyra_snr_delta_mean_db": mean(&lyra_deltas),
                    "lyra_snr_cells": lyra_deltas.len(),
                    "lyra_snr_wins": wins(&lyra_deltas),
                }),
            ),
            ("cases", serde_json::json!(rows)),
            (
                "limitations",
                serde_json::json!([
                    "external codecs are compared on the first channel (Opus/Lyra decode to \
                     mono-comparable output); VOLE is measured per channel and averaged",
                    "Lyra is measured only on 16 kHz mono, its native domain; cells outside a \
                     competitor's measured bitrate range are null rather than extrapolated",
                    "ViSQOL is reported where the pinned binary and model run; absent metric \
                     cells are null, never invented",
                    "Opus is asked for a rate ladder with VBR, so its achieved bitrate differs \
                     from the request; all comparison is at achieved bitrate",
                ]),
            ),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_round_trips() {
        let dir = std::env::temp_dir().join("vole-wav-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.wav");
        let s: Vec<i16> = (0..1000).map(|i| (i * 37) as i16).collect();
        write_wav(&path, &s, 2, 16_000).unwrap();
        let (back, ch, rate) = read_wav(&path).unwrap();
        assert_eq!(ch, 2);
        assert_eq!(rate, 16_000);
        assert_eq!(back, s);
    }

    #[test]
    fn alignment_finds_a_known_delay() {
        let n = 4000usize;
        let reference: Vec<i32> = (0..n)
            .map(|i| ((i as f64 * 0.1).sin() * 1000.0) as i32)
            .collect();
        let mut test = vec![0i32; 300];
        test.extend_from_slice(&reference);
        let lag = best_lag(&reference, &test, 500);
        assert!(lag.abs() >= 250, "lag {lag}");
        let (a, b) = aligned_pair(&reference, &test, lag);
        assert!(snr_db(&a, &b) > 60.0);
    }

    #[test]
    fn interpolation_refuses_to_extrapolate() {
        let pts = vec![
            Point {
                bps: 10.0,
                snr: 1.0,
                visqol: Some(1.0),
            },
            Point {
                bps: 20.0,
                snr: 3.0,
                visqol: Some(3.0),
            },
        ];
        assert_eq!(interp(&pts, 15.0, |p| Some(p.snr)), Some(2.0));
        assert_eq!(interp(&pts, 5.0, |p| Some(p.snr)), None);
        assert_eq!(interp(&pts, 25.0, |p| Some(p.snr)), None);
        assert_eq!(interp(&pts, 15.0, |p| p.visqol), Some(2.0));
    }
}
