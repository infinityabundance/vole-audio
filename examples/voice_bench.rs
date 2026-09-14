//! Fast local voice-quality harness (NOT part of the court battery).
//!
//! Runs only the VOLE clean ladder over the frozen `effectiveness` corpus and
//! prints actual bitrate and aligned SNR per shape/rate, so an encoder change
//! can be measured in seconds instead of re-running the full external-competitor
//! court. It deliberately does not touch `receipts/` and makes no claim: it is a
//! development instrument. The authoritative measurement remains
//! `court learned-voice-stream`.
//!
//! ```text
//! cargo run --release --example voice_bench               # all shapes/rates
//! cargo run --release --example voice_bench -- 320 1      # one shape
//! cargo run --release --example voice_bench -- diag       # predictor RD curve
//! cargo run --release --example voice_bench -- celp       # excitation coder RD
//! cargo run --release --example voice_bench -- exp2       # voice.exp2 core A/B
//! cargo run --release --example voice_bench -- visqol     # perceptual A/B (ViSQOL)
//! cargo run --release --example voice_bench -- dump       # write case WAVs
//! ```

use std::path::Path;
use std::process::Command;

use vole_audio::learned::corpus_real;
use vole_audio::voice::exp2::{self, Exp2Codec, Options};
use vole_audio::voice::predict as vp;
use vole_audio::voice::{VoiceConfig, VoiceDecoder, VoiceEncoder};

const CASE_TARGET_SAMPLES: usize = 16_000 * 4;
const CASE_CLIP_POOL: usize = 40;
const ALIGN_LAG: i64 = 512;
const RATES: [u32; 6] = [6_000, 8_000, 12_000, 16_000, 24_000, 32_000];
const SHAPES: [(u32, u8); 3] = [(320, 1), (160, 1), (160, 2)];
const VQ_INDEX_BYTES: usize = 4;

fn percentile(sorted: &[i64], p: f64) -> i64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * p / 100.0).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn best_lag(reference: &[i32], test: &[i32], bound: i64) -> i64 {
    let n = reference.len().min(test.len()) as i64;
    if n == 0 {
        return 0;
    }
    let mut best = (0i64, f64::NEG_INFINITY);
    for lag in -bound..=bound {
        let (a_lo, b_lo) = (lag.max(0) as usize, (-lag).max(0) as usize);
        let len = (n - lag.abs()).max(1) as usize;
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..len {
            let r = f64::from(reference[a_lo + i]);
            let t = f64::from(test[b_lo + i]);
            num += r * t;
            den += t * t;
        }
        if den <= 0.0 {
            continue;
        }
        let score = num / den.sqrt();
        if score > best.1 {
            best = (lag, score);
        }
    }
    best.0
}

fn snr_aligned(reference: &[i32], test: &[i32]) -> f64 {
    let lag = best_lag(reference, test, ALIGN_LAG);
    let (a_lo, b_lo) = (lag.max(0) as usize, (-lag).max(0) as usize);
    let n = (reference.len().min(test.len()) as i64 - lag.abs()).max(0) as usize;
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for i in 0..n {
        let r = f64::from(reference[a_lo + i]);
        let t = f64::from(test[b_lo + i]);
        sig += r * r;
        err += (r - t) * (r - t);
    }
    if err <= 0.0 {
        f64::INFINITY
    } else if sig <= 0.0 {
        f64::NEG_INFINITY
    } else {
        10.0 * (sig / err).log10()
    }
}

fn cases() -> Vec<(String, Vec<i32>)> {
    let clips = corpus_real::effectiveness_clips();
    let scratch = std::path::PathBuf::from("target/voice-court/scratch");
    let loaded = corpus_real::load_cases(&clips, CASE_CLIP_POOL, &scratch)
        .expect("effectiveness corpus loads");
    let mut groups: Vec<(String, Vec<i32>)> = Vec::new();
    for c in loaded {
        if usize::from(c.clip.channels) != 1 || c.clip.sample_rate_hz != 16_000 {
            continue;
        }
        match groups.last_mut() {
            Some((_, acc)) if acc.len() < CASE_TARGET_SAMPLES => {
                let want = (CASE_TARGET_SAMPLES - acc.len()).min(c.samples.len());
                acc.extend_from_slice(&c.samples[..want]);
            }
            _ => groups.push((c.clip.id.clone(), c.samples.clone())),
        }
    }
    groups
}

fn config_for(frame_len: u32, frames_per_packet: u8, bps: u32) -> VoiceConfig {
    VoiceConfig {
        sample_rate_hz: 16_000,
        frame_len,
        frames_per_packet,
        target_bits_per_second: bps,
        dtx: false,
        capsule_cadence: vole_audio::voice::DEFAULT_CAPSULE_CADENCE,
        redundancy: false,
    }
}

/// Wire-format split of one packet: (model bytes, residual bytes, frames,
/// VQ-coded frames). Parses the documented layout; development instrument only.
fn packet_split(packet: &[u8], frame_len: usize) -> Option<(usize, usize, usize, usize)> {
    let flags = packet[0];
    let mut vq_frames = 0usize;
    if flags & 1 != 0 {
        return Some((0, 0, 0, 0)); // NO_DATA
    }
    let nf = usize::from((flags >> 4) & 0b11) + 1;
    let mut pos = 2usize;
    if flags & 2 != 0 {
        let mask = *packet.get(pos)?;
        pos += 2;
        if mask & 1 != 0 {
            pos += 3;
        }
        if mask & 2 != 0 {
            pos += 1;
        }
        if mask & 4 != 0 {
            pos += 4;
        }
        if mask & 8 != 0 {
            pos += 2;
        }
    }
    if flags & 4 != 0 {
        let mask = *packet.get(pos)?;
        pos += 2;
        if mask & 1 != 0 {
            pos += 1;
        }
        if mask & 2 != 0 {
            pos += 3;
        }
        if mask & 4 != 0 {
            pos += 1;
        }
        if mask & 8 != 0 {
            pos += 8;
        }
    }
    let mut model = 0usize;
    let mut residual = 0usize;
    for _ in 0..nf {
        let mode = *packet.get(pos)?;
        let order_idx = ((mode >> 3) & 0b111) as usize;
        if order_idx >= 5 {
            return None;
        }
        let order = [8usize, 10, 12, 14, 16][order_idx];
        let vq = mode & 3 == 3;
        let width = [5u8, 6, 7, 8][((mode >> 6) & 0b11) as usize];
        let pitch = (mode >> 2) & 1 == 1;
        let k_bytes = if vq {
            VQ_INDEX_BYTES
        } else {
            (order * usize::from(width)).div_ceil(8)
        };
        let overhead = 2 + if pitch { 3 } else { 0 } + k_bytes + 2;
        let rlen = u16::from_le_bytes([
            *packet.get(pos + overhead - 2)?,
            *packet.get(pos + overhead - 1)?,
        ]) as usize;
        model += overhead;
        residual += rlen;
        vq_frames += usize::from(vq);
        // A scalar frame carries a trailing per-subframe gain block; a CELP
        // payload does not. The residual's codec id tells them apart.
        let celp = rlen > 0 && *packet.get(pos + overhead - rlen)? == 3;
        let gain_block = if celp {
            0
        } else {
            vp::gain_block_bytes(frame_len)
        };
        pos += overhead + rlen + gain_block;
    }
    Some((model, residual, nf, vq_frames))
}

fn main() {
    if !corpus_real::available() {
        eprintln!("the frozen real speech corpus is not available");
        std::process::exit(1);
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("diag") => {
            diagnose();
            return;
        }
        Some("celp") => {
            celery_probe();
            return;
        }
        Some("dump") => {
            dump_cases();
            return;
        }
        Some("pg") => {
            prediction_gain();
            return;
        }
        Some("cmp") => {
            coder_compare(320);
            return;
        }
        Some("cmp160") => {
            coder_compare(160);
            return;
        }
        Some("exp2") => {
            exp2_probe(args.get(1).map(String::as_str));
            return;
        }
        Some("visqol") => {
            visqol_probe();
            return;
        }
        Some("weights") => {
            weight_probe();
            return;
        }
        _ => {}
    }
    let filter: Option<(u32, u8)> = if args.len() >= 2 {
        Some((args[0].parse().unwrap(), args[1].parse().unwrap()))
    } else {
        None
    };
    let cases = cases();
    println!("cases: {}", cases.len());
    for (frame_len, fpp) in SHAPES {
        if let Some(f) = filter
            && f != (frame_len, fpp)
        {
            continue;
        }
        for bps in RATES {
            let cfg = config_for(frame_len, fpp, bps);
            let n = cfg.packet_samples();
            let mut bytes = 0usize;
            let mut model_bytes = 0usize;
            let mut residual_bytes = 0usize;
            let mut vq_frames = 0usize;
            let mut total_samples = 0usize;
            let mut snr_sum = 0.0f64;
            let mut snr_min = f64::INFINITY;
            let mut snr_max = f64::NEG_INFINITY;
            let mut encode_us: Vec<i64> = Vec::new();
            for (_, source) in &cases {
                let mut enc = VoiceEncoder::new(cfg).unwrap();
                let mut dec = VoiceDecoder::new(cfg).unwrap();
                let mut out = Vec::with_capacity(source.len());
                for chunk in source.chunks(n) {
                    if chunk.len() < n {
                        break;
                    }
                    let t0 = std::time::Instant::now();
                    let pkt = enc.encode_packet(chunk).unwrap();
                    encode_us.push(t0.elapsed().as_micros() as i64);
                    bytes += pkt.len();
                    if let Some((m, r, _, v)) = packet_split(&pkt, n) {
                        model_bytes += m;
                        residual_bytes += r;
                        vq_frames += v;
                    }
                    let (_k, samples) = dec.decode_packet(&pkt).unwrap();
                    out.extend_from_slice(&samples);
                }
                total_samples += source.len();
                let s = snr_aligned(source, &out);
                snr_sum += s;
                snr_min = snr_min.min(s);
                snr_max = snr_max.max(s);
            }
            let seconds = total_samples as f64 / 16_000.0;
            let actual_bps = bytes as f64 * 8.0 / seconds;
            let mean = snr_sum / cases.len() as f64;
            encode_us.sort_unstable();
            println!(
                "shape {frame_len}x{fpp} target {bps:>5} bps -> actual {actual_bps:>9.1} bps | \
                 SNR mean {mean:>6.2} dB (min {snr_min:>6.2}, max {snr_max:>6.2}) | \
                 enc p99 {:>5} us max {:>5} us | model {model_bytes:>7} res {residual_bytes:>7} B vq {vq_frames:>4}",
                percentile(&encode_us, 99.0),
                encode_us.last().copied().unwrap_or(0)
            );
        }
    }
}

fn opts(celp: bool, track_acelp: bool, tcx: bool) -> Options {
    Options {
        celp,
        track_acelp,
        tcx,
        proc: true,
        env_weight: exp2::DEFAULT_ENV_WEIGHT,
    }
}

/// The same, with the procedural core withheld entirely.
fn opts_no_proc(celp: bool, track_acelp: bool, tcx: bool) -> Options {
    Options {
        proc: false,
        ..opts(celp, track_acelp, tcx)
    }
}

/// The procedural core alone. Separates "the mechanism is weak" from "the
/// selector chose it away", which a whole-stack row cannot distinguish.
fn opts_proc_only() -> Options {
    Options {
        celp: false,
        track_acelp: false,
        tcx: false,
        proc: true,
        env_weight: exp2::DEFAULT_ENV_WEIGHT,
    }
}

/// `voice.exp2` core A/B on real speech: the same held-aside code path with the
/// 7C.2-E fractional-track core offered and withheld, at every declared rate.
/// Development instrument; the court remains the authority.
fn exp2_probe(filter: Option<&str>) {
    let cases = cases();
    let mut source: Vec<i32> = Vec::new();
    for (_, s) in &cases {
        source.extend_from_slice(s);
    }
    const FRAME: usize = 320;
    let configs: [(&str, Options); 6] = [
        ("scalar+noise      ", opts_no_proc(false, false, false)),
        ("proc only (7C.2-H)", opts_proc_only()),
        ("+celp (7C.2-B/D)  ", opts_no_proc(true, false, false)),
        ("+acelp (7C.2-E)   ", opts_no_proc(true, true, false)),
        ("+proc (7C.2-H)    ", opts(true, true, false)),
        ("+tcx (7C.2-G)     ", opts(true, true, true)),
    ];
    println!(
        "voice.exp2 core A/B over {} samples of frozen effectiveness speech",
        source.len()
    );
    println!("  rate | configuration       | SNR mean | bits/frame | enc p99 | families (frames)");
    for (bps, bits) in exp2::DECLARED_RATES {
        if let Some(f) = filter
            && f != bps.to_string()
        {
            continue;
        }
        for (label, opts) in configs {
            let mut state = vp::VoiceState::new();
            let mut sig = 0.0f64;
            let mut err = 0.0f64;
            let mut used = 0usize;
            let mut frames = 0usize;
            let mut per_family: std::collections::BTreeMap<&str, usize> =
                std::collections::BTreeMap::new();
            let mut pulses = 0usize;
            let mut enc_us: Vec<i64> = Vec::new();
            for chunk in source.chunks(FRAME) {
                if chunk.len() < FRAME {
                    break;
                }
                let enc_state = state.clone();
                let t0 = std::time::Instant::now();
                let (bytes, nbits) = Exp2Codec::encode_frame_with(&enc_state, chunk, bits, opts);
                enc_us.push(t0.elapsed().as_micros() as i64);
                assert!(
                    nbits <= bits,
                    "{bps}: {nbits} bits over the {bits}-bit allowance"
                );
                used += nbits;
                if let Ok(f) = exp2::Frame2::read(&bytes, FRAME) {
                    *per_family.entry(exp2::family_label(&f)).or_default() += 1;
                    pulses += exp2::pulse_count(&f);
                }
                let out = Exp2Codec::decode_frame(&mut state, &bytes, FRAME).unwrap();
                let f: Vec<f64> = chunk.iter().map(|&v| f64::from(v)).collect();
                let o: Vec<f64> = out.iter().map(|&v| f64::from(v)).collect();
                sig += f.iter().map(|v| v * v).sum::<f64>();
                err += f
                    .iter()
                    .zip(&o)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>();
                frames += 1;
            }
            enc_us.sort_unstable();
            let snr = if err > 0.0 {
                10.0 * (sig / err).log10()
            } else {
                f64::NAN
            };
            let fam: String = per_family
                .iter()
                .map(|(k, v)| format!("{k} {v}"))
                .collect::<Vec<_>>()
                .join(", ");
            println!(
                "  {:>4} | {label} | {:>8.2} | {:>10.1} | {:>6}us | {fam} (pulses/frame {:.1})",
                bps,
                snr,
                used as f64 / frames.max(1) as f64,
                percentile(&enc_us, 99.0),
                pulses as f64 / frames.max(1) as f64
            );
        }
    }
}

/// Write mono 16 kHz 16-bit PCM. Shared by the dump and perceptual instruments.
fn write_wav(path: &Path, samples: &[i32]) -> std::io::Result<()> {
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    let data_len = (samples.len() * 2) as u32;
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&16_000u32.to_le_bytes());
    out.extend_from_slice(&32_000u32.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        out.extend_from_slice(&(s.clamp(-32_768, 32_767) as i16).to_le_bytes());
    }
    std::fs::write(path, out)
}

/// The frozen ViSQOL protocol the court uses, duplicated here on purpose: this is
/// a development instrument and must not be able to change what the court
/// measures by editing a shared wrapper.
fn visqol_mos(reference: &Path, degraded: &Path) -> Option<f64> {
    let exe = Path::new("research/visqol-master/bazel-bin/visqol");
    let model = Path::new(
        "research/visqol-master/model/\
         lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite",
    );
    if !exe.is_file() || !model.is_file() {
        return None;
    }
    let csv = degraded.with_extension("visqol.csv");
    let _ = std::fs::remove_file(&csv);
    let out = Command::new(exe)
        .arg(format!("--reference_file={}", reference.display()))
        .arg(format!("--degraded_file={}", degraded.display()))
        .arg(format!("--similarity_to_quality_model={}", model.display()))
        .arg(format!("--results_csv={}", csv.display()))
        .arg("--use_speech_mode=true")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
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

/// Encode, decode and score one configuration over the development corpus.
/// Returns `(MOS-LQO mean, bits/frame, encode p99 µs, cases scored)`.
#[allow(clippy::too_many_arguments)]
fn score_config(
    cases: &[(String, Vec<i32>)],
    refs: &[std::path::PathBuf],
    dir: &Path,
    bits: usize,
    opts: Options,
    tag: &str,
    frame: usize,
) -> Option<(f64, f64, i64, usize)> {
    let mut mos_sum = 0.0;
    let mut scored = 0usize;
    let mut used = 0usize;
    let mut frames = 0usize;
    let mut enc_us: Vec<i64> = Vec::new();
    for (i, (_, source)) in cases.iter().enumerate() {
        let mut state = vp::VoiceState::new();
        let mut out: Vec<i32> = Vec::with_capacity(source.len());
        for chunk in source.chunks(frame) {
            if chunk.len() < frame {
                break;
            }
            let enc_state = state.clone();
            let t0 = std::time::Instant::now();
            let (bytes, nbits) = Exp2Codec::encode_frame_with(&enc_state, chunk, bits, opts);
            enc_us.push(t0.elapsed().as_micros() as i64);
            used += nbits;
            frames += 1;
            out.extend_from_slice(&Exp2Codec::decode_frame(&mut state, &bytes, frame).unwrap());
        }
        let p = dir.join(format!("deg-{tag}-{i}.wav"));
        write_wav(&p, &out).unwrap();
        if let Some(m) = visqol_mos(&refs[i], &p) {
            mos_sum += m;
            scored += 1;
        }
    }
    enc_us.sort_unstable();
    (scored > 0).then(|| {
        (
            mos_sum / scored as f64,
            used as f64 / frames.max(1) as f64,
            percentile(&enc_us, 99.0),
            scored,
        )
    })
}

/// Write the development references once and report ViSQOL's own ceiling.
fn write_references(
    cases: &[(String, Vec<i32>)],
    dir: &Path,
) -> Option<(Vec<std::path::PathBuf>, f64)> {
    let refs: Vec<std::path::PathBuf> = cases
        .iter()
        .enumerate()
        .map(|(i, (_, s))| {
            let p = dir.join(format!("ref-{i}.wav"));
            write_wav(&p, s).unwrap();
            p
        })
        .collect();
    let mut sum = 0.0;
    let mut n = 0usize;
    for p in &refs {
        if let Some(m) = visqol_mos(p, p) {
            sum += m;
            n += 1;
        }
    }
    (n > 0).then(|| (refs, sum / n as f64))
}

/// Fit the selector's envelope weight against ViSQOL on the **development**
/// corpus.
///
/// This is the charter's prescribed procedure: the proxy's free parameter is
/// fitted on development material and then frozen, and the held-out challenger
/// court is what decides whether the proxy correlates with a perceptual judge.
/// The sweep also reports bits/frame, because §14 measured the MSE objective
/// leaving up to 53 % of the allowance idle and a fix must spend it.
fn weight_probe() {
    let cases = cases();
    let dir = std::path::PathBuf::from("target/voice-bench-vq");
    std::fs::create_dir_all(&dir).unwrap();
    const FRAME: usize = 320;
    let Some((refs, ceiling)) = write_references(&cases, &dir) else {
        println!("visqol is not available (research/visqol-master is missing)");
        return;
    };
    let weights = [0.0f64, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0];
    let wanted = [6_000u32, 8_000, 9_200, 12_000, 16_000];
    println!(
        "envelope-weight sweep over {} development cases (ViSQOL speech mode, ceiling {ceiling:.3})",
        cases.len()
    );
    println!("  weight | rate | MOS-LQO mean | bits/frame | enc p99");
    for w in weights {
        for (bps, bits) in exp2::DECLARED_RATES {
            if !wanted.contains(&bps) {
                continue;
            }
            let o = Options {
                celp: true,
                track_acelp: true,
                tcx: false,
                proc: true,
                env_weight: w,
            };
            let tag = format!("w{}-{bps}", (w * 100.0) as u32);
            if let Some((mos, bpf, p99, _)) =
                score_config(&cases, &refs, &dir, bits, o, &tag, FRAME)
            {
                println!("  {w:>6} | {bps:>4} | {mos:>12.3} | {bpf:>10.1} | {p99:>6}us");
            }
        }
    }
}

/// Perceptual A/B of the `voice.exp2` cores on the development corpus, with
/// ViSQOL as arbiter.
///
/// This exists because §12–§13 of the charter measured the MSE objective
/// structurally preferring silence, so waveform SNR cannot decide whether a core
/// is good. The development corpus is used deliberately: the held-out challenger
/// corpus stays untouched so a selector fitted here can still be judged there.
fn visqol_probe() {
    let cases = cases();
    let dir = std::path::PathBuf::from("target/voice-bench-vq");
    std::fs::create_dir_all(&dir).unwrap();
    const FRAME: usize = 320;
    let configs: [(&str, Options); 6] = [
        ("noise only        ", opts_no_proc(false, false, false)),
        ("proc only (7C.2-H)", opts_proc_only()),
        ("+celp (7C.2-B/D)  ", opts_no_proc(true, false, false)),
        ("+acelp (7C.2-E)   ", opts_no_proc(true, true, false)),
        ("+proc (7C.2-H)    ", opts(true, true, false)),
        ("+tcx (7C.2-G)     ", opts(true, true, true)),
    ];
    // References first, and a self-comparison so the instrument reports its own
    // ceiling instead of implying that 5.0 is reachable.
    let Some((refs, ceiling)) = write_references(&cases, &dir) else {
        println!("visqol is not available (research/visqol-master is missing)");
        return;
    };
    println!(
        "voice.exp2 perceptual A/B over {} development cases (ViSQOL speech mode)",
        cases.len()
    );
    println!("  self-comparison ceiling: {ceiling:.3} MOS-LQO (not 5.0)");
    println!("  rate | configuration       | MOS-LQO mean | bits/frame | enc p99");
    for (bps, bits) in exp2::DECLARED_RATES {
        for (ci, (label, opts)) in configs.iter().enumerate() {
            let tag = format!("c{ci}-{bps}");
            if let Some((mos, bpf, p99, _)) =
                score_config(&cases, &refs, &dir, bits, *opts, &tag, FRAME)
            {
                println!("  {bps:>4} | {label} | {mos:>12.3} | {bpf:>10.1} | {p99:>6}us");
            }
        }
    }
}

/// Write the court's cases as 16 kHz mono WAV so external competitors can be
/// measured on exactly the same material.
fn dump_cases() {
    let dir = std::path::PathBuf::from("target/voice-bench");
    std::fs::create_dir_all(&dir).unwrap();
    for (i, (id, source)) in cases().iter().enumerate() {
        let path = dir.join(format!("case-{i}-{id}.wav"));
        let mut out = Vec::with_capacity(44 + source.len() * 2);
        let data_len = (source.len() * 2) as u32;
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_len).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&16_000u32.to_le_bytes());
        out.extend_from_slice(&32_000u32.to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
        for &s in source {
            out.extend_from_slice(&(s.clamp(-32_768, 32_767) as i16).to_le_bytes());
        }
        std::fs::write(&path, out).unwrap();
        println!("{}", path.display());
    }
}

/// Deterministic unit-RMS codebook used by the CELP probe.
fn random_codebook(bits: usize, len: usize) -> Vec<Vec<f64>> {
    let mut cb = Vec::with_capacity(1 << bits);
    let mut s = 0x1234_5678_9abc_def0u64;
    for _ in 0..(1 << bits) {
        let mut v = vec![0.0f64; len];
        for slot in v.iter_mut() {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            *slot = (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
        }
        let rms = (v.iter().map(|x| x * x).sum::<f64>() / len as f64).sqrt();
        for x in v.iter_mut() {
            *x /= rms.max(1e-9);
        }
        cb.push(v);
    }
    cb
}

/// Closed-loop CELP distortion for one frame at a fixed per-frame byte budget,
/// starting from `state`, using an analysis-by-synthesis search per subframe.
/// Returns `(distortion, bytes)`.
fn celp_frame(
    state: &vp::VoiceState,
    f: &[f64],
    model: &vp::FrameModel,
    kq: &[i32],
    cb: &[Vec<f64>],
    sub_len: usize,
    gains: &[f64],
) -> (f64, usize) {
    let mut probe = state.clone();
    let mut dist = 0.0f64;
    for sub in 0..(f.len() / sub_len) {
        let lo = sub * sub_len;
        let hi = lo + sub_len;
        let target = &f[lo..hi];
        let rms = (target.iter().map(|x| x * x).sum::<f64>() / sub_len as f64).sqrt();
        let mut best: Option<(f64, Vec<f64>)> = None;
        for code in cb.iter() {
            for &g_unit in gains.iter() {
                let g = g_unit * rms.max(1.0) * 0.7;
                let ex: Vec<f64> = code.iter().map(|x| x * g).collect();
                let mut st = probe.clone();
                let out = vp::synthesize_excitation(
                    &mut st,
                    kq,
                    model.width,
                    model.lag,
                    model.ltpg_q,
                    &ex,
                );
                let d: f64 = target
                    .iter()
                    .zip(out.iter())
                    .map(|(x, y)| (x - y) * (x - y))
                    .sum();
                if best.as_ref().is_none_or(|b| d < b.0) {
                    best = Some((d, ex));
                }
            }
        }
        let (d, ex) = best.unwrap();
        dist += d;
        vp::synthesize_excitation(&mut probe, kq, model.width, model.lag, model.ltpg_q, &ex);
    }
    // 7-bit codeword + 6-bit gain per subframe, model cost excluded.
    (dist, 2 * (f.len() / sub_len))
}

/// Excitation-coder RD probe: scalar dead-zone residual versus an
/// analysis-by-synthesis random-codebook excitation, measured per frame from a
/// common source-driven predictor state (so the comparison isolates the
/// excitation coder, not the trajectory).
fn celery_probe() {
    let cases = cases();
    let (_, source) = &cases[0];
    let frame_len = 320usize;
    let sub_len = 80usize;
    let cb = random_codebook(7, sub_len);
    let gains: [f64; 8] = [0.125, 0.2, 0.32, 0.5, 0.8, 1.26, 2.0, 3.17];
    // Residual-byte budgets (the model cost is common to both coders).
    let budgets = [8usize, 12, 16, 24, 32, 48];

    let mut scalar_sig = vec![0.0f64; budgets.len()];
    let mut scalar_err = vec![0.0f64; budgets.len()];
    let mut scalar_bytes = vec![0usize; budgets.len()];
    let mut celp_sig = vec![0.0f64; budgets.len()];
    let mut celp_err = vec![0.0f64; budgets.len()];
    let mut celp_bytes = vec![0usize; budgets.len()];
    let mut frames = 0usize;

    let mut state = vp::VoiceState::new();
    for chunk in source.chunks(frame_len) {
        if chunk.len() < frame_len {
            break;
        }
        let f: Vec<f64> = chunk.iter().map(|&v| f64::from(v)).collect();
        let ferr: f64 = f.iter().map(|v| v * v).sum();
        let cands = vp::analyse(&state, chunk, 0);
        if cands.is_empty() {
            break;
        }
        let c = cands
            .iter()
            .filter(|c| c.model.order == 16)
            .min_by(|a, b| a.energy.total_cmp(&b.energy))
            .unwrap_or(&cands[0]);
        let kq = vp::quantise_k(&c.k_raw, 6);
        let _descr = c.model.description_bytes() + 2;

        // Scalar: every step, exact residual bytes, closed-loop distortion.
        let mut scalar: Vec<(f64, usize)> = Vec::new();
        for gain in -4..=44 {
            let syn = vp::close_loop(&state, &f, &c.model, &kq, gain);
            let bytes = vole_audio::voice::residual::encode_best(
                &syn.symbols,
                &vole_audio::learned::residual_codec2::SEARCH_CODECS,
            )
            .len();
            scalar.push((syn.distortion, bytes));
        }
        let (cdist, cbytes) = celp_frame(&state, &f, &c.model, &kq, &cb, sub_len, &gains);

        for (i, &b) in budgets.iter().enumerate() {
            if let Some((d, bytes)) = scalar
                .iter()
                .filter(|(_, bytes)| *bytes <= b)
                .min_by(|a, b| a.0.total_cmp(&b.0))
            {
                scalar_sig[i] += ferr;
                scalar_err[i] += d;
                scalar_bytes[i] += bytes;
            }
            if cbytes <= b {
                celp_sig[i] += ferr;
                celp_err[i] += cdist;
                celp_bytes[i] += cbytes;
            }
        }
        frames += 1;
        // A realistic predictor context for the next frame: source-driven state.
        for &x in chunk.iter() {
            state.push(f64::from(x), 0.0);
        }
    }
    let kbps = |b: f64| b * 8.0 / (frame_len as f64 / 16_000.0);
    println!(
        "excitation-coder RD ({} frames, fixed order-16 model, model cost excluded):",
        frames
    );
    println!(" residual |     scalar          |     CELP");
    for i in 0..budgets.len() {
        let ss = if scalar_err[i] > 0.0 {
            10.0 * (scalar_sig[i] / scalar_err[i]).log10()
        } else {
            f64::NAN
        };
        let cs = if celp_err[i] > 0.0 {
            10.0 * (celp_sig[i] / celp_err[i]).log10()
        } else {
            f64::NAN
        };
        let sb = scalar_bytes[i] as f64 / frames as f64;
        let cob = celp_bytes[i] as f64 / frames as f64;
        println!(
            " {:>6} B | {:>5.1} B {:>7.0} bps {:>6.1} dB | {:>5.1} B {:>7.0} bps {:>6.1} dB | \
             delta {:+.1} dB",
            budgets[i],
            sb,
            kbps(sb),
            ss,
            cob,
            kbps(cob),
            cs,
            cs - ss
        );
    }
    let _ = frames;
}

/// Predictor diagnosis: for a ladder of per-frame byte budgets, find the best
/// candidate/gain under the budget and report the achievable SNR. Development
/// instrument only.
/// Head-to-head RD of the scalar dead-zone DPCM residual against the CELP
/// analysis-by-synthesis excitation, on identical frames and identical byte
/// accounting.
///
/// Development instrument only, and read it with care: it advances the history
/// with the *true* signal (open loop), which flatters the predictor, so the
/// absolute SNRs here are not the codec's. Use it for the scalar-vs-CELP
/// *direction* on the same frames; the authoritative engine A/B is
/// `celp::SELECTED` toggled against the main bench and the court.
fn coder_compare(frame_len: usize) {
    use vole_audio::voice::celp;
    use vole_audio::voice::residual;
    let cases = cases();
    let budgets: [usize; 12] = [8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96];
    let mut sig = 0.0f64;
    let mut sc_err = vec![0.0f64; budgets.len()];
    let mut sc_by = vec![0usize; budgets.len()];
    let mut cl_err = vec![0.0f64; budgets.len()];
    let mut cl_by = vec![0usize; budgets.len()];
    let mut frames = 0usize;
    let mut sc_wins = 0usize;
    let mut cl_wins = 0usize;
    for (_, source) in &cases {
        let mut state = vp::VoiceState::new();
        for chunk in source.chunks(frame_len) {
            if chunk.len() < frame_len {
                break;
            }
            let f: Vec<f64> = chunk.iter().map(|&v| f64::from(v)).collect();
            sig += f.iter().map(|v| v * v).sum::<f64>();
            let cands = vp::analyse(&state, chunk, 0);
            let Some(best) = cands.iter().min_by(|a, b| a.energy.total_cmp(&b.energy)) else {
                break;
            };
            let model = best.model;
            let kq = &best.k_q;
            let descr = model.description_bytes();
            // Scalar: every gain in the ladder.
            let mut scalar: Vec<(usize, f64)> = Vec::new();
            for gain in -4..=44 {
                let s = vp::close_loop(&state, &f, &model, kq, gain);
                scalar.push((descr + residual::estimated_bytes(&s.symbols), s.distortion));
            }
            // CELP: every pulse count the format admits.
            let mut celp_rd: Vec<(usize, f64)> = Vec::new();
            for np in 1..=celp::MAX_PULSES {
                let (params, d, _shot) = celp::analyse(&state, &f, &model, kq, np);
                celp_rd.push((descr + celp::encode_payload(&params).len(), d));
            }
            // Which coder wins at each budget.
            for (bi, &b) in budgets.iter().enumerate() {
                let sc = scalar
                    .iter()
                    .filter(|(c, _)| *c <= b)
                    .min_by(|a, x| a.1.total_cmp(&x.1));
                let cl = celp_rd
                    .iter()
                    .filter(|(c, _)| *c <= b)
                    .min_by(|a, x| a.1.total_cmp(&x.1));
                if let Some(&(c, d)) = sc {
                    sc_err[bi] += d;
                    sc_by[bi] += c;
                }
                if let Some(&(c, d)) = cl {
                    cl_err[bi] += d;
                    cl_by[bi] += c;
                }
                match (sc, cl) {
                    (Some(a), Some(x)) => {
                        if a.1 <= x.1 {
                            sc_wins += 1;
                        } else {
                            cl_wins += 1;
                        }
                    }
                    (Some(_), None) => sc_wins += 1,
                    (None, Some(_)) => cl_wins += 1,
                    _ => {}
                }
            }
            frames += 1;
            for &x in chunk.iter() {
                state.push(f64::from(x), 0.0);
            }
        }
    }
    println!("scalar vs CELP RD, {frames} frames of {frame_len} samples (open loop state)");
    println!("  budget   scalar: B/fr  SNR      CELP: B/fr  SNR");
    for (bi, &b) in budgets.iter().enumerate() {
        if sig <= 0.0 {
            continue;
        }
        let sb = sc_by[bi] as f64 / frames as f64;
        let cb = cl_by[bi] as f64 / frames as f64;
        let ss = if sc_err[bi] > 0.0 {
            10.0 * (sig / sc_err[bi]).log10()
        } else {
            f64::NAN
        };
        let cs = if cl_err[bi] > 0.0 {
            10.0 * (sig / cl_err[bi]).log10()
        } else {
            f64::NAN
        };
        println!("  {b:>4} B  {sb:>7.1}  {ss:>7.2} dB  {cb:>7.1}  {cs:>7.2} dB");
    }
    println!("  per-cell wins: scalar {sc_wins}, celp {cl_wins}");
}

/// Upper-bound prediction gain: short-term (LPC) and long-term (pitch), measured
/// open-loop with the true signal as history. This localises whether a low SNR is
/// a prediction failure or a residual-coding failure. Development instrument only.
fn prediction_gain() {
    let cases = cases();
    let frame_len = 320usize;
    let mut sig = 0.0f64;
    let mut short_e = 0.0f64;
    let mut long_e = 0.0f64;
    let mut frames = 0usize;
    for (_, source) in &cases {
        let mut state = vp::VoiceState::new();
        for chunk in source.chunks(frame_len) {
            if chunk.len() < frame_len {
                break;
            }
            let f: Vec<f64> = chunk.iter().map(|&v| f64::from(v)).collect();
            sig += f.iter().map(|v| v * v).sum::<f64>();
            let cands = vp::analyse(&state, chunk, 0);
            let Some(best) = cands.iter().min_by(|a, b| a.energy.total_cmp(&b.energy)) else {
                break;
            };
            short_e += best.energy;
            // Long-term gain on the actual short-term residual, open loop.
            let w = vp::weights_of(&best.k_q, best.model.width);
            let hist = state.chronological(vp::HIST_LEN);
            let mut resid: Vec<f64> = Vec::with_capacity(hist.len() + f.len());
            for (i, &h) in hist.iter().enumerate() {
                let mut p = 0.0;
                for (j, &wj) in w.iter().enumerate() {
                    if i > j {
                        p += wj * hist[i - 1 - j];
                    }
                }
                resid.push(h - p);
            }
            let base = resid.len();
            for (i, &x) in f.iter().enumerate() {
                let mut p = 0.0;
                for (j, &wj) in w.iter().enumerate() {
                    let t = i as i64 - 1 - j as i64;
                    let v = if t >= 0 {
                        f[t as usize]
                    } else {
                        hist[(base as i64 + t) as usize]
                    };
                    p += wj * v;
                }
                resid.push(x - p);
            }
            let mut best_e = f64::INFINITY;
            for lag in 32..=288usize {
                if lag > base {
                    break;
                }
                let mut num = 0.0;
                let mut den = 0.0;
                for i in base..resid.len() {
                    num += resid[i] * resid[i - lag];
                    den += resid[i - lag] * resid[i - lag];
                }
                if den <= 1e-12 {
                    continue;
                }
                let g = (num / den).clamp(0.0, 1.5);
                let mut e = 0.0;
                for i in base..resid.len() {
                    let r = resid[i] - g * resid[i - lag];
                    e += r * r;
                }
                if e < best_e {
                    best_e = e;
                }
            }
            long_e += best_e.min(best.energy);
            frames += 1;
            for &x in chunk.iter() {
                state.push(f64::from(x), 0.0);
            }
        }
    }
    let db = |e: f64| 10.0 * (sig / e).log10();
    println!(
        "prediction gain over {frames} frames of {frame_len} samples (open loop, true history)"
    );
    println!("  signal energy          {sig:>12.4e}");
    println!(
        "  LPC residual energy    {short_e:>12.4e}   gain {:>5.2} dB",
        db(short_e)
    );
    println!(
        "  +pitch residual energy {long_e:>12.4e}   gain {:>5.2} dB",
        db(long_e)
    );
}

fn diagnose() {
    let cases = cases();
    let (_, source) = &cases[0];
    let frame_len = 320usize;
    let budgets: [usize; 11] = [8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 96];
    let mut total_frames = 0usize;
    let mut signal: Vec<f64> = budgets.iter().map(|_| 0.0).collect();
    let mut error: Vec<f64> = budgets.iter().map(|_| 0.0).collect();
    let mut bytes: Vec<usize> = budgets.iter().map(|_| 0).collect();
    let mut state = vp::VoiceState::new();
    for chunk in source.chunks(frame_len) {
        if chunk.len() < frame_len {
            break;
        }
        let f: Vec<f64> = chunk.iter().map(|&v| f64::from(v)).collect();
        let ferr: f64 = f.iter().map(|v| v * v).sum();
        let cands = vp::analyse(&state, chunk, 0);
        if cands.is_empty() {
            break;
        }
        let descr = cands
            .iter()
            .map(|c| c.model.description_bytes() + 2)
            .min()
            .unwrap_or(4);
        let mut evaluated: Vec<(usize, f64)> = Vec::new();
        for c in cands.iter() {
            for gain in -4..=44 {
                let s = vp::close_loop(&state, &f, &c.model, &c.k_q, gain);
                let cost = descr
                    + vole_audio::voice::residual::encode_best(
                        &s.symbols,
                        &vole_audio::learned::residual_codec2::SEARCH_CODECS,
                    )
                    .len();
                evaluated.push((cost, s.distortion));
            }
        }
        for (bi, &budget) in budgets.iter().enumerate() {
            if let Some(&(cost, distortion)) = evaluated
                .iter()
                .filter(|(c, _)| *c <= budget)
                .min_by(|a, b| a.1.total_cmp(&b.1))
            {
                bytes[bi] += cost;
                signal[bi] += ferr;
                error[bi] += distortion;
            }
        }
        total_frames += 1;
        for &x in chunk.iter() {
            state.push(f64::from(x), 0.0);
        }
    }
    println!("predictor RD curve over {total_frames} frames (320 samples each):");
    for (bi, &budget) in budgets.iter().enumerate() {
        if signal[bi] <= 0.0 {
            continue;
        }
        let snr = 10.0 * (signal[bi] / error[bi]).log10();
        let kbps = bytes[bi] as f64 * 8.0 / (total_frames as f64 * frame_len as f64 / 16_000.0);
        println!(
            "  budget {budget:>3} B -> used {:>5.1} B/frame ({kbps:>7.0} bps) SNR {snr:>6.2} dB",
            bytes[bi] as f64 / total_frames as f64
        );
    }
}
