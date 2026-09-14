//! `court learned-voice-stream` — Phase 7C: `vole.audio.stream.voice.exp1`.
//!
//! The constitution is `docs/PHASE_7C.md`; it was frozen before this file
//! existed. Nothing here may redefine what counts as a measurement.
//!
//! The court is deliberately *three* measurements in one, because 7C's gate is
//! not a bitrate and not a clean listening score:
//!
//! ```text
//! 1  clean quality at matched actual bitrate   (VOLE vs Opus, EVS, Lyra)
//! 2  quality under packet loss and jitter      (deterministic impairment engine)
//! 3  one-way latency with its full accounting  (frame accumulation → playout)
//! ```
//!
//! External codecs are separate processes and are never imported, linked, used
//! as a fallback, or copied into a VOLE object. EVS is additionally driven
//! through the G.192 *bad frame* mechanism (sync word `0x6b20`), which is the
//! standard's own erasure flag — that is how a real loss comparison against EVS
//! concealment is possible without a network.
//!
//! Only VOLE-derived integers enter the frozen projection, so the frozen hash is
//! stable across host tool versions. Competitor numbers appear in the receipt
//! and in the summary, never in the frozen hash.
//!
//! Absent metrics are `null` / `NOT_AVAILABLE`. Nothing is invented.

use crate::courts::learned_common as common;
use crate::courts::learned_lossy as media;
use crate::error::{Error, Result};
use crate::evidence::timing::Stopwatch;
use crate::status::Verdict;
use crate::voice::impair::{
    self, Arrival, Impairment, LatencyBreakdown, SimStats, TransportPacket,
};
use crate::voice::{self, VoiceConfig, VoiceDecoder, VoiceEncoder};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Frozen static-result hash over the VOLE-derived projection. Empty until the
/// first observation is frozen; see `docs/PHASE_7C.md` §11.
pub const LEARNED_VOICE_STREAM_SHA256: &str =
    "cc2addfa019a96b3c722ddaabe9fd0dff7eee4ed0cc47039eb435acecc2dba50";

/// VOLE voice operating points (bits per second).
const VOLE_RATES: [u32; 6] = [6_000, 8_000, 12_000, 16_000, 24_000, 32_000];
/// The three packet shapes the constitution defines.
const SHAPES: [(u32, u8); 3] = [(160, 1), (320, 1), (160, 2)];
/// Reduced shape set for the impairment ladders (runtime bound, disclosed).
const IMPAIR_SHAPES: [(u32, u8); 3] = [(160, 1), (320, 1), (160, 2)];
/// Rates the impairment ladders run at.
const IMPAIR_RATES: [u32; 2] = [8_000, 16_000];
/// Mandatory random-loss points (per mille).
const LOSS_PER_MILLE: [u32; 5] = [0, 10, 30, 50, 100];
/// Deterministic burst means.
const BURST_MEANS: [u32; 3] = [2, 4, 8];
/// Jitter ladder (uniform half-width, microseconds).
const JITTER_US: [i64; 5] = [0, 1_000, 2_500, 5_000, 10_000];
/// Clock-drift ladder (parts per million).
const DRIFT_PPM: [i32; 4] = [-100, -20, 20, 100];
/// Opus requested rates (kbps).
const OPUS_RATES: [u32; 5] = [6, 8, 12, 16, 24];
/// EVS native-mode rates (bits per second).
const EVS_RATES: [u32; 7] = [5_900, 7_200, 8_000, 9_600, 13_200, 16_400, 24_400];
/// Lyra rates (bits per second).
const LYRA_RATES: [u32; 3] = [3_200, 6_000, 9_200];
/// The constitution's network test parameter: 10 ms one-way.
const NOMINAL_NETWORK_US: i64 = 10_000;
/// Bound on the samples a case may contribute.
const MAX_SAMPLES: usize = 16_000 * 12;
/// Fixed impairment seed. Every result replays exactly from it.
const SEED: u64 = 0x7C5E_ED01;
/// Alignment search bound in samples.
const ALIGN_LAG: i64 = 512;
/// Segmental-SNR segment for the recovery definition (20 ms at 16 kHz).
const RECOVERY_SEGMENT: usize = 320;
/// Segmental SNR that counts as recovered.
const RECOVERY_SNR_DB: f64 = 20.0;
/// Hold time required for the recovery definition, in segments.
const RECOVERY_HOLD_SEGMENTS: usize = 1;

// ---------------------------------------------------------------------------
// Tool discovery
// ---------------------------------------------------------------------------

pub(crate) struct VoiceTools {
    pub(crate) media: media::Tools,
    pub(crate) evs_cod: Option<PathBuf>,
    pub(crate) evs_dec: Option<PathBuf>,
}

impl VoiceTools {
    pub(crate) fn discover() -> VoiceTools {
        let r = media::research_root();
        let evs_cod = r.join("clean-evs-master/EVS_cod");
        let evs_dec = r.join("clean-evs-master/EVS_dec");
        VoiceTools {
            media: media::Tools::discover(),
            evs_cod: media::tool_available(&evs_cod).then_some(evs_cod),
            evs_dec: media::tool_available(&evs_dec).then_some(evs_dec),
        }
    }
}

// ---------------------------------------------------------------------------
// EVS (external; G.192 with the bad-frame erasure flag)
// ---------------------------------------------------------------------------

/// One G.192 frame: its byte offset and its data-bit count.
struct EvsFrame {
    offset: usize,
    bits: usize,
}

/// Walk a G.192 bitstream into its frames.
fn evs_frames(bytes: &[u8]) -> Vec<EvsFrame> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 4 <= bytes.len() {
        let bits = usize::from(u16::from_le_bytes([bytes[pos + 2], bytes[pos + 3]]));
        let stride = 4 + bits * 2;
        if pos + stride > bytes.len() {
            break;
        }
        out.push(EvsFrame { offset: pos, bits });
        pos += stride;
    }
    out
}

/// Mark a set of frames as received-in-error using the standard's own erasure
/// sync word. No EVS code is modified; this is the format's documented loss
/// simulation.
fn evs_apply_loss(bytes: &[u8], lost: &[bool]) -> Vec<u8> {
    let mut out = bytes.to_vec();
    for (i, f) in evs_frames(bytes).iter().enumerate() {
        if lost.get(i).copied().unwrap_or(false) {
            out[f.offset] = 0x20;
            out[f.offset + 1] = 0x6b;
        }
    }
    out
}

fn evs_encode(tools: &VoiceTools, raw16k: &Path, work: &Path, bps: u32) -> Option<PathBuf> {
    let cod = tools.evs_cod.as_ref()?;
    let bit = work.join(format!("evs-{bps}.192"));
    let _ = std::fs::remove_file(&bit);
    let ok = media::quiet(
        Command::new(cod)
            .arg("-q")
            .arg(bps.to_string())
            .arg("16")
            .arg(raw16k)
            .arg(&bit),
    )
    .status()
    .ok()?
    .success();
    (ok && bit.is_file()).then_some(bit)
}

fn evs_decode(tools: &VoiceTools, bit: &Path, work: &Path, tag: &str) -> Option<Vec<i32>> {
    let dec = tools.evs_dec.as_ref()?;
    let out = work.join(format!("evs-{tag}.16k"));
    let _ = std::fs::remove_file(&out);
    let ok = media::quiet(Command::new(dec).arg("-q").arg("16").arg(bit).arg(&out))
        .status()
        .ok()?
        .success();
    if !ok || !out.is_file() {
        return None;
    }
    let b = std::fs::read(&out).ok()?;
    Some(
        b.as_chunks::<2>()
            .0
            .iter()
            .map(|c| i32::from(i16::from_le_bytes([c[0], c[1]])))
            .collect(),
    )
}

/// Write a raw 16-bit little-endian mono stream, which is what EVS reads.
pub(crate) fn write_raw16(path: &Path, samples: &[i32]) -> Result<()> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        out.extend_from_slice(&(s.clamp(-32_768, 32_767) as i16).to_le_bytes());
    }
    std::fs::write(path, out).map_err(Error::io)
}

// ---------------------------------------------------------------------------
// Small numeric helpers
// ---------------------------------------------------------------------------

pub(crate) fn percentile(sorted: &[i64], p: f64) -> i64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * p / 100.0).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Segmental SNR in dB over fixed 20 ms segments, per segment.
fn segmental_snr_db(reference: &[i32], test: &[i32], segment: usize) -> Vec<f64> {
    let n = reference.len().min(test.len());
    let mut out = Vec::with_capacity(n / segment + 1);
    let mut off = 0usize;
    while off + segment <= n {
        let mut sig = 0.0f64;
        let mut err = 0.0f64;
        for i in off..off + segment {
            let r = f64::from(reference[i]);
            let t = f64::from(test[i]);
            sig += r * r;
            err += (r - t) * (r - t);
        }
        out.push(if err <= 0.0 {
            f64::INFINITY
        } else if sig <= 0.0 {
            f64::NEG_INFINITY
        } else {
            10.0 * (sig / err).log10()
        });
        off += segment;
    }
    out
}

/// The frozen recovery definition of `docs/PHASE_7C.md` §7.2: milliseconds after
/// the last concealed sample until 20 ms segmental SNR against the **no-loss
/// reference** reaches 20 dB and stays there for the declared hold.
///
/// The no-loss reference is the same codec configuration decoded with no
/// impairment, not the source PCM: recovery asks how fast the decoder returns to
/// its own clean trajectory, which would otherwise be unmeasurable whenever the
/// codec's own clean quality sits near the threshold.
fn recovery_ms(
    clean: &[i32],
    impaired: &[i32],
    last_concealed_frame: u64,
    samples_per_frame: usize,
    rate: u32,
) -> Option<f64> {
    let segment = RECOVERY_SEGMENT;
    let resume_at = ((last_concealed_frame as usize + 1) * samples_per_frame).min(clean.len());
    let segs = segmental_snr_db(clean, impaired, segment);
    let first = resume_at.div_ceil(segment);
    for i in first..segs.len() {
        let hold_ok = segs[i] >= RECOVERY_SNR_DB
            && (0..=RECOVERY_HOLD_SEGMENTS)
                .all(|k| segs.get(i + k).is_some_and(|&v| v >= RECOVERY_SNR_DB));
        if hold_ok {
            let ms = (i * segment).saturating_sub(resume_at) as f64 * 1000.0 / f64::from(rate);
            return Some(ms);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

pub(crate) struct Case {
    pub(crate) id: String,
    pub(crate) samples: Vec<i32>,
}

/// Conversation-scale cases are built by concatenating consecutive frozen
/// clips. A single `effectiveness` clip is about a second long — far too short
/// to measure concealment, jitter tolerance or clock behaviour — and a voice
/// codec's constitution is a *call*, not a phoneme. The concatenation is
/// deterministic (manifest order) and adds no separator, so no artificial
/// silence is injected into the active-speech rate.
const CASE_TARGET_SAMPLES: usize = 16_000 * 4;
/// Clips read to build the cases.
const CASE_CLIP_POOL: usize = 40;

fn cases() -> Result<Vec<Case>> {
    if !crate::learned::corpus_real::available() {
        return Err(Error::new(
            crate::error::Kind::Unavailable,
            "the voice court needs the frozen real speech corpus",
        ));
    }
    cases_from(
        &crate::learned::corpus_real::effectiveness_clips(),
        CASE_CLIP_POOL,
        CASE_TARGET_SAMPLES,
    )
}

/// Build conversation-scale cases by concatenating consecutive clips of a
/// provided frozen corpus, in manifest order, with no injected separator.
/// Deterministic, so a challenger court can use a different split without
/// changing the construction rule.
pub(crate) fn cases_from(
    clips: &[crate::learned::corpus_real::RealClip],
    pool: usize,
    target: usize,
) -> Result<Vec<Case>> {
    let scratch = PathBuf::from("target/voice-court/scratch");
    let loaded = crate::learned::corpus_real::load_cases(clips, pool, &scratch)?;
    let mut groups: Vec<(String, Vec<i32>)> = Vec::new();
    for c in loaded {
        let ch = usize::from(c.clip.channels);
        if ch != 1 || c.clip.sample_rate_hz != 16_000 {
            // The constitution fixes 16 kHz mono; a clip outside that domain is
            // skipped rather than resampled silently.
            continue;
        }
        match groups.last_mut() {
            Some((_, acc)) if acc.len() < target => {
                let want = (target - acc.len()).min(c.samples.len());
                acc.extend_from_slice(&c.samples[..want]);
            }
            _ => groups.push((c.clip.id.clone(), c.samples.clone())),
        }
    }
    let out: Vec<Case> = groups
        .into_iter()
        .map(|(id, samples)| {
            let frames = samples.len().min(MAX_SAMPLES);
            Case {
                id,
                samples: samples[..frames].to_vec(),
            }
        })
        .collect();
    if out.is_empty() {
        return Err(Error::new(
            crate::error::Kind::Unavailable,
            "no 16 kHz mono speech clip in the frozen corpus",
        ));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// VOLE measurement
// ---------------------------------------------------------------------------

pub(crate) struct CleanRun {
    pub(crate) bytes: usize,
    pub(crate) samples: Vec<i32>,
    pub(crate) encode_ns: Vec<i64>,
    pub(crate) decode_ns: Vec<i64>,
    pub(crate) inactive: usize,
    pub(crate) active_bytes: usize,
    pub(crate) inactive_bytes: usize,
}

/// The constitution's default configuration for one operating point.
///
/// `redundancy` is **off by default**. The court prices it (7C.7) and the
/// measurement is a rejection for this profile: backward redundancy spends
/// frame bytes that the residual needs, and it can only repair a frame a jitter
/// buffer has not yet released, which this codec's measured depth does not
/// create. The mechanism is retained and priced rather than deleted, so a
/// future depth or delay change can re-open the question on evidence.
pub(crate) fn config_for(
    frame_len: u32,
    frames_per_packet: u8,
    bps: u32,
    dtx: bool,
) -> VoiceConfig {
    VoiceConfig {
        sample_rate_hz: 16_000,
        frame_len,
        frames_per_packet,
        target_bits_per_second: bps,
        dtx,
        capsule_cadence: voice::DEFAULT_CAPSULE_CADENCE,
        redundancy: false,
    }
}

/// Encode and cleanly decode one configuration, timing every packet.
pub(crate) fn run_clean(cfg: VoiceConfig, source: &[i32]) -> Result<CleanRun> {
    let n = cfg.packet_samples();
    let mut enc = VoiceEncoder::new(cfg)?;
    let mut dec = VoiceDecoder::new(cfg)?;
    let mut packets = Vec::new();
    let mut bytes = 0usize;
    let mut encode_ns = Vec::new();
    let mut decode_ns = Vec::new();
    let mut out = Vec::with_capacity(source.len());
    let mut inactive = 0usize;
    let mut active_bytes = 0usize;
    let mut inactive_bytes = 0usize;
    for chunk in source.chunks(n) {
        if chunk.len() < n {
            break;
        }
        let sw = Stopwatch::start();
        let pkt = enc.encode_packet(chunk)?;
        encode_ns.push(sw.elapsed_ns());
        let is_inactive = voice::packet_is_inactive(&pkt);
        if is_inactive {
            inactive += 1;
            inactive_bytes += pkt.len();
        } else {
            active_bytes += pkt.len();
        }
        bytes += pkt.len();
        let sw = Stopwatch::start();
        let (_kind, samples) = dec.decode_packet(&pkt)?;
        decode_ns.push(sw.elapsed_ns());
        out.extend_from_slice(&samples);
        packets.push(pkt);
    }
    Ok(CleanRun {
        bytes,
        samples: out,
        encode_ns,
        decode_ns,
        inactive,
        active_bytes,
        inactive_bytes,
    })
}

/// The impairment ladders for one configuration.
struct ImpairmentReport {
    loss: Vec<serde_json::Value>,
    burst: Vec<serde_json::Value>,
    jitter: Vec<serde_json::Value>,
    drift: Vec<serde_json::Value>,
    mixed: Vec<serde_json::Value>,
}

fn impairment_report(
    cfg: VoiceConfig,
    source: &[i32],
    clean: &CleanRun,
) -> Result<ImpairmentReport> {
    let n = cfg.packet_samples();
    let frames_per_packet = usize::from(cfg.frames_per_packet);
    let frame_duration_us = i64::from(cfg.frame_len) * 1_000_000 / i64::from(cfg.sample_rate_hz);
    let frames = (source.len() / n) as u64 * frames_per_packet as u64;

    // One encode pass; the impairment engine reuses the packets, exactly as a
    // real sender would.
    let mut enc = VoiceEncoder::new(cfg)?;
    let mut packets: Vec<TransportPacket> = Vec::new();
    let mut encode_us: Vec<i64> = Vec::new();
    for chunk in source.chunks(n) {
        if chunk.len() < n {
            break;
        }
        let sw = Stopwatch::start();
        let payload = enc.encode_packet(chunk)?;
        encode_us.push((sw.elapsed_ns() / 1_000).max(0));
        packets.push(TransportPacket {
            seq: packets.len() as u32,
            payload,
        });
    }

    let run = |imp: Impairment, depth: i64| -> Result<(SimStats, Vec<i32>, Option<f64>, i64)> {
        let arrivals = impair::impair(
            &packets,
            &imp,
            frame_duration_us,
            frames_per_packet,
            &encode_us,
            NOMINAL_NETWORK_US,
        );
        let depth_needed = impair::required_depth_us(
            &arrivals,
            frames_per_packet,
            frame_duration_us,
            NOMINAL_NETWORK_US,
            imp.drift_ppm,
        );
        let used_depth = if depth < 0 { depth_needed } else { depth };
        let (events, stats) = impair::schedule(
            frames,
            &arrivals,
            frames_per_packet,
            frame_duration_us,
            NOMINAL_NETWORK_US,
            used_depth,
            imp.drift_ppm,
        );
        let mut dec = VoiceDecoder::new(cfg)?;
        let mut out = Vec::with_capacity(source.len());
        let mut last_concealed: Option<u64> = None;
        // A frame decodes from *any* delivered copy: the primary if it arrived,
        // otherwise a redundancy copy. Skipping duplicates here would make the
        // scheduler's "delivered" verdict disagree with the decoder's, which is
        // precisely the case duplication exists to cover.
        let mut by_seq: Vec<Option<&Arrival>> = vec![None; packets.len()];
        for a in &arrivals {
            if a.lost {
                continue;
            }
            let slot = &mut by_seq[a.seq as usize];
            if slot.is_none_or(|cur| a.arrival_us < cur.arrival_us) {
                *slot = Some(a);
            }
        }
        for e in &events {
            let p = (e.frame / frames_per_packet as u64) as usize;
            if e.concealed {
                last_concealed = Some(e.frame);
                out.extend_from_slice(&dec.conceal());
            } else {
                let a = by_seq[p].expect("a scheduled packet must have arrived");
                let (_k, samples) = dec.decode_packet(&a.payload)?;
                out.extend_from_slice(&samples);
            }
        }
        let recovery = last_concealed.and_then(|f| {
            recovery_ms(
                &clean.samples,
                &out,
                f,
                cfg.frame_len as usize,
                cfg.sample_rate_hz,
            )
        });
        Ok((stats, out, recovery, depth_needed))
    };

    let mut loss = Vec::new();
    for &lm in &LOSS_PER_MILLE {
        let imp = Impairment {
            seed: SEED,
            loss_per_mille: lm,
            ..Impairment::default()
        };
        let (stats, out, recovery, depth) = run(imp, -1)?;
        let (snr, identical) = degradation(&clean.samples, &out);
        loss.push(serde_json::json!({
            "loss_per_mille": lm,
            "concealed": stats.frames_concealed,
            "frames": stats.frames,
            "late": stats.frames_late,
            "duplicates_discarded": stats.duplicates_discarded,
            "required_depth_us": depth,
            "snr_db": snr,
            "identical_to_clean": identical,
            "recovery_ms": recovery,
        }));
    }

    let mut burst = Vec::new();
    for &b in &BURST_MEANS {
        let imp = Impairment {
            seed: SEED,
            loss_per_mille: 30,
            burst_mean: b,
            ..Impairment::default()
        };
        let (stats, out, recovery, depth) = run(imp, -1)?;
        let (snr, identical) = degradation(&clean.samples, &out);
        burst.push(serde_json::json!({
            "burst_mean": b,
            "loss_per_mille": 30,
            "concealed": stats.frames_concealed,
            "longest_run": longest_conceal_run(&imp, &packets, frames_per_packet, frame_duration_us, &stats),
            "required_depth_us": depth,
            "snr_db": snr,
            "identical_to_clean": identical,
            "recovery_ms": recovery,
        }));
    }

    let mut jitter = Vec::new();
    for &j in &JITTER_US {
        let imp = Impairment {
            seed: SEED,
            jitter_us: j,
            ..Impairment::default()
        };
        let (stats, out, recovery, depth) = run(imp, -1)?;
        let (snr, identical) = degradation(&clean.samples, &out);
        jitter.push(serde_json::json!({
            "jitter_us": j,
            "two_mode": false,
            "required_depth_us": depth,
            "concealed": stats.frames_concealed,
            "late": stats.frames_late,
            "snr_db": snr,
            "identical_to_clean": identical,
            "recovery_ms": recovery,
        }));
    }
    {
        let imp = Impairment {
            seed: SEED,
            jitter_us: 5_000,
            jitter_two_mode: true,
            ..Impairment::default()
        };
        let (stats, out, recovery, depth) = run(imp, -1)?;
        let (snr, identical) = degradation(&clean.samples, &out);
        jitter.push(serde_json::json!({
            "jitter_us": 5_000,
            "two_mode": true,
            "required_depth_us": depth,
            "concealed": stats.frames_concealed,
            "late": stats.frames_late,
            "snr_db": snr,
            "identical_to_clean": identical,
            "recovery_ms": recovery,
        }));
    }

    let mut drift = Vec::new();
    for &ppm in &DRIFT_PPM {
        let imp = Impairment {
            seed: SEED,
            drift_ppm: ppm,
            ..Impairment::default()
        };
        let (stats, out, recovery, depth) = run(imp, -1)?;
        let (snr, identical) = degradation(&clean.samples, &out);
        drift.push(serde_json::json!({
            "drift_ppm": ppm,
            "required_depth_us": depth,
            "concealed": stats.frames_concealed,
            "late": stats.frames_late,
            "headroom_us": stats.max_headroom_us,
            "snr_db": snr,
            "identical_to_clean": identical,
            "recovery_ms": recovery,
        }));
    }

    let mut mixed = Vec::new();
    for (label, imp) in [
        (
            "reorder_20pm",
            Impairment {
                seed: SEED,
                loss_per_mille: 30,
                reorder_per_mille: 20,
                jitter_us: 2_500,
                ..Impairment::default()
            },
        ),
        (
            "duplicate_50pm",
            Impairment {
                seed: SEED,
                loss_per_mille: 30,
                duplicate_per_mille: 50,
                ..Impairment::default()
            },
        ),
        (
            "all_on",
            Impairment {
                seed: SEED,
                loss_per_mille: 30,
                burst_mean: 3,
                duplicate_per_mille: 20,
                reorder_per_mille: 20,
                jitter_us: 5_000,
                jitter_two_mode: true,
                drift_ppm: 20,
            },
        ),
    ] {
        let (stats, out, recovery, depth) = run(imp, -1)?;
        let (snr, identical) = degradation(&clean.samples, &out);
        mixed.push(serde_json::json!({
            "pattern": label,
            "concealed": stats.frames_concealed,
            "required_depth_us": depth,
            "snr_db": snr,
            "identical_to_clean": identical,
            "recovery_ms": recovery,
        }));
    }

    Ok(ImpairmentReport {
        loss,
        burst,
        jitter,
        drift,
        mixed,
    })
}

pub(crate) fn snr_aligned(reference: &[i32], test: &[i32]) -> f64 {
    let lag = media::best_lag(reference, test, ALIGN_LAG);
    let (a, b) = media::aligned_pair(reference, test, lag);
    media::snr_db(&a, &b)
}

/// Degradation of an impaired run against its clean control run.
///
/// A run with no concealed frames reproduces the clean trajectory bit for bit,
/// which is infinite SNR. Reporting `null` alone would look like a missing
/// measurement, so the boolean says *why* the number is absent.
fn degradation(clean: &[i32], out: &[i32]) -> (Option<f64>, bool) {
    let lag = media::best_lag(clean, out, ALIGN_LAG);
    let (a, b) = media::aligned_pair(clean, out, lag);
    let identical = a == b;
    (media::finite(media::snr_db(&a, &b)), identical)
}

/// The longest consecutive run of concealed frames in a realisation.
fn longest_conceal_run(
    imp: &Impairment,
    packets: &[TransportPacket],
    frames_per_packet: usize,
    frame_duration_us: i64,
    stats: &SimStats,
) -> u64 {
    let arrivals = impair::impair(
        packets,
        imp,
        frame_duration_us,
        frames_per_packet,
        &vec![0i64; packets.len()],
        NOMINAL_NETWORK_US,
    );
    let (events, _) = impair::schedule(
        stats.frames,
        &arrivals,
        frames_per_packet,
        frame_duration_us,
        NOMINAL_NETWORK_US,
        0,
        imp.drift_ppm,
    );
    let mut best = 0u64;
    let mut run = 0u64;
    for e in &events {
        if e.concealed {
            run += 1;
            best = best.max(run);
        } else {
            run = 0;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Court
// ---------------------------------------------------------------------------

fn latency_json(
    cfg: &VoiceConfig,
    encode_ns: &[i64],
    decode_ns: &[i64],
    depth_us: i64,
) -> serde_json::Value {
    let mut enc_sorted = encode_ns.to_vec();
    enc_sorted.sort_unstable();
    let mut dec_sorted = decode_ns.to_vec();
    dec_sorted.sort_unstable();
    let frame_accumulation_us =
        i64::from(cfg.frame_len) * i64::from(cfg.frames_per_packet) * 1_000_000
            / i64::from(cfg.sample_rate_hz);
    let breakdown = LatencyBreakdown {
        frame_accumulation_us,
        lookahead_us: 0,
        encode_us: percentile(&enc_sorted, 50.0) / 1_000,
        packetisation_us: 0,
        network_us: NOMINAL_NETWORK_US,
        jitter_buffer_us: depth_us,
        decode_us: percentile(&dec_sorted, 50.0) / 1_000,
        playout_scheduling_us: 0,
    };
    serde_json::json!({
        "frame_accumulation_us": breakdown.frame_accumulation_us,
        "lookahead_us": breakdown.lookahead_us,
        "encode_p50_us": percentile(&enc_sorted, 50.0) / 1_000,
        "encode_p90_us": percentile(&enc_sorted, 90.0) / 1_000,
        "encode_p95_us": percentile(&enc_sorted, 95.0) / 1_000,
        "encode_p99_us": percentile(&enc_sorted, 99.0) / 1_000,
        "encode_max_us": enc_sorted.last().copied().unwrap_or(0) / 1_000,
        "encode_deadline_misses": encode_ns.iter().filter(|&&v| v > 5_000_000).count(),
        "decode_p50_us": percentile(&dec_sorted, 50.0) / 1_000,
        "decode_p90_us": percentile(&dec_sorted, 90.0) / 1_000,
        "decode_p95_us": percentile(&dec_sorted, 95.0) / 1_000,
        "decode_p99_us": percentile(&dec_sorted, 99.0) / 1_000,
        "decode_max_us": dec_sorted.last().copied().unwrap_or(0) / 1_000,
        "decode_deadline_misses": decode_ns.iter().filter(|&&v| v > 5_000_000).count(),
        "network_us": breakdown.network_us,
        "one_way_p50_us": breakdown.one_way_us(),
        "envelope": "20 ms processing + network envelope (encode ≤ 5 ms, network test 10 ms, decode ≤ 5 ms)",
        "lookahead_disclosure": "analysis reads the reconstructed history and the current frame only; no future sample is touched",
    })
}

/// Run the court; writes `receipts/learned-voice-stream/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.stream.voice.court.v1");
    common::push_label(&mut projection, voice::VOICE_PROFILE);

    let tools = VoiceTools::discover();
    let work = PathBuf::from("target/voice-court");
    std::fs::create_dir_all(&work).map_err(Error::io)?;

    let corpus = cases()?;
    let mut all_ok = true;
    let mut rows = Vec::new();
    let mut clean_opus_deltas: Vec<f64> = Vec::new();
    let mut clean_evs_deltas: Vec<f64> = Vec::new();
    let mut clean_lyra_deltas: Vec<f64> = Vec::new();
    let mut loss_snrs: Vec<f64> = Vec::new();
    let mut one_way_p99: Vec<i64> = Vec::new();
    let mut encode_p99: Vec<i64> = Vec::new();

    for case in &corpus {
        let ref_wav = work.join(format!("ref-{}.wav", case.id));
        media::write_wav(&ref_wav, &media::to_i16(&case.samples), 1, 16_000)?;
        let raw = work.join(format!("ref-{}.16k", case.id));
        write_raw16(&raw, &case.samples)?;
        let seconds = case.samples.len() as f64 / 16_000.0;

        // ---- VOLE clean ladder -------------------------------------------------
        let mut vole_clean: Vec<serde_json::Value> = Vec::new();
        let mut vole_points: Vec<media::Point> = Vec::new();
        let mut clean_by_shape: Vec<(usize, VoiceConfig, CleanRun)> = Vec::new();
        for (si, &(frame_len, fpp)) in SHAPES.iter().enumerate() {
            for &bps in &VOLE_RATES {
                let cfg = config_for(frame_len, fpp, bps, false);
                let run = run_clean(cfg, &case.samples)?;
                let actual_bps = run.bytes as f64 * 8.0 / seconds;
                let snr = snr_aligned(&case.samples, &run.samples);
                let lag = media::best_lag(&case.samples, &run.samples, ALIGN_LAG);
                let (a, b) = media::aligned_pair(&case.samples, &run.samples, lag);
                let spec = media::spectral_distortion_db(&a, &b);
                let trans = media::transient_error_db(&a, &b);
                let mos = run_visqol(&tools, &work, &case.id, &a, &b);
                if run.bytes == 0 || snr.is_nan() {
                    all_ok = false;
                }
                common::push_label(&mut projection, &case.id);
                common::push_label(&mut projection, &format!("shape-{frame_len}x{fpp}"));
                common::push_u64(&mut projection, u64::from(bps));
                common::push_u64(&mut projection, run.bytes as u64);
                // An exact reconstruction has infinite SNR; freeze a finite cap.
                common::push_u64(&mut projection, (snr.clamp(-100.0, 200.0) * 100.0) as u64);
                let bytes = run.bytes;
                vole_points.push(media::Point {
                    bps: actual_bps,
                    snr,
                    visqol: mos,
                });
                for ns in &run.encode_ns {
                    encode_p99.push(*ns / 1_000);
                }
                let latency = latency_json(&cfg, &run.encode_ns, &run.decode_ns, 0);
                if let Some(v) = latency["decode_p99_us"].as_i64() {
                    one_way_p99.push(v);
                }
                clean_by_shape.push((si, cfg, run));
                rows.push(serde_json::json!({
                    "case": case.id,
                    "kind": "clean",
                    "frame_len": frame_len,
                    "frames_per_packet": fpp,
                    "target_bps": bps,
                    "actual_bps": actual_bps,
                    "bytes": bytes,
                    "snr_db": media::finite(snr),
                    "spectral_db": media::finite(spec),
                    "transient_db": media::finite(trans),
                    "visqol_mos": mos,
                    "latency": latency,
                }));
                vole_clean.push(serde_json::json!({
                    "frame_len": frame_len,
                    "frames_per_packet": fpp,
                    "target_bps": bps,
                    "actual_bps": actual_bps,
                }));
            }
        }
        vole_points.sort_by(|a, b| a.bps.partial_cmp(&b.bps).unwrap());

        // ---- Competitors: clean ladder ----------------------------------------
        let opus = opus_points(&tools, &ref_wav, &work, &case.samples);
        let evs = evs_points(&tools, &raw, &work, &case.samples);
        let lyra = lyra_points(&tools, &ref_wav, &work, seconds, &case.samples);

        // Matched-actual-bitrate comparison for the clean ladder.
        let matched = serde_json::json!({
            "opus": matched_summary(&vole_points, &opus),
            "evs": matched_summary(&vole_points, &evs),
            "lyra": matched_summary(&vole_points, &lyra),
        });
        if let Some(d) = matched["opus"]["snr_delta_mean_db"].as_f64() {
            clean_opus_deltas.push(d);
        }
        if let Some(d) = matched["evs"]["snr_delta_mean_db"].as_f64() {
            clean_evs_deltas.push(d);
        }
        if let Some(d) = matched["lyra"]["snr_delta_mean_db"].as_f64() {
            clean_lyra_deltas.push(d);
        }

        // ---- Impairment ladders ------------------------------------------------
        let mut impairment = Vec::new();
        for (frame_len, fpp) in IMPAIR_SHAPES {
            for &bps in &IMPAIR_RATES {
                let cfg = config_for(frame_len, fpp, bps, false);
                let idx = clean_by_shape
                    .iter()
                    .position(|(_, c, _)| {
                        c.frame_len == frame_len
                            && c.frames_per_packet == fpp
                            && c.target_bits_per_second == bps
                    })
                    .ok_or_else(|| Error::internal("missing clean control run"))?;
                let clean = &clean_by_shape[idx].2;
                let report = impairment_report(cfg, &case.samples, clean)?;
                for point in report.loss.iter() {
                    if let Some(v) = point["snr_db"].as_f64() {
                        loss_snrs.push(v);
                    }
                }
                impairment.push(serde_json::json!({
                    "frame_len": frame_len,
                    "frames_per_packet": fpp,
                    "target_bps": bps,
                    "clean_snr_db": media::finite(snr_aligned(&case.samples, &clean.samples)),
                    "loss": report.loss,
                    "burst": report.burst,
                    "jitter": report.jitter,
                    "drift": report.drift,
                    "mixed": report.mixed,
                }));
            }
        }

        // ---- DTX ---------------------------------------------------------------
        let mut dtx = Vec::new();
        for (frame_len, fpp) in IMPAIR_SHAPES {
            let cfg = config_for(frame_len, fpp, 8_000, true);
            let run = run_clean(cfg, &case.samples)?;
            let packets = (case.samples.len() / cfg.packet_samples()).max(1) as f64;
            let active_packets = packets - run.inactive as f64;
            let active_bps = if active_packets > 0.0 {
                run.active_bytes as f64 * 8.0
                    / (active_packets * cfg.packet_samples() as f64 / 16_000.0)
            } else {
                0.0
            };
            let inactive_bps = if run.inactive > 0 {
                run.inactive_bytes as f64 * 8.0
                    / (run.inactive as f64 * cfg.packet_samples() as f64 / 16_000.0)
            } else {
                0.0
            };
            common::push_label(&mut projection, &format!("dtx-{frame_len}x{fpp}"));
            common::push_u64(&mut projection, run.active_bytes as u64);
            common::push_u64(&mut projection, run.inactive_bytes as u64);
            dtx.push(serde_json::json!({
                "frame_len": frame_len,
                "frames_per_packet": fpp,
                "target_bps": 8_000,
                "inactive_packets": run.inactive,
                "whole_call_bps": run.bytes as f64 * 8.0 / seconds,
                "active_bps": active_bps,
                "inactive_bps": inactive_bps,
                "note": "active, inactive and whole-call rates are reported separately; \
                         long silence is never allowed to flatter the active codec",
            }));
        }

        // ---- Redundancy: what it costs and what it buys (7C.7) ----------------
        let mut redundancy = Vec::new();
        for &(frame_len, fpp) in &[(320u32, 1u8), (160, 1)] {
            for &bps in &[8_000u32, 16_000] {
                let mut accrued = Vec::new();
                for enabled in [false, true] {
                    let cfg = VoiceConfig {
                        redundancy: enabled,
                        ..config_for(frame_len, fpp, bps, false)
                    };
                    let run = run_clean(cfg, &case.samples)?;
                    let impaired = impairment_report(cfg, &case.samples, &run)?;
                    let loss30 = impaired
                        .loss
                        .iter()
                        .find(|v| v["loss_per_mille"] == serde_json::json!(30))
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    accrued.push(serde_json::json!({
                        "redundancy": enabled,
                        "bytes": run.bytes,
                        "actual_bps": run.bytes as f64 * 8.0 / seconds,
                        "snr_db": media::finite(snr_aligned(&case.samples, &run.samples)),
                        "loss_30pm": loss30,
                    }));
                }
                redundancy.push(serde_json::json!({
                    "frame_len": frame_len,
                    "frames_per_packet": fpp,
                    "target_bps": bps,
                    "variants": accrued,
                    "note": "redundancy overhead is priced separately from the primary codec rate, \
                             and its benefit is the loss-30 cell of the same pair",
                }));
            }
        }

        // ---- EVS under loss ----------------------------------------------------
        let evs_loss = evs_loss_ladder(&tools, &raw, &work, &case.samples);

        rows.push(serde_json::json!({
            "case": case.id,
            "kind": "summary",
            "vole_clean": vole_clean,
            "opus": opus.iter().map(point_json).collect::<Vec<_>>(),
            "evs": evs.iter().map(point_json).collect::<Vec<_>>(),
            "lyra": lyra.iter().map(point_json).collect::<Vec<_>>(),
            "matched": matched,
            "impairment": impairment,
            "dtx": dtx,
            "redundancy": redundancy,
            "evs_loss": evs_loss,
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
    encode_p99.sort_unstable();
    one_way_p99.sort_unstable();

    let verdict = if all_ok {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };

    common::finish_with_profile(
        "learned-voice-stream",
        crate::learned::profile::LearnedProfile::Exp3,
        receipts_root,
        LEARNED_VOICE_STREAM_SHA256,
        &projection,
        verdict,
        format!(
            "voice codec {} over {} real speech cases: clean matched-bitrate SNR {} of {} cases vs \
             Opus (mean {:+.2} dB), {} of {} vs EVS (mean {:+.2} dB), {} of {} vs Lyra (mean {:+.2} \
             dB); {} impaired cells; encode p99 {} µs; the profile is lossy and makes no exactness \
             claim",
            voice::VOICE_PROFILE,
            corpus.len(),
            wins(&clean_opus_deltas),
            clean_opus_deltas.len(),
            mean(&clean_opus_deltas),
            wins(&clean_evs_deltas),
            clean_evs_deltas.len(),
            mean(&clean_evs_deltas),
            wins(&clean_lyra_deltas),
            clean_lyra_deltas.len(),
            mean(&clean_lyra_deltas),
            loss_snrs.len(),
            percentile(&encode_p99, 99.0),
        ),
        vec![
            (
                "harness",
                serde_json::json!({
                    "profile": voice::VOICE_PROFILE,
                    "constitution": "docs/PHASE_7C.md",
                    "opus": tools.media.opusenc.as_ref().map(|p| p.to_string_lossy().to_string()),
                    "evs": tools.evs_cod.as_ref().map(|p| p.to_string_lossy().to_string()),
                    "lyra": tools.media.lyra_enc.as_ref().map(|p| p.to_string_lossy().to_string()),
                    "visqol": tools.media.visqol.as_ref().map(|p| p.to_string_lossy().to_string()),
                    "shapes": SHAPES.iter().map(|(f, n)| format!("{f}x{n}")).collect::<Vec<_>>(),
                    "vole_rates": VOLE_RATES,
                    "loss_per_mille": LOSS_PER_MILLE,
                    "burst_means": BURST_MEANS,
                    "jitter_us": JITTER_US,
                    "drift_ppm": DRIFT_PPM,
                    "seed": SEED,
                    "nominal_network_us": NOMINAL_NETWORK_US,
                    "align_lag": ALIGN_LAG,
                    "impair_shapes": IMPAIR_SHAPES.iter().map(|(f, n)| format!("{f}x{n}")).collect::<Vec<_>>(),
                    "impair_rates": IMPAIR_RATES,
                    "note": "competitors are external processes; no code is imported, linked, wrapped \
                             or used as a fallback, and no competitor payload enters a VOLE object",
                }),
            ),
            (
                "matched_bitrate_summary",
                serde_json::json!({
                    "opus_snr_cases": clean_opus_deltas.len(),
                    "opus_snr_wins": wins(&clean_opus_deltas),
                    "opus_snr_mean_db": mean(&clean_opus_deltas),
                    "evs_snr_cases": clean_evs_deltas.len(),
                    "evs_snr_wins": wins(&clean_evs_deltas),
                    "evs_snr_mean_db": mean(&clean_evs_deltas),
                    "lyra_snr_cases": clean_lyra_deltas.len(),
                    "lyra_snr_wins": wins(&clean_lyra_deltas),
                    "lyra_snr_mean_db": mean(&clean_lyra_deltas),
                    "impaired_cells": loss_snrs.len(),
                    "impaired_snr_mean_db": mean(&loss_snrs),
                    "encode_p99_us": percentile(&encode_p99, 99.0),
                    "encode_max_us": encode_p99.last().copied().unwrap_or(0),
                    "decode_p99_us": percentile(&one_way_p99, 99.0),
                }),
            ),
            ("cases", serde_json::json!(rows)),
            (
                "limitations",
                serde_json::json!([
                    "the profile is lossy: `decode(encode(x)) == x` is explicitly not claimed and no \
                     exactness metric is reported for it",
                    "the recovery definition is frozen in docs/PHASE_7C.md §7.2 before measurement: \
                     20 dB segmental SNR against the no-loss reference, held for 20 ms",
                    "Opus and Lyra are driven only through their pinned CLIs, which offer no packet-loss \
                     simulation; their loss/jitter behaviour is therefore NOT_AVAILABLE rather than guessed",
                    "EVS loss is simulated with the standard's own G.192 bad-frame sync word, which is \
                     the format's documented erasure flag",
                    "impairment ladders run on a reduced shape/rate set (disclosed in `harness`) to bound \
                     runtime; the clean ladder covers every shape and rate",
                    "ViSQOL speech mode is applied where the pinned binary and model run; absent metric \
                     cells are null, never invented",
                    "the frozen effectiveness corpus provides 12.288 s of speech in twelve 1.024 s clips; \
                     the court builds three cases of about 4.1 s. A clock-drift ladder over a few seconds \
                     can only expose drift far larger than its rungs, so a zero-concealment drift cell \
                     means \"not detectable at this length\", not \"drift-immune\"",
                    "the impairment engine models a channel; it is not a network stack and makes no claim \
                     about any real transport",
                ]),
            ),
        ],
    )
}

pub(crate) fn run_visqol(
    tools: &VoiceTools,
    work: &Path,
    case: &str,
    reference: &[i32],
    degraded: &[i32],
) -> Option<f64> {
    let rw = work.join(format!("visqol-ref-{case}.wav"));
    let dw = work.join(format!("visqol-deg-{case}.wav"));
    if media::write_wav(&rw, &media::to_i16(reference), 1, 16_000).is_err() {
        return None;
    }
    if media::write_wav(&dw, &media::to_i16(degraded), 1, 16_000).is_err() {
        return None;
    }
    media::visqol_mos(&tools.media, &rw, &dw, true)
}

fn point_json(p: &media::Point) -> serde_json::Value {
    serde_json::json!({
        "actual_bps": p.bps,
        "snr_db": media::finite(p.snr),
        "visqol_mos": p.visqol,
    })
}

/// ViSQOL for one external codec operating point. Speech mode, because every
/// case in this court is speech.
fn competitor_visqol(
    tools: &VoiceTools,
    work: &Path,
    tag: &str,
    reference: &[i32],
    degraded: &[i32],
) -> Option<f64> {
    tools.media.visqol.as_ref()?;
    let rw = work.join(format!("visqol-cmp-ref-{tag}.wav"));
    let dw = work.join(format!("visqol-cmp-{tag}.wav"));
    media::write_wav(&rw, &media::to_i16(reference), 1, 16_000).ok()?;
    media::write_wav(&dw, &media::to_i16(degraded), 1, 16_000).ok()?;
    media::visqol_mos(&tools.media, &rw, &dw, true)
}

/// Interpolate a competitor's measured curve at each VOLE operating point.
pub(crate) fn matched_summary(
    vole: &[media::Point],
    competitor: &[media::Point],
) -> serde_json::Value {
    if competitor.is_empty() {
        return serde_json::json!({
            "available": false,
            "cells": 0,
            "snr_wins": 0,
            "snr_delta_mean_db": null,
            "cells_detail": [],
        });
    }
    let mut deltas = Vec::new();
    let mut mos_deltas = Vec::new();
    let mut detail = Vec::new();
    for v in vole {
        let c = media::interp(competitor, v.bps, |p| Some(p.snr));
        let d = match (media::finite(v.snr), c.and_then(media::finite)) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        };
        if let Some(d) = d {
            deltas.push(d);
        }
        let c_mos = media::interp(competitor, v.bps, |p| p.visqol);
        let d_mos = match (v.visqol, c_mos.and_then(media::finite)) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        };
        if let Some(d) = d_mos {
            mos_deltas.push(d);
        }
        detail.push(serde_json::json!({
            "vole_actual_bps": v.bps,
            "competitor_snr_at_vole_bps": c.and_then(media::finite),
            "competitor_visqol_at_vole_bps": c_mos.and_then(media::finite),
            "delta_snr_db": d,
            "delta_visqol_mos": d_mos,
        }));
    }
    let mean = |v: &[f64]| {
        if v.is_empty() {
            None
        } else {
            Some(v.iter().sum::<f64>() / v.len() as f64)
        }
    };
    serde_json::json!({
        "available": true,
        "cells": deltas.len(),
        "snr_wins": deltas.iter().filter(|&&d| d > 0.0).count(),
        "snr_delta_mean_db": mean(&deltas),
        "visqol_cells": mos_deltas.len(),
        "visqol_wins": mos_deltas.iter().filter(|&&d| d > 0.0).count(),
        "visqol_delta_mean": mean(&mos_deltas),
        "cells_detail": detail,
    })
}

// ---------------------------------------------------------------------------
// Competitor ladders
// ---------------------------------------------------------------------------

pub(crate) fn opus_points(
    tools: &VoiceTools,
    wav: &Path,
    work: &Path,
    reference: &[i32],
) -> Vec<media::Point> {
    let (Some(enc), Some(dec)) = (tools.media.opusenc.as_ref(), tools.media.opusdec.as_ref())
    else {
        return Vec::new();
    };
    let seconds = reference.len() as f64 / 16_000.0;
    let mut out = Vec::new();
    for &framesize in &[10u32, 20] {
        for &kbps in &OPUS_RATES {
            let opus = work.join(format!("opus-{framesize}-{kbps}.opus"));
            let dec_wav = work.join(format!("opus-{framesize}-{kbps}.wav"));
            let _ = std::fs::remove_file(&opus);
            let ok = media::quiet(
                Command::new(enc)
                    .arg("--quiet")
                    .arg(format!("--bitrate={kbps}"))
                    .arg("--vbr")
                    .arg(format!("--framesize={framesize}"))
                    .arg(wav)
                    .arg(&opus),
            )
            .status()
            .ok()
            .is_some_and(|s| s.success());
            if !ok {
                continue;
            }
            let ok = media::quiet(Command::new(dec).arg("--quiet").arg(&opus).arg(&dec_wav))
                .status()
                .ok()
                .is_some_and(|s| s.success());
            if !ok {
                continue;
            }
            let Ok((samples, _ch, _rate)) = media::read_wav(&dec_wav) else {
                continue;
            };
            let decoded: Vec<i32> = samples.iter().map(|&v| i32::from(v)).collect();
            let lag = media::best_lag(reference, &decoded, ALIGN_LAG);
            let (a, b) = media::aligned_pair(reference, &decoded, lag);
            let size = std::fs::metadata(&opus).map(|m| m.len()).unwrap_or(0) as f64;
            out.push(media::Point {
                bps: size * 8.0 / seconds,
                snr: media::snr_db(&a, &b),
                visqol: competitor_visqol(tools, work, &format!("opus-{framesize}-{kbps}"), &a, &b),
            });
        }
    }
    out.sort_by(|a, b| a.bps.partial_cmp(&b.bps).unwrap());
    out
}

pub(crate) fn evs_points(
    tools: &VoiceTools,
    raw: &Path,
    work: &Path,
    reference: &[i32],
) -> Vec<media::Point> {
    let seconds = reference.len() as f64 / 16_000.0;
    let mut out = Vec::new();
    for &bps in &EVS_RATES {
        let Some(bit) = evs_encode(tools, raw, work, bps) else {
            continue;
        };
        let Some(decoded) = evs_decode(tools, &bit, work, &format!("clean-{bps}")) else {
            continue;
        };
        let lag = media::best_lag(reference, &decoded, ALIGN_LAG);
        let (a, b) = media::aligned_pair(reference, &decoded, lag);
        // The bitstream is the G.192 framing, so the *audio* payload is what
        // a wire would carry: 2 bytes of framing per frame are excluded.
        let frames = evs_frames(&std::fs::read(&bit).unwrap_or_default()).len() as f64;
        let payload_bytes: f64 = evs_frames(&std::fs::read(&bit).unwrap_or_default())
            .iter()
            .map(|f| (f.bits / 8) as f64)
            .sum();
        let _ = frames;
        out.push(media::Point {
            bps: payload_bytes * 8.0 / seconds,
            snr: media::snr_db(&a, &b),
            visqol: competitor_visqol(tools, work, &format!("evs-{bps}"), &a, &b),
        });
    }
    out.sort_by(|a, b| a.bps.partial_cmp(&b.bps).unwrap());
    out
}

pub(crate) fn lyra_points(
    tools: &VoiceTools,
    wav: &Path,
    work: &Path,
    seconds: f64,
    reference: &[i32],
) -> Vec<media::Point> {
    let (Some(enc), Some(dec)) = (tools.media.lyra_enc.as_ref(), tools.media.lyra_dec.as_ref())
    else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for &bps in &LYRA_RATES {
        let dir = work.join(format!("lyra-{bps}"));
        let _ = std::fs::create_dir_all(&dir);
        let ok = media::quiet(
            Command::new(enc)
                .arg(format!("--input_path={}", wav.display()))
                .arg(format!("--output_dir={}", dir.display()))
                .arg(format!("--bitrate={bps}"))
                .arg(format!("--model_path={}", tools.media.lyra_model.display())),
        )
        .status()
        .ok()
        .is_some_and(|s| s.success());
        if !ok {
            continue;
        }
        let stem = wav.file_stem().map(|s| s.to_string_lossy().to_string());
        let Some(stem) = stem else { continue };
        let encoded = dir.join(format!("{stem}.lyra"));
        if !encoded.is_file() {
            continue;
        }
        let size = std::fs::metadata(&encoded).map(|m| m.len()).unwrap_or(0) as f64;
        let ok = media::quiet(
            Command::new(dec)
                .arg(format!("--encoded_path={}", encoded.display()))
                .arg(format!("--output_dir={}", dir.display()))
                .arg(format!("--model_path={}", tools.media.lyra_model.display())),
        )
        .status()
        .ok()
        .is_some_and(|s| s.success());
        if !ok {
            continue;
        }
        let decoded_wav = dir.join(format!("{stem}_decoded.wav"));
        let Ok((samples, _ch, _rate)) = media::read_wav(&decoded_wav) else {
            continue;
        };
        let decoded: Vec<i32> = samples.iter().map(|&v| i32::from(v)).collect();
        let lag = media::best_lag(reference, &decoded, ALIGN_LAG);
        let (a, b) = media::aligned_pair(reference, &decoded, lag);
        out.push(media::Point {
            bps: size * 8.0 / seconds,
            snr: media::snr_db(&a, &b),
            visqol: competitor_visqol(tools, work, &format!("lyra-{bps}"), &a, &b),
        });
    }
    out.sort_by(|a, b| a.bps.partial_cmp(&b.bps).unwrap());
    out
}

/// EVS under the same random-loss ladder, using the G.192 erasure flag.
fn evs_loss_ladder(
    tools: &VoiceTools,
    raw: &Path,
    work: &Path,
    reference: &[i32],
) -> serde_json::Value {
    let bps = 13_200u32;
    let Some(bit) = evs_encode(tools, raw, work, bps) else {
        return serde_json::json!({ "available": false });
    };
    let Ok(bytes) = std::fs::read(&bit) else {
        return serde_json::json!({ "available": false });
    };
    let frames = evs_frames(&bytes);
    let mut out = Vec::new();
    for &lm in &LOSS_PER_MILLE {
        let mut state = SEED | 1;
        let lost: Vec<bool> = (0..frames.len())
            .map(|_| {
                if lm == 0 {
                    false
                } else {
                    common::splitmix64(&mut state) % 1000 < u64::from(lm)
                }
            })
            .collect();
        let marked = evs_apply_loss(&bytes, &lost);
        let path = work.join(format!("evs-loss-{lm}.192"));
        if std::fs::write(&path, &marked).is_err() {
            continue;
        }
        let Some(decoded) = evs_decode(tools, &path, work, &format!("loss-{lm}")) else {
            continue;
        };
        let lag = media::best_lag(reference, &decoded, ALIGN_LAG);
        let (a, b) = media::aligned_pair(reference, &decoded, lag);
        out.push(serde_json::json!({
            "loss_per_mille": lm,
            "requested_frames_lost": lost.iter().filter(|&&v| v).count(),
            "frames": frames.len(),
            "snr_db": media::finite(media::snr_db(&a, &b)),
            "source": "G.192 bad-frame sync word 0x6b20 (the standard's own erasure flag)",
        }));
    }
    serde_json::json!({
        "available": true,
        "bitrate_bps": bps,
        "ladder": out,
        "note": "EVS concealment is its own algorithm; no EVS code is linked into VOLE",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g192_frames_and_erasure_marking_round_trip() {
        // Two synthetic G.192 frames: 8 data bits each, 4 bytes of framing + 16
        // bytes of data.
        let mut b = Vec::new();
        for _ in 0..2 {
            b.extend_from_slice(&0x6b21u16.to_le_bytes());
            b.extend_from_slice(&8u16.to_le_bytes());
            for _ in 0..8 {
                b.extend_from_slice(&0x007fu16.to_le_bytes());
            }
        }
        let frames = evs_frames(&b);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].offset, 0);
        assert_eq!(frames[1].offset, 20);
        let marked = evs_apply_loss(&b, &[true, false]);
        assert_eq!(u16::from_le_bytes([marked[0], marked[1]]), 0x6b20);
        assert_eq!(
            u16::from_le_bytes([marked[20], marked[21]]),
            0x6b21,
            "a good frame must keep the good sync word"
        );
    }

    #[test]
    fn recovery_definition_is_precise() {
        // Perfect reference and a degraded copy that is corrupted for exactly one
        // 20 ms frame and then clean again.
        let clean: Vec<i32> = (0..3200)
            .map(|i| ((i as f64) * 0.05).sin().mul_add(8000.0, 0.0) as i32)
            .collect();
        let mut bad = clean.clone();
        for v in bad.iter_mut().take(640).skip(320) {
            *v = 0;
        }
        let r = recovery_ms(&clean, &bad, 1, 320, 16_000).expect("recovers");
        assert!(
            (0.0..=40.0).contains(&r),
            "recovery of {r} ms is implausible"
        );
        // A scan started inside permanent corruption never reports recovery.
        let mut worse = clean.clone();
        let n = worse.len();
        for v in worse.iter_mut().skip(n - 320) {
            *v = 0;
        }
        assert!(recovery_ms(&clean, &worse, 8, 320, 16_000).is_none());
        // And a reference that matches exactly recovers immediately, because the
        // comparison is against the codec's own no-loss trajectory.
        assert_eq!(recovery_ms(&clean, &clean, 1, 320, 16_000), Some(0.0));
    }

    #[test]
    fn percentiles_are_monotone() {
        let v: Vec<i64> = (1..=100).collect();
        assert!(percentile(&v, 50.0) <= percentile(&v, 90.0));
        assert!(percentile(&v, 90.0) <= percentile(&v, 99.0));
        assert_eq!(percentile(&v, 0.0), 1);
        assert_eq!(percentile(&v, 100.0), 100);
    }

    #[test]
    fn matched_comparison_refuses_to_extrapolate() {
        let vole = vec![media::Point {
            bps: 100_000.0,
            snr: 10.0,
            visqol: None,
        }];
        let competitor = vec![
            media::Point {
                bps: 8_000.0,
                snr: 5.0,
                visqol: None,
            },
            media::Point {
                bps: 16_000.0,
                snr: 8.0,
                visqol: None,
            },
        ];
        let s = matched_summary(&vole, &competitor);
        assert_eq!(s["available"], serde_json::json!(true));
        assert_eq!(s["cells"], serde_json::json!(0));
        assert_eq!(s["snr_delta_mean_db"], serde_json::json!(null));
    }
}
