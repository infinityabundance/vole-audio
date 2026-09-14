//! `court learned-voice-stream-exp2` — Phase 7C.2-A challenger court.
//!
//! The constitution is [`docs/PHASE_7C2.md`], frozen before this file. The
//! regression court remains `learned-voice-stream`; this court measures the
//! same codec under test on a **speaker-disjoint held-out** corpus and reports
//! the headline metric the phase is judged on — an integrated
//! *bitrate-at-equal-quality* delta against each competitor — instead of a
//! count of individual cells.
//!
//! Codec under test: `voice.exp1` is the **labelled control** until the
//! `voice.exp2` profile lands (7C.2-C). The court is frozen now, before that
//! work, so the challenger corpus cannot be tuned to.
//!
//! Only VOLE-derived integers enter the frozen projection, so the frozen hash is
//! stable across host tool versions. Competitor numbers appear in the receipt,
//! never in the frozen hash.

use crate::courts::learned_common as common;
use crate::courts::learned_lossy as media;
use crate::courts::learned_voice_stream as vc;
use crate::error::{Error, Result};
use crate::status::Verdict;
use crate::voice::{self, VoiceConfig};
use std::path::{Path, PathBuf};

/// Frozen static-result hash over the VOLE-derived projection. Empty until the
/// first observation is frozen; see `docs/PHASE_7C2.md` §7.
pub const LEARNED_VOICE_STREAM_EXP2_SHA256: &str =
    "23eaf3610af66d789bb188aedd6e1359ba9ac06ba706d691a2f853845588d43c";

/// VOLE voice operating points (bits per second).
const VOLE_RATES: [u32; 6] = [6_000, 8_000, 12_000, 16_000, 24_000, 32_000];
/// Frame shapes measured.
const SHAPES: [(u32, u8); 3] = [(160, 1), (320, 1), (160, 2)];
/// Global best-lag alignment bound, as in the regression court.
const ALIGN_LAG: i64 = 512;
/// Conversation-scale case target (20 ms frames × 200).
const CASE_TARGET_SAMPLES: usize = 16_000 * 4;
/// Clips read to build the cases (the whole held-out split).
const CASE_CLIP_POOL: usize = 8;
/// Deterministic bootstrap seed.
const SEED: u64 = 0x7C2C_2A01;
/// Bootstrap resamples.
const BOOTSTRAP: usize = 2_000;

// ---------------------------------------------------------------------------
// Integrated bitrate-at-equal-quality
// ---------------------------------------------------------------------------

/// log2-rate at which a monotone curve reaches quality `q`, by linear
/// interpolation.
fn hull_rate_at_quality(hull: &[(f64, f64)], q: f64) -> Option<f64> {
    if hull.len() < 2 {
        return None;
    }
    let (lo, hi) = (hull[0].1, hull[hull.len() - 1].1);
    if q < lo || q > hi {
        return None;
    }
    for w in hull.windows(2) {
        let (r0, q0) = w[0];
        let (r1, q1) = w[1];
        if q >= q0 && q <= q1 {
            if (q1 - q0).abs() < 1e-12 {
                return Some(r0);
            }
            let t = (q - q0) / (q1 - q0);
            return Some(r0 + t * (r1 - r0));
        }
    }
    None
}

/// The monotone non-decreasing quality envelope of `(log2 rate, quality)`: one
/// knot per measured rate, quality replaced by the best value at or below that
/// rate. Unlike a strict Pareto frontier it survives a quality curve that
/// saturates, which is what makes the integrated metrics well defined here.
fn monotone_curve(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut pts: Vec<(f64, f64)> = points
        .iter()
        .copied()
        .filter(|(r, q)| r.is_finite() && q.is_finite())
        .collect();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0).then(b.1.total_cmp(&a.1)));
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(pts.len());
    let mut best = f64::NEG_INFINITY;
    for (r, q) in pts {
        if q > best {
            best = q;
        }
        out.push((r, best));
    }
    out.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-12);
    out
}

/// Interpolate quality at `r` on a monotone curve.
fn interp_quality(curve: &[(f64, f64)], r: f64) -> Option<f64> {
    for w in curve.windows(2) {
        let (r0, q0) = w[0];
        let (r1, q1) = w[1];
        if r >= r0 && r <= r1 {
            if (r1 - r0).abs() < 1e-12 {
                return Some(q1);
            }
            let t = (r - r0) / (r1 - r0);
            return Some(q0 + t * (q1 - q0));
        }
    }
    None
}

/// Mean integrated **quality-at-equal-rate** difference over the shared rate
/// range, in the metric's own units (dB for SNR, MOS for ViSQOL). Negative
/// means VOLE is worse at the same bitrate. Defined whenever the rate ranges
/// overlap, which is the robust dual of the rate-at-equal-quality form.
fn integrated_quality_delta(vole: &[(f64, f64)], competitor: &[(f64, f64)]) -> Option<f64> {
    let v = monotone_curve(vole);
    let c = monotone_curve(competitor);
    if v.len() < 2 || c.len() < 2 {
        return None;
    }
    let lo = v[0].0.max(c[0].0);
    let hi = v[v.len() - 1].0.min(c[c.len() - 1].0);
    // `!(lo < hi)` also rejects a NaN bound, which `lo >= hi` would not.
    if !matches!(lo.partial_cmp(&hi), Some(std::cmp::Ordering::Less)) {
        return None;
    }
    let mut knots: Vec<f64> = vec![lo, hi];
    for &(r, _) in v.iter().chain(c.iter()) {
        if r > lo && r < hi {
            knots.push(r);
        }
    }
    knots.sort_by(f64::total_cmp);
    knots.dedup();
    let mut acc = 0.0;
    let mut width_sum = 0.0;
    for w in knots.windows(2) {
        let (a, b) = (w[0], w[1]);
        let m = 0.5 * (a + b);
        let qv = interp_quality(&v, m)?;
        let qc = interp_quality(&c, m)?;
        acc += (qv - qc) * (b - a);
        width_sum += b - a;
    }
    (width_sum > 0.0).then_some(acc / width_sum)
}

/// Quality range of a point set, if it has one.
fn quality_range(points: &[(f64, f64)]) -> Option<(f64, f64)> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &(_, q) in points {
        if q.is_finite() {
            lo = lo.min(q);
            hi = hi.max(q);
        }
    }
    (lo <= hi).then_some((lo, hi))
}

/// Mean integrated rate delta in dB over the competitor's measured quality
/// range: `10·log10(rate_vole(q) / rate_competitor(q))`. Negative means VOLE
/// needs fewer bits for the same quality.
///
/// Returns `Err` with an explicit reason when the delta is **not defined** — the
/// constitution forbids extrapolation, so a quality range with no overlap simply
/// has no bitrate-at-equal-quality figure, and saying so is the correct result.
fn integrated_rate_delta_db(
    vole: &[(f64, f64)],
    competitor: &[(f64, f64)],
) -> std::result::Result<f64, String> {
    let hull = monotone_curve(vole);
    let comp = monotone_curve(competitor);
    if hull.len() < 2 {
        return Err(format!(
            "VOLE curve has no rate-quality frontier ({} cells, {} on it)",
            vole.len(),
            hull.len()
        ));
    }
    if comp.len() < 2 {
        return Err(format!(
            "competitor curve has no frontier ({} cells)",
            comp.len()
        ));
    }
    let (Some((vlo, vhi)), Some((clo, chi))) = (quality_range(vole), quality_range(competitor))
    else {
        return Err("a curve has no finite quality".to_string());
    };
    if chi < vlo || clo > vhi {
        return Err(format!(
            "quality ranges do not overlap: VOLE [{vlo:.3},{vhi:.3}] vs competitor [{clo:.3},{chi:.3}]"
        ));
    }
    let mut deltas = Vec::new();
    for &(_cr, cq) in &comp {
        let Some(vr) = hull_rate_at_quality(&hull, cq) else {
            continue;
        };
        let Some(cr2) = hull_rate_at_quality(&comp, cq) else {
            continue;
        };
        deltas.push(vr - cr2);
    }
    if deltas.is_empty() {
        return Err(
            "no competitor operating point lies inside the shared quality range".to_string(),
        );
    }
    // Hull rates are log2, so a mean log2 difference converts to a dB ratio by
    // 10·log10(2^d) = d · 10·ln2/ln10 = d · 3.0103.
    let mean = deltas.iter().sum::<f64>() / deltas.len() as f64;
    Ok(mean * 10.0 * std::f64::consts::LN_2 / std::f64::consts::LN_10)
}

fn bootstrap_ci(values: &[f64], seed: u64) -> Option<(f64, f64, f64)> {
    if values.is_empty() {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut state = seed | 1;
    let mut means = Vec::with_capacity(BOOTSTRAP);
    for _ in 0..BOOTSTRAP {
        let mut acc = 0.0;
        for _ in 0..values.len() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            acc += values[(state as usize) % values.len()];
        }
        means.push(acc / values.len() as f64);
    }
    means.sort_by(f64::total_cmp);
    let lo = means[(BOOTSTRAP as f64 * 0.025) as usize];
    let hi = means[(BOOTSTRAP as f64 * 0.975) as usize];
    Some((mean, lo, hi))
}

/// Points as `(bps, metric)` pairs, dropping unavailable cells.
fn pairs(points: &[media::Point], metric: Metric) -> Vec<(f64, f64)> {
    points
        .iter()
        .filter_map(|p| {
            let q = match metric {
                Metric::Snr => p.snr,
                Metric::Visqol => p.visqol.unwrap_or(f64::NAN),
            };
            (p.bps.is_finite() && q.is_finite()).then_some((p.bps.log2(), q))
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Metric {
    Snr,
    Visqol,
}

/// Run the court; writes `receipts/learned-voice-stream-exp2/`.
pub fn run(receipts_root: &Path) -> Result<Verdict> {
    let mut projection = Vec::new();
    common::push_label(&mut projection, "vole.audio.stream.voice.exp2.court.v1");
    common::push_label(&mut projection, voice::VOICE_PROFILE);
    common::push_label(
        &mut projection,
        &crate::learned::corpus_real::challenger_corpus_sha256(),
    );

    let tools = vc::VoiceTools::discover();
    let work = PathBuf::from("target/voice-exp2-court");
    std::fs::create_dir_all(&work).map_err(Error::io)?;

    let corpus = vc::cases_from(
        &crate::learned::corpus_real::challenger_clips(),
        CASE_CLIP_POOL,
        CASE_TARGET_SAMPLES,
    )?;

    let mut all_ok = true;
    let mut rows = Vec::new();
    let mut snr_deltas: Vec<f64> = Vec::new();
    let mut visqol_deltas: Vec<f64> = Vec::new();
    let mut bd_opus: Vec<f64> = Vec::new();
    let mut bd_evs: Vec<f64> = Vec::new();
    let mut bd_lyra: Vec<f64> = Vec::new();
    // Quality-at-equal-rate, kept per metric so units never mix in a mean.
    let mut qs_opus: Vec<f64> = Vec::new();
    let mut qs_evs: Vec<f64> = Vec::new();
    let mut qs_lyra: Vec<f64> = Vec::new();
    let mut qv_opus: Vec<f64> = Vec::new();
    let mut qv_evs: Vec<f64> = Vec::new();
    let mut qv_lyra: Vec<f64> = Vec::new();
    let mut encode_us: Vec<i64> = Vec::new();

    for case in &corpus {
        let ref_wav = work.join(format!("ref-{}.wav", case.id));
        media::write_wav(&ref_wav, &media::to_i16(&case.samples), 1, 16_000)?;
        let raw = work.join(format!("ref-{}.16k", case.id));
        vc::write_raw16(&raw, &case.samples)?;
        let seconds = case.samples.len() as f64 / 16_000.0;

        // ---- VOLE control ladder -------------------------------------------
        let mut vole_points: Vec<media::Point> = Vec::new();
        for (frame_len, fpp) in SHAPES {
            for bps in VOLE_RATES {
                let cfg: VoiceConfig = vc::config_for(frame_len, fpp, bps, false);
                let run = vc::run_clean(cfg, &case.samples)?;
                let actual_bps = run.bytes as f64 * 8.0 / seconds;
                let snr = vc::snr_aligned(&case.samples, &run.samples);
                let lag = media::best_lag(&case.samples, &run.samples, ALIGN_LAG);
                let (a, b) = media::aligned_pair(&case.samples, &run.samples, lag);
                let mos = vc::run_visqol(&tools, &work, &case.id, &a, &b);
                if run.bytes == 0 || snr.is_nan() {
                    all_ok = false;
                }
                for ns in &run.encode_ns {
                    encode_us.push(*ns / 1_000);
                }
                common::push_label(&mut projection, &case.id);
                common::push_label(&mut projection, &format!("shape-{frame_len}x{fpp}"));
                common::push_u64(&mut projection, u64::from(bps));
                common::push_u64(&mut projection, run.bytes as u64);
                common::push_u64(&mut projection, (snr.clamp(-100.0, 200.0) * 100.0) as u64);
                vole_points.push(media::Point {
                    bps: actual_bps,
                    snr,
                    visqol: mos,
                });
                rows.push(serde_json::json!({
                    "case": case.id,
                    "shape": format!("{frame_len}x{fpp}"),
                    "target_bps": bps,
                    "actual_bps": actual_bps,
                    "bytes": run.bytes,
                    "snr_db": media::finite(snr),
                    "visqol_mos": mos,
                }));
            }
        }
        vole_points.sort_by(|a, b| {
            a.bps
                .partial_cmp(&b.bps)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // ---- competitors ----------------------------------------------------
        let opus = vc::opus_points(&tools, &ref_wav, &work, &case.samples);
        let evs = vc::evs_points(&tools, &raw, &work, &case.samples);
        let lyra = vc::lyra_points(&tools, &ref_wav, &work, seconds, &case.samples);

        let matched = serde_json::json!({
            "opus": vc::matched_summary(&vole_points, &opus),
            "evs": vc::matched_summary(&vole_points, &evs),
            "lyra": vc::matched_summary(&vole_points, &lyra),
        });
        if let Some(d) = matched["opus"]["snr_delta_mean_db"].as_f64() {
            snr_deltas.push(d);
        }
        if let Some(d) = matched["opus"]["visqol_delta_mean"].as_f64() {
            visqol_deltas.push(d);
        }
        // Integrated deltas over the overlap, per case: SNR where it is defined,
        // otherwise ViSQOL (the metric Lyra makes meaningful). An undefined delta
        // is recorded with its reason, never extrapolated.
        let vole_snr = pairs(&vole_points, Metric::Snr);
        let vole_vq = pairs(&vole_points, Metric::Visqol);
        let mut integrated = serde_json::Map::new();
        for (name, comp, sink, qsink, qvsink) in [
            ("opus", &opus, &mut bd_opus, &mut qs_opus, &mut qv_opus),
            ("evs", &evs, &mut bd_evs, &mut qs_evs, &mut qv_evs),
            ("lyra", &lyra, &mut bd_lyra, &mut qs_lyra, &mut qv_lyra),
        ] {
            let comp_snr = pairs(comp, Metric::Snr);
            let comp_vq = pairs(comp, Metric::Visqol);
            // Primary: quality-at-equal-rate, per metric (robust). Secondary:
            // rate-at-equal-quality in dB, recorded where definable with the
            // reason when it is not.
            let q_snr = integrated_quality_delta(&vole_snr, &comp_snr);
            let q_vq = integrated_quality_delta(&vole_vq, &comp_vq);
            let r_snr = integrated_rate_delta_db(&vole_snr, &comp_snr);
            let r_vq = integrated_rate_delta_db(&vole_vq, &comp_vq);
            if let Some(d) = q_snr {
                qsink.push(d);
            }
            if let Some(d) = q_vq {
                qvsink.push(d);
            }
            let (rate_delta, rate_metric, rate_reason) = match (&r_snr, &r_vq) {
                (Ok(d), _) => (Some(*d), "snr", None),
                (Err(_), Ok(d)) => (Some(*d), "visqol", None),
                (Err(e1), Err(e2)) => (None, "none", Some(format!("snr: {e1}; visqol: {e2}"))),
            };
            if rate_metric == "snr"
                && let Some(d) = rate_delta
            {
                sink.push(d);
            }
            integrated.insert(
                name.to_string(),
                serde_json::json!({
                    "quality_at_equal_rate_snr_db": q_snr,
                    "quality_at_equal_rate_visqol_mos": q_vq,
                    "rate_at_equal_quality_db": rate_delta,
                    "rate_metric": rate_metric,
                    "rate_reason": rate_reason,
                    "vole_quality_range": quality_range(&vole_snr).map(|(a, b)| serde_json::json!([a, b])),
                    "competitor_quality_range": quality_range(&comp_snr).map(|(a, b)| serde_json::json!([a, b])),
                }),
            );
        }

        rows.push(serde_json::json!({
            "case": case.id,
            "kind": "summary",
            "held_out_speaker": true,
            "matched": matched,
            "integrated_rate_delta": integrated,
            "vole": vole_points.iter().map(|p| serde_json::json!({"bps": p.bps, "snr": media::finite(p.snr), "visqol": p.visqol})).collect::<Vec<_>>(),
            "opus": opus.iter().map(|p| serde_json::json!({"bps": p.bps, "snr": media::finite(p.snr), "visqol": p.visqol})).collect::<Vec<_>>(),
            "evs": evs.iter().map(|p| serde_json::json!({"bps": p.bps, "snr": media::finite(p.snr), "visqol": p.visqol})).collect::<Vec<_>>(),
            "lyra": lyra.iter().map(|p| serde_json::json!({"bps": p.bps, "snr": media::finite(p.snr), "visqol": p.visqol})).collect::<Vec<_>>(),
        }));
    }

    let ci = |v: &[f64]| match bootstrap_ci(v, SEED) {
        Some((m, lo, hi)) => serde_json::json!({"mean": m, "lo": lo, "hi": hi, "cases": v.len()}),
        None => serde_json::Value::Null,
    };
    encode_us.sort_unstable();

    let verdict = if all_ok {
        Verdict::Supported
    } else {
        Verdict::FailedCorrectness
    };
    let bd = |v: &[f64]| {
        if v.is_empty() {
            "n/a".to_string()
        } else {
            format!("{:+.2}", v.iter().sum::<f64>() / v.len() as f64)
        }
    };

    common::finish_with_profile(
        "learned-voice-stream-exp2",
        crate::learned::profile::LearnedProfile::Exp3,
        receipts_root,
        LEARNED_VOICE_STREAM_EXP2_SHA256,
        &projection,
        verdict,
        format!(
            "voice.exp2 challenger court over {} held-out cases of {} (codec under test: {} \
             control): quality-at-equal-rate vs Opus {} dB SNR / {} MOS, vs EVS {} / {}, \
             vs Lyra {} / {}; bitrate-at-equal-quality vs Opus {} dB, vs EVS {} dB, vs Lyra {} dB \
             (n/a where the quality ranges do not overlap); matched-bitrate SNR vs Opus {} dB; \
             encode p99 {} µs; the profile is lossy and makes no exactness claim",
            corpus.len(),
            crate::learned::corpus_real::challenger_corpus_sha256(),
            voice::VOICE_PROFILE,
            bd(&qs_opus),
            bd(&qv_opus),
            bd(&qs_evs),
            bd(&qv_evs),
            bd(&qs_lyra),
            bd(&qv_lyra),
            bd(&bd_opus),
            bd(&bd_evs),
            bd(&bd_lyra),
            bd(&snr_deltas),
            vc::percentile(&encode_us, 99.0),
        ),
        vec![
            (
                "harness",
                serde_json::json!({
                    "court": "learned-voice-stream-exp2",
                    "constitution": "docs/PHASE_7C2.md",
                    "codec_under_test": voice::VOICE_PROFILE,
                    "codec_under_test_note": "voice.exp1 is the labelled control until 7C.2-C lands voice.exp2",
                    "corpus": "held-out test-clean (speaker-disjoint)",
                    "corpus_sha256": crate::learned::corpus_real::challenger_corpus_sha256(),
                    "opus": tools.media.opusenc.as_ref().map(|p| p.to_string_lossy().to_string()),
                    "evs": tools.evs_cod.as_ref().map(|p| p.to_string_lossy().to_string()),
                    "lyra": tools.media.lyra_enc.as_ref().map(|p| p.to_string_lossy().to_string()),
                    "visqol": tools.media.visqol.as_ref().map(|p| p.to_string_lossy().to_string()),
                    "shapes": SHAPES.iter().map(|(f, n)| format!("{f}x{n}")).collect::<Vec<_>>(),
                    "vole_rates": VOLE_RATES,
                    "metric": "quality-at-equal-rate (SNR dB, ViSQOL MOS) plus bitrate-at-equal-quality (dB), integrated over the shared rate range with the monotone quality envelope",
                    "bootstrap": BOOTSTRAP,
                    "seed": SEED,
                    "note": "competitors are external processes; no code is imported, linked, wrapped or used as a fallback",
                }),
            ),
            (
                "integrated_metrics",
                serde_json::json!({
                    "quality_at_equal_rate_snr_db": {"opus": ci(&qs_opus), "evs": ci(&qs_evs), "lyra": ci(&qs_lyra)},
                    "quality_at_equal_rate_visqol_mos": {"opus": ci(&qv_opus), "evs": ci(&qv_evs), "lyra": ci(&qv_lyra)},
                    "bitrate_at_equal_quality_snr_db": {"opus": ci(&bd_opus), "evs": ci(&bd_evs), "lyra": ci(&bd_lyra)},
                    "matched_snr_vs_opus_db": ci(&snr_deltas),
                    "matched_visqol_vs_opus": ci(&visqol_deltas),
                    "encode_p99_us": vc::percentile(&encode_us, 99.0),
                    "interpretation": "quality deltas are VOLE minus competitor at equal rate (negative = VOLE worse); bitrate deltas are dB of rate at equal quality (negative = VOLE needs fewer bits)",
                }),
            ),
            ("cases", serde_json::json!(rows)),
            (
                "limitations",
                serde_json::json!([
                    "the held-out corpus is read speech only: no whisper/breath, no realistic background noise, no non-English speech; broader claims are blocked until more rights-clean speaker-disjoint material is frozen",
                    "the codec under test is the voice.exp1 control until the voice.exp2 profile lands (7C.2-C)",
                    "integrated deltas are computed over the competitor's measured quality range only; no extrapolation",
                    "Opus and Lyra expose no packet-loss simulation through their pinned CLIs, so their loss behaviour is NOT_AVAILABLE rather than guessed",
                    "ViSQOL speech mode is applied where the pinned binary and model run; absent cells are null",
                ]),
            ),
        ],
    )
}
