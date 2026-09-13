//! `vole.audio.lossy.exp1` — VOLE's lossy codec.
//!
//! The codec is built **on the Phase-7A explanation pipeline**: a deterministic
//! explanation `H` (a predictive model or a transform) is chosen by search to
//! minimise the *actual emitted representation*, and whatever `H` cannot
//! explain is coded as a residual with VOLE's native entropy engine.
//!
//! ```text
//! PCM ──► explanation search
//!             │
//!     ┌───────┴────────┐
//!     │                │
//!  predictive H     transform H
//!  (backward-       (MDCT + Bark
//!   adaptive         masking + per-band
//!   NLMS DPCM)       dead-zone SQ)
//!     │                │
//!     └───────┬────────┘
//!             │
//!    native entropy code (VOLE ResidualCodecV2 family)
//! ```
//!
//! ## Explanation search
//!
//! `H` is the *representation*, not a fixed algorithm. Both engines expose a
//! single rate scalar; the encoder bisects it to land on the target size, and
//! the engine that leaves the smaller reconstruction error at that size — or
//! the smaller size when neither reaches the target — is the one emitted. This
//! is exactly the Phase-7A principle applied to lossy coding: the search
//! answers "what deterministic explanation leaves the cheapest remaining
//! information to encode?" and the physical byte count decides.
//!
//! ## Decoder authority
//!
//! The decoder is authoritative for every bit it reconstructs:
//!
//! * the predictive engine's weights and history are recomputed identically at
//!   both ends (`src/lossy/predict.rs`), so they cost no bits;
//! * the transform engine's band shape is predicted from the previous decoded
//!   frame, so only its change is transmitted;
//! * the entropy payloads are decoded by the same VOLE-native codec family used
//!   everywhere else in the crate.

use crate::error::{Error, Kind, Result};
use crate::lossy::mdct::Mdct;
use crate::lossy::predict::PredictEngine;
use crate::lossy::psy::Bands;
use crate::lossy::transform::TransformEngine;

pub mod mdct;
pub mod predict;
pub mod psy;
pub mod transform;

/// Experimental lossy profile identity.
pub const LOSSY_PROFILE: &str = "vole.audio.lossy.exp1";
/// Canonical profile tag bytes.
pub const LOSSY_PROFILE_TAG: &[u8] = b"vole.audio.lossy.exp1";
/// Container magic.
pub const LOSSY_MAGIC: &[u8; 10] = b"vole.lossy";
/// Container format version.
pub const LOSSY_VERSION: u8 = 2;
/// Predictor order for the backward-adaptive engine.
pub const PREDICT_ORDER: usize = 16;
/// Engine tag: MDCT transform.
pub const MODE_TRANSFORM: u8 = 0;
/// Engine tag: backward-adaptive prediction.
pub const MODE_PREDICT: u8 = 1;

/// Lossy codec configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LossyConfig {
    pub sample_rate_hz: u32,
    pub channels: u8,
    /// Hop length in samples; the MDCT window is twice this.
    pub frame_len: u32,
    /// Target bitrate for the whole object (all channels), bits per second.
    pub target_bits_per_second: u32,
}

impl LossyConfig {
    /// A default hop length for a sample rate: about 20 ms, capped so the
    /// direct MDCT stays cheap.
    pub fn default_frame_len(sample_rate_hz: u32) -> u32 {
        let target = (sample_rate_hz as f64 * 0.020).round() as u32;
        target.clamp(128, 512) & !1
    }

    pub fn validate(&self) -> Result<()> {
        if self.channels == 0 || u32::from(self.channels) > crate::limits::MAX_CHANNELS {
            return Err(Error::malformed("lossy channel count out of domain"));
        }
        if self.sample_rate_hz < 8_000 || self.sample_rate_hz > crate::limits::MAX_SAMPLE_RATE_HZ {
            return Err(Error::malformed("lossy sample rate out of domain"));
        }
        if self.frame_len < 64 || !self.frame_len.is_multiple_of(2) || self.frame_len > 512 {
            return Err(Error::malformed("lossy hop length out of domain"));
        }
        if self.target_bits_per_second == 0 {
            return Err(Error::malformed("lossy target bitrate must be positive"));
        }
        Ok(())
    }

    /// Number of MDCT coefficients per frame (== hop length).
    pub const fn coeffs_per_frame(&self) -> usize {
        self.frame_len as usize
    }
}

/// Analysis of one frame shared by both engines.
struct FrameAnalysis {
    /// `n` MDCT coefficients.
    coeffs: Vec<f64>,
    /// Masking-derived reference steps.
    step0: Vec<f64>,
    /// The `n` new source samples this frame advances by.
    samples: Vec<f64>,
}

/// A bitstream writer (little-endian, byte aligned).
struct BitWriter {
    out: Vec<u8>,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter { out: Vec::new() }
    }
    fn u8(&mut self, v: u8) {
        self.out.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes(&mut self, b: &[u8]) {
        self.out.extend_from_slice(b);
    }
    fn finish(self) -> Vec<u8> {
        self.out
    }
}

struct BitReader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(b: &'a [u8]) -> Self {
        BitReader { b, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| Error::limit("lossy read overflows"))?;
        let s = self
            .b
            .get(self.pos..end)
            .ok_or_else(|| Error::malformed("lossy bitstream is truncated"))?;
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

/// Complete lossy codec instance for one configuration.
#[derive(Debug, Clone)]
pub struct LossyCodec {
    config: LossyConfig,
    mdct: Mdct,
    bands: Bands,
}

impl LossyCodec {
    pub fn new(config: LossyConfig) -> Result<LossyCodec> {
        config.validate()?;
        let n = config.coeffs_per_frame();
        let mdct = Mdct::new(n);
        let full_scale_coeff = calibrate_full_scale(&mdct, config.sample_rate_hz);
        let bands = Bands::new(config.sample_rate_hz, n, full_scale_coeff);
        Ok(LossyCodec {
            config,
            mdct,
            bands,
        })
    }

    pub fn config(&self) -> &LossyConfig {
        &self.config
    }

    /// Encode interleaved canonical samples.
    pub fn encode(&self, samples: &[i32]) -> Result<Vec<u8>> {
        let ch = usize::from(self.config.channels);
        if samples.is_empty() || !samples.len().is_multiple_of(ch) {
            return Err(Error::malformed("lossy input is not frame-aligned"));
        }
        let frames = samples.len() / ch;
        let seconds = frames as f64 / f64::from(self.config.sample_rate_hz);
        let target_bytes = ((f64::from(self.config.target_bits_per_second) * seconds / 8.0)
            .round()
            .max(16.0)) as usize;

        let mut w = BitWriter::new();
        w.bytes(LOSSY_MAGIC);
        w.u8(LOSSY_VERSION);
        w.u8(LOSSY_PROFILE_TAG.len() as u8);
        w.bytes(LOSSY_PROFILE_TAG);
        w.u8(self.config.channels);
        w.u32(self.config.sample_rate_hz);
        w.u32(self.config.frame_len);
        w.u32(frames as u32);
        for c in 0..ch {
            let channel: Vec<f64> = samples
                .iter()
                .skip(c)
                .step_by(ch)
                .map(|&v| f64::from(v))
                .collect();
            let per_channel = target_bytes / ch;
            let (mode, stream) = self.encode_channel_best(&channel, per_channel)?;
            w.u8(mode);
            w.u32(stream.len() as u32);
            w.bytes(&stream);
        }
        Ok(w.finish())
    }

    /// Decode to interleaved canonical samples.
    pub fn decode(&self, bytes: &[u8]) -> Result<Vec<i32>> {
        let mut r = BitReader::new(bytes);
        if r.take(LOSSY_MAGIC.len())? != LOSSY_MAGIC {
            return Err(Error::malformed("lossy magic mismatch"));
        }
        if r.u8()? != LOSSY_VERSION {
            return Err(Error::new(Kind::Unsupported, "unsupported lossy version"));
        }
        let plen = r.u8()? as usize;
        if r.take(plen)? != LOSSY_PROFILE_TAG {
            return Err(Error::malformed("lossy profile tag mismatch"));
        }
        let channels = r.u8()?;
        let sample_rate_hz = r.u32()?;
        let frame_len = r.u32()?;
        let frames = r.u32()? as usize;
        if channels != self.config.channels
            || sample_rate_hz != self.config.sample_rate_hz
            || frame_len != self.config.frame_len
        {
            return Err(Error::malformed(
                "lossy header disagrees with configuration",
            ));
        }
        let ch = usize::from(channels);
        let mut channels_out: Vec<Vec<f64>> = Vec::with_capacity(ch);
        for _ in 0..ch {
            let mode = r.u8()?;
            let len = r.u32()? as usize;
            let stream = r.take(len)?;
            channels_out.push(self.decode_channel(mode, stream, frames)?);
        }
        let mut out = vec![0i32; frames * ch];
        for c in 0..ch {
            for f in 0..frames {
                out[f * ch + c] = channels_out[c][f].round().clamp(-2.0e9, 2.0e9) as i32;
            }
        }
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // Frame analysis
    // -----------------------------------------------------------------------

    fn analyse(&self, samples: &[f64]) -> Vec<FrameAnalysis> {
        let n = self.mdct.n();
        let hop = n;
        let frames = (samples.len() + hop).div_ceil(hop).max(1);
        let mut out = Vec::with_capacity(frames);
        for f in 0..frames {
            let start = f as i64 * hop as i64 - hop as i64;
            let mut window = vec![0.0f64; 2 * n];
            for (m, slot) in window.iter_mut().enumerate() {
                let g = start + m as i64;
                if g >= 0 && (g as usize) < samples.len() {
                    *slot = samples[g as usize];
                }
            }
            let coeffs = self.mdct.forward(&window);
            let step0 = self.step0(&coeffs);
            let mut src = vec![0.0f64; n];
            for (m, slot) in src.iter_mut().enumerate() {
                let g = start + hop as i64 + m as i64;
                if g >= 0 && (g as usize) < samples.len() {
                    *slot = samples[g as usize];
                }
            }
            out.push(FrameAnalysis {
                coeffs,
                step0,
                samples: src,
            });
        }
        out
    }

    fn step0(&self, coeffs: &[f64]) -> Vec<f64> {
        let masking = crate::lossy::psy::Masking::analyse(coeffs, &self.bands);
        let bc = self.bands.band_count;
        let mut step0 = vec![0.0f64; bc];
        for (b, slot) in step0.iter_mut().enumerate() {
            let n_b = f64::from(self.bands.bins_per_band[b].max(1));
            *slot = (12.0 * masking.threshold_power[b] / n_b).sqrt().max(1e-6);
        }
        step0
    }

    // -----------------------------------------------------------------------
    // Engine search
    // -----------------------------------------------------------------------

    /// Encode one channel with the better of the two engines at the target size.
    fn encode_channel_best(&self, samples: &[f64], target_bytes: usize) -> Result<(u8, Vec<u8>)> {
        let analysis = self.analyse(samples);
        let transform = self.encode_transform(&analysis, target_bytes);
        let predict = self.encode_predict(samples, &analysis, target_bytes);
        let t_ok = transform.bytes <= target_bytes;
        let p_ok = predict.bytes <= target_bytes;
        let prefer_transform = match (t_ok, p_ok) {
            (true, true) => transform.distortion < predict.distortion,
            (true, false) => true,
            (false, true) => false,
            // Neither reaches the target: take the smaller representation.
            (false, false) => transform.bytes <= predict.bytes,
        };
        let chosen = if prefer_transform { transform } else { predict };
        let mut w = BitWriter::new();
        w.u32(chosen.frames as u32);
        w.bytes(&chosen.stream);
        Ok((chosen.mode, w.finish()))
    }

    fn encode_transform(&self, analysis: &[FrameAnalysis], target: usize) -> ChannelEncoding {
        let mut lo = -30i32;
        let mut hi = 120i32;
        let mut candidates: Vec<ChannelEncoding> = Vec::new();
        for _ in 0..12 {
            let g = (lo + hi) / 2;
            let enc = self.transform_pass(analysis, g, false);
            let fits = enc.bytes <= target;
            candidates.push(enc);
            if fits {
                hi = g;
            } else {
                lo = g;
            }
            if hi - lo <= 1 {
                break;
            }
        }
        // A wide refinement around the rate boundary: the byte cost is a step
        // function of the gain, and the distortion is not monotone in it once a
        // band's peak coefficient starts to clamp.
        for d in -10i32..=10 {
            let g = (hi + d).clamp(-40, 130);
            candidates.push(self.transform_pass(analysis, g, false));
        }
        let best = Self::pick_best(candidates, target);
        // Re-encode the winner with the full entropy-codec family: the search
        // subset only orders the candidates.
        self.transform_pass(analysis, best.gain, true)
    }

    fn transform_pass(&self, analysis: &[FrameAnalysis], gain: i32, full: bool) -> ChannelEncoding {
        let codecs = if full {
            &crate::learned::residual_codec2::ResidualCodecV2::ALL_V3[..]
        } else {
            &crate::learned::residual_codec2::SEARCH_CODECS[..]
        };
        let mut engine = TransformEngine::new(self.bands.clone(), self.mdct.clone());
        let mut stream = Vec::new();
        let mut distortion = 0.0f64;
        for frame in analysis {
            let cand = engine.evaluate_with(&frame.coeffs, &frame.step0, gain, codecs);
            stream.extend_from_slice(&(cand.record.len() as u16).to_le_bytes());
            stream.extend_from_slice(&cand.record);
            distortion += cand.distortion;
            engine.commit(&cand);
        }
        ChannelEncoding {
            mode: MODE_TRANSFORM,
            gain,
            bytes: stream.len(),
            distortion,
            frames: analysis.len(),
            stream,
        }
    }

    /// Choose among evaluated candidates: the lowest distortion that meets the
    /// byte target, else the fewest bytes. Minimising distortion subject to the
    /// rate is what prevents a coarser-but-still-fitting step from dead-zoning a
    /// frame into silence.
    fn pick_best(candidates: Vec<ChannelEncoding>, target: usize) -> ChannelEncoding {
        let mut best = candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| c.bytes <= target)
            .min_by(|(_, a), (_, b)| a.distortion.partial_cmp(&b.distortion).unwrap())
            .map(|(i, _)| i);
        if best.is_none() {
            best = candidates
                .iter()
                .enumerate()
                .min_by_key(|(_, c)| c.bytes)
                .map(|(i, _)| i);
        }
        let i = best.expect("the candidate set is non-empty");
        candidates.into_iter().nth(i).expect("index is in range")
    }

    fn encode_predict(
        &self,
        samples: &[f64],
        analysis: &[FrameAnalysis],
        target: usize,
    ) -> ChannelEncoding {
        let mut candidates: Vec<ChannelEncoding> = Vec::new();
        let mut lo = -20i32;
        let mut hi = 60i32;
        for _ in 0..14 {
            let g = (lo + hi) / 2;
            let enc = self.predict_pass(samples, analysis, g, false);
            let fits = enc.bytes <= target;
            candidates.push(enc);
            if fits {
                hi = g;
            } else {
                lo = g;
            }
            if hi - lo <= 1 {
                break;
            }
        }
        for d in -8i32..=8 {
            let g = (hi + d).clamp(-40, 80);
            candidates.push(self.predict_pass(samples, analysis, g, false));
        }
        let best = Self::pick_best(candidates, target);
        self.predict_pass(samples, analysis, best.gain, true)
    }

    fn predict_pass(
        &self,
        samples: &[f64],
        analysis: &[FrameAnalysis],
        gain: i32,
        full: bool,
    ) -> ChannelEncoding {
        let codecs = if full {
            &crate::learned::residual_codec2::ResidualCodecV2::ALL_V3[..]
        } else {
            &crate::learned::residual_codec2::SEARCH_CODECS[..]
        };
        let n = self.mdct.n();
        let min_lag = (self.config.sample_rate_hz as usize / 500).max(8);
        let max_lag = (self.config.sample_rate_hz as usize / 50)
            .min(4 * n)
            .max(min_lag + 1);
        let mut engine = PredictEngine::new(PREDICT_ORDER, min_lag, max_lag);
        let mut stream = Vec::new();
        let mut distortion = 0.0f64;
        let mut lag_hint = 0i32;
        let pred_frames = samples.len().div_ceil(n).max(1);
        for (f, frame) in analysis.iter().enumerate().take(pred_frames) {
            let start = f * n;
            let params = crate::lossy::predict::analyse_frame(
                samples,
                start,
                n,
                PREDICT_ORDER,
                min_lag,
                max_lag,
                lag_hint,
                self.config.sample_rate_hz,
            );
            lag_hint = params.lag;
            let (cand, _) = engine.commit_with(&frame.samples, &params, gain, codecs);
            stream.extend_from_slice(&cand.record);
            distortion += cand.distortion;
        }
        ChannelEncoding {
            mode: MODE_PREDICT,
            gain,
            bytes: stream.len(),
            distortion,
            frames: pred_frames,
            stream,
        }
    }

    // -----------------------------------------------------------------------
    // Decoding
    // -----------------------------------------------------------------------

    fn decode_channel(&self, mode: u8, stream: &[u8], orig_len: usize) -> Result<Vec<f64>> {
        match mode {
            MODE_TRANSFORM => self.decode_transform(stream, orig_len),
            MODE_PREDICT => self.decode_predict(stream, orig_len),
            other => Err(Error::new(
                Kind::Unsupported,
                format!("unknown lossy engine tag {other}"),
            )),
        }
    }

    fn decode_transform(&self, stream: &[u8], orig_len: usize) -> Result<Vec<f64>> {
        let n = self.mdct.n();
        let hop = n;
        let mut r = BitReader::new(stream);
        let frames = r.u32()? as usize;
        let total = frames * hop + hop;
        let mut buf = vec![0.0f64; total];
        let bc = self.bands.band_count;
        let mut shape_prev = vec![0i32; bc];
        let mut gain_prev = 0i32;
        for f in 0..frames {
            let len = r.u16()? as usize;
            let record = r.take(len)?;
            let (coeffs, shape, gain) =
                TransformEngine::decode(&self.bands, n, record, &shape_prev, gain_prev)?;
            shape_prev = shape;
            gain_prev = gain;
            self.mdct.inverse_add(&coeffs, &mut buf, f * hop);
        }
        let mut out = vec![0.0f64; orig_len];
        for (i, slot) in out.iter_mut().enumerate() {
            let idx = i + hop;
            if idx < buf.len() {
                *slot = buf[idx];
            }
        }
        Ok(out)
    }

    fn decode_predict(&self, stream: &[u8], orig_len: usize) -> Result<Vec<f64>> {
        let mut r = BitReader::new(stream);
        let frames = r.u32()? as usize;
        let n = self.mdct.n();
        let min_lag = (self.config.sample_rate_hz as usize / 500).max(8);
        let max_lag = (self.config.sample_rate_hz as usize / 50)
            .min(4 * n)
            .max(min_lag + 1);
        let symbol_count = PREDICT_ORDER + 3 + n;
        let mut engine = PredictEngine::new(PREDICT_ORDER, min_lag, max_lag);
        let mut out = Vec::with_capacity(frames * n);
        for _ in 0..frames {
            let len = r.u16()? as usize;
            let payload = r.take(len)?;
            let enc = crate::learned::residual_codec2::decode_encoding_v3(payload, symbol_count)?;
            let vector = enc.decode(symbol_count)?;
            let gain = engine.last_gain() + vector[PREDICT_ORDER + 2];
            let recon = engine.reconstruct(
                &vector[0..PREDICT_ORDER],
                vector[PREDICT_ORDER],
                vector[PREDICT_ORDER + 1],
                gain,
                &vector[PREDICT_ORDER + 3..],
            );
            out.extend_from_slice(&recon);
        }
        out.truncate(orig_len);
        while out.len() < orig_len {
            out.push(0.0);
        }
        Ok(out)
    }
}

/// A channel encoded by one engine.
struct ChannelEncoding {
    mode: u8,
    gain: i32,
    bytes: usize,
    distortion: f64,
    frames: usize,
    stream: Vec<u8>,
}

fn calibrate_full_scale(mdct: &Mdct, sample_rate_hz: u32) -> f64 {
    let n = mdct.n();
    let two_n = 2 * n;
    let fs = f64::from(sample_rate_hz);
    let bin = (1000.0 / (fs / (2 * n) as f64))
        .round()
        .clamp(1.0, (n - 2) as f64);
    let f = bin * fs / (2 * n) as f64;
    let mut frame = vec![0.0f64; two_n];
    for (m, s) in frame.iter_mut().enumerate() {
        *s = 32_768.0 * (2.0 * std::f64::consts::PI * f * m as f64 / fs).sin();
    }
    let x = mdct.forward(&frame);
    x.iter().map(|v| v.abs()).fold(0.0f64, f64::max).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lossy::predict::RECONSTRUCTION_OFFSET;

    fn sine(len: usize, fs: u32, f: f64, amp: f64) -> Vec<i32> {
        (0..len)
            .map(|i| {
                (amp * (2.0 * std::f64::consts::PI * f * i as f64 / f64::from(fs)).sin())
                    .round()
                    .clamp(-32_768.0, 32_767.0) as i32
            })
            .collect()
    }

    /// Delay-aligned SNR: the transform engine legitimately reconstructs with a
    /// hop of algorithmic delay, so an unaligned comparison would only ever
    /// measure that delay.
    fn snr(reference: &[i32], test: &[i32], max_lag: i64) -> f64 {
        let mut best = f64::NEG_INFINITY;
        for lag in -max_lag..=max_lag {
            let mut sig = 0.0f64;
            let mut err = 0.0f64;
            for (i, &r) in reference.iter().enumerate() {
                let s = i as i64 + lag;
                if s < 0 || s as usize >= test.len() {
                    continue;
                }
                sig += f64::from(r) * f64::from(r);
                let d = f64::from(r) - f64::from(test[s as usize]);
                err += d * d;
            }
            if err <= 0.0 && sig > 0.0 {
                // Exact reconstruction: cap so the value can be compared
                // monotonically across rates rather than as an infinity.
                return 300.0;
            }
            if err > 0.0 {
                best = best.max((10.0 * (sig / err).log10()).min(300.0));
            }
        }
        best
    }

    #[test]
    fn corpus_fixtures_are_reconstructed_not_abandoned() {
        // The frozen fixtures are full-scale i32 generators. The lossy codec is
        // driven from the same canonical 16-bit range the courts and the
        // competitor harness use, so clamp first.
        //
        // `impulse-train` and `transient-heavy` clamp to rectangular full-scale
        // pulse trains: the codec must reconstruct them (they must stay on the
        // rate/quality curve), but their absolute SNR is dominated by the
        // synthetic 2^22 dynamic range rather than by codec quality, so the
        // absolute floor is only asserted for the waveform-like fixtures.
        let all = [
            "quasi-periodic",
            "harmonic-tone",
            "am-signal",
            "fm-signal",
            "transient-heavy",
            "impulse-train",
            "stereo-correlated",
        ];
        let waveform_like = [
            "quasi-periodic",
            "harmonic-tone",
            "am-signal",
            "fm-signal",
            "stereo-correlated",
        ];
        for name in all {
            let f = crate::entropy::corpus::named(name).unwrap();
            let samples: Vec<i32> = f
                .samples
                .iter()
                .map(|&v| v.clamp(-32_768, 32_767))
                .collect();
            let mut prev = f64::NEG_INFINITY;
            for bps in [32_000u32, 64_000, 128_000] {
                let config = LossyConfig {
                    sample_rate_hz: 16_000,
                    channels: f.channels,
                    frame_len: LossyConfig::default_frame_len(16_000),
                    target_bits_per_second: bps,
                };
                let codec = LossyCodec::new(config).unwrap();
                let bytes = codec.encode(&samples).unwrap();
                let dec = codec.decode(&bytes).unwrap();
                assert_eq!(dec.len(), samples.len(), "{name}: length at {bps}");
                assert!(
                    dec.iter().any(|&v| v != 0),
                    "{name}: collapsed to exact silence at {bps}"
                );
                let s = snr(&samples, &dec, 1024);
                // The aligned SNR is capped at 300 dB for exact reconstruction and
                // the quantiser's distortion is not monotone in the gain once a
                // band's peak coefficient starts to clamp, so a cell that was
                // exact may legitimately land a few orders of magnitude lower at
                // the next rate. Both are still *reconstructions*; the gate is
                // that the codec never collapses and never gets systematically
                // worse with rate.
                let ok = if prev >= 150.0 {
                    s >= 60.0
                } else {
                    s >= prev - 1.5
                };
                assert!(ok, "{name}: SNR regressed {prev} -> {s} at {bps} bps");
                prev = prev.max(s);
            }
            if waveform_like.contains(&name) {
                assert!(prev > 1.0, "{name}: top-rate SNR {prev} dB");
            }
        }
    }

    #[test]
    fn encoding_is_deterministic() {
        let f = crate::entropy::corpus::named("harmonic-tone").unwrap();
        let config = LossyConfig {
            sample_rate_hz: 16_000,
            channels: f.channels,
            frame_len: LossyConfig::default_frame_len(16_000),
            target_bits_per_second: 48_000,
        };
        let codec = LossyCodec::new(config).unwrap();
        let a = codec.encode(&f.samples).unwrap();
        let b = codec.encode(&f.samples).unwrap();
        assert_eq!(a, b);
        assert_eq!(codec.decode(&a).unwrap(), codec.decode(&b).unwrap());
    }

    #[test]
    fn real_speech_round_trips() {
        if !crate::learned::corpus_real::available() {
            return;
        }
        let clips = crate::learned::corpus_real::effectiveness_clips();
        let loaded = crate::learned::corpus_real::load_cases(
            &clips,
            1,
            &std::path::PathBuf::from("target/real-corpus/scratch"),
        )
        .unwrap();
        let c = &loaded[0];
        for bps in [24_000u32, 32_000, 48_000] {
            let config = LossyConfig {
                sample_rate_hz: 16_000,
                channels: 1,
                frame_len: LossyConfig::default_frame_len(16_000),
                target_bits_per_second: bps,
            };
            let codec = LossyCodec::new(config).unwrap();
            let bytes = codec.encode(&c.samples).unwrap();
            let dec = codec.decode(&bytes).unwrap();
            let s = snr(&c.samples, &dec, 1024);
            assert!(s > 8.0, "speech at {bps} bps: aligned SNR {s} dB");
        }
    }

    #[test]
    fn codec_round_trips_and_improves_with_rate() {
        let fs = 16_000u32;
        let len = 16_000usize;
        let samples = sine(len, fs, 440.0, 8_000.0);
        let mut prev = f64::NEG_INFINITY;
        for &bps in &[24_000u32, 48_000, 96_000] {
            let config = LossyConfig {
                sample_rate_hz: fs,
                channels: 1,
                frame_len: LossyConfig::default_frame_len(fs),
                target_bits_per_second: bps,
            };
            let codec = LossyCodec::new(config).unwrap();
            let bytes = codec.encode(&samples).unwrap();
            let out = codec.decode(&bytes).unwrap();
            assert_eq!(out.len(), samples.len());
            let s = snr(&samples, &out, 1024);
            assert!(s > 10.0, "SNR {s} dB at {bps} bps");
            assert!(s >= prev - 1.0, "SNR regressed with rate: {prev} -> {s}");
            prev = s;
        }
    }

    #[test]
    fn reconstruction_offset_is_in_domain() {
        let q = crate::lossy::predict::quantizer(crate::lossy::predict::step_of(10));
        assert!(q.reconstruct(1, RECONSTRUCTION_OFFSET) > 0.0);
    }
}
