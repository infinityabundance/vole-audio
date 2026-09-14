//! Phase 7C.2-A/B: `vole.audio.stream.voice.exp2` — the bit-accurate,
//! mode-specific frame syntax.
//!
//! The `exp1` frame record serialises a *generic* envelope (`mode`, an 8-bit
//! gain delta, an optional 16-bit pitch lag + 8-bit LTP gain, the spectral
//! blob, a 16-bit residual length) and then a *CELP-specific* payload that
//! repeats pitch and gain per subframe. On a CELP frame the generic pitch and
//! gain fields are pure duplication, and the 16-bit length is unnecessary
//! because the payload length follows from the pulse counts. Byte alignment
//! then rounds every record up.
//!
//! Measured on the committed `exp1` serializer for a 320-sample, pitched,
//! VQ-spectrum CELP frame with no pulses:
//!
//! ```text
//! flags+base gain 16 b · mode 8 b · gain delta 8 b · lag+LTP 24 b
//! VQ spectrum 32 b · residual length 16 b · codec id 8 b
//! CELP overhead 4 × 23 b = 92 b
//! = 204 b, padded to 208 b = 26 B = 10.4 kbps before a single pulse
//! ```
//!
//! `exp2` carries the *same information* in the same frame as
//! `spectral tier 2 b + 28-bit index + family 2 b + 4 × 23 b = 128 b = 16 B`,
//! and its payload length is derivable, so no length field is spent. This
//! module is the serializer only: it does not yet replace the live profile,
//! which is a later 7C.2 seal. It is verified by exact round-trip and exact
//! bit accounting, including the floor comparison above.

use crate::error::{Error, Kind, Result};
use crate::voice::VoiceState;
use crate::voice::celp;
use crate::voice::fcelp;
use crate::voice::predict as vp;
use crate::voice::pvq;
use crate::voice::residual;
use crate::voice::vq;

// ---------------------------------------------------------------------------
// Bit I/O (MSB-first)
// ---------------------------------------------------------------------------

/// MSB-first bit writer with exact accounting.
pub struct BitWriter {
    out: Vec<u8>,
    acc: u8,
    n: u8,
    total: usize,
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    /// An empty writer.
    pub fn new() -> BitWriter {
        BitWriter {
            out: Vec::new(),
            acc: 0,
            n: 0,
            total: 0,
        }
    }

    /// Append one bit.
    pub fn bit(&mut self, b: bool) {
        self.acc = (self.acc << 1) | u8::from(b);
        self.n += 1;
        self.total += 1;
        if self.n == 8 {
            self.out.push(self.acc);
            self.acc = 0;
            self.n = 0;
        }
    }

    /// Append the low `n` bits of `v`, most significant first.
    pub fn bits(&mut self, v: u32, n: u8) {
        for i in (0..n).rev() {
            self.bit((v >> i) & 1 == 1);
        }
    }

    /// Append whole bytes as 8 bits each (bit-aligned, no padding).
    pub fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.bits(u32::from(x), 8);
        }
    }

    /// Exact number of information bits written.
    pub fn len_bits(&self) -> usize {
        self.total
    }

    /// The byte artifact (zero-padded to a byte boundary).
    pub fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.acc <<= 8 - self.n;
            self.out.push(self.acc);
        }
        self.out
    }
}

/// MSB-first bit reader.
pub struct BitReader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    /// A reader over `b`.
    pub fn new(b: &'a [u8]) -> BitReader<'a> {
        BitReader { b, pos: 0 }
    }

    fn bit(&mut self) -> Result<bool> {
        let byte = self.pos >> 3;
        if byte >= self.b.len() {
            return Err(Error::malformed("voice.exp2 bits exhausted"));
        }
        let bit = (self.b[byte] >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Ok(bit == 1)
    }

    /// Read `n` bits, most significant first.
    pub fn bits(&mut self, n: u8) -> Result<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | u32::from(self.bit()?);
        }
        Ok(v)
    }

    /// Read `n` whole bytes.
    pub fn bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.bits(8)? as u8);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Frame syntax
// ---------------------------------------------------------------------------

/// Bits of the spectral tier selector.
pub const TIER_BITS: u8 = 2;
/// Bits of the excitation family selector.
pub const FAMILY_BITS: u8 = 2;

/// Spectral description. The tier is carried in the frame, so a low-rate mode
/// does not pay for the full quantiser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Spectral {
    /// The frozen 28-bit split-MSVQ index.
    Vq([u8; vq::VQ_INDEX_BYTES]),
    /// The **nested** operating point (7C.2-F): stage 0 of each split only, with
    /// stage 1 implied zero. The MSVQ is embedded, so this is a legal, coarser
    /// spectrum that costs 16 bits instead of 28 — not a second codebook. At
    /// 3.2 kbps the full index is 53 % of the frame, which is why a nested point
    /// has to exist before anything else can be transmitted there.
    Vq0([u8; vq::VQ_INDEX_BYTES]),
    /// Scalar reflection codes at one ladder width; order is the code count.
    Scalar { width: u8, codes: Vec<i32> },
}

impl Spectral {
    /// Exact information bits for this description.
    pub fn bits(&self) -> usize {
        match self {
            Spectral::Vq(_) => usize::from(TIER_BITS) + vq::VQ_INDEX_BYTES * 8,
            Spectral::Vq0(_) => usize::from(TIER_BITS) + vq::STAGE0_BITS as usize,
            Spectral::Scalar { width, codes } => {
                usize::from(TIER_BITS) + 5 + codes.len() * usize::from(*width)
            }
        }
    }

    fn write(&self, w: &mut BitWriter) {
        match self {
            Spectral::Vq(idx) => {
                w.bits(0, TIER_BITS);
                w.bytes(idx);
            }
            Spectral::Vq0(idx) => {
                w.bits(2, TIER_BITS);
                let (s0, s1) = vq::stage0(idx);
                w.bits(u32::from(s0), 8);
                w.bits(u32::from(s1), 8);
            }
            Spectral::Scalar { width, codes } => {
                w.bits(1, TIER_BITS);
                let oi = vp::ORDER_LADDER
                    .iter()
                    .position(|&o| o == codes.len())
                    .unwrap_or(0) as u32;
                let wi = vp::WIDTH_LADDER
                    .iter()
                    .position(|&x| x == *width)
                    .unwrap_or(0) as u32;
                w.bits(oi, 3);
                w.bits(wi, 2);
                let mask = (1u32 << width) - 1;
                for &c in codes {
                    w.bits((c as u32) & mask, *width);
                }
            }
        }
    }

    fn read(r: &mut BitReader<'_>) -> Result<Spectral> {
        match r.bits(TIER_BITS)? {
            0 => {
                let v = r.bytes(vq::VQ_INDEX_BYTES)?;
                let mut idx = [0u8; vq::VQ_INDEX_BYTES];
                idx.copy_from_slice(&v);
                Ok(Spectral::Vq(idx))
            }
            1 => {
                let oi = r.bits(3)? as usize;
                let wi = r.bits(2)? as usize;
                let order = *vp::ORDER_LADDER
                    .get(oi)
                    .ok_or_else(|| Error::malformed("voice.exp2 order outside the ladder"))?;
                let width = *vp::WIDTH_LADDER
                    .get(wi)
                    .ok_or_else(|| Error::malformed("voice.exp2 width outside the ladder"))?;
                let mut codes = Vec::with_capacity(order);
                for _ in 0..order {
                    let raw = r.bits(width)?;
                    let shift = 32 - u32::from(width);
                    codes.push(((raw << shift) as i32) >> shift);
                }
                Ok(Spectral::Scalar { width, codes })
            }
            2 => {
                let s0 = r.bits(8)? as u8;
                let s1 = r.bits(8)? as u8;
                let idx = vq::pack_stage0(s0, s1);
                Ok(Spectral::Vq0(idx))
            }
            _ => Err(Error::new(
                Kind::Unsupported,
                "voice.exp2 spectral tier reserved",
            )),
        }
    }
}

/// Excitation description.
#[derive(Debug, Clone, PartialEq)]
pub enum Excitation {
    /// Scalar dead-zone residual: frame gain plus an entropy artifact, whose
    /// length is explicit because the payload is self-delimiting only within
    /// the entropy coder.
    Scalar { gain: i32, payload: Vec<u8> },
    /// ACELP. **No length field**: the record length follows from the pulse
    /// counts, which are in the stream.
    Celp(celp::Params),
    /// The low-information fallback core: a deterministic, packet-local shaped
    /// stochastic excitation (white excitation through the synthesis filter, so
    /// its spectrum follows the transmitted envelope) with one level and a
    /// three-bit seed the encoder chooses. This is what makes a 64-bit (3.2
    /// kbps) frame expressible at all.
    Noise { gain: i32, seed: u8 },
    /// 7C.2-E track ACELP: a **fractional** long-term predictor and an
    /// interleaved-track algebraic innovation. Its pulse count is the codebook
    /// class, so it spends no count field and cannot truncate a subframe. It is
    /// a *second* core, selected against the others by exact bits and measured
    /// distortion, never forced.
    Ac3(fcelp::Params),
    /// 7C.2-G transform escape: the LPC residual coded in an orthonormal
    /// transform domain with PVQ shapes and one gain per 5 ms block.
    ///
    /// Carried as a **sub-mode of the fallback family** (family 2, one
    /// discriminator bit) rather than as a new family, so every ACELP frame is
    /// untouched: adding a fifth family would cost one bit on *every* frame,
    /// including the low rates where this core cannot fit at all.
    Tcx {
        /// Pulses per block, identical in every block.
        k: usize,
        /// One gain code per block.
        gains: Vec<i32>,
        /// One PVQ index per block.
        indices: Vec<u32>,
    },
}

impl Excitation {
    /// Exact information bits for this description.
    pub fn bits(&self) -> usize {
        match self {
            Excitation::Scalar { payload, .. } => {
                usize::from(FAMILY_BITS) + 8 + 12 + payload.len() * 8
            }
            Excitation::Celp(p) => {
                // Multirate side information (7C.2-D): one frame pitch anchor plus
                // 4-bit contour deltas, one pitch gain plus 3-bit deltas, one
                // innovation gain plus 4-bit deltas, and a single frame pulse count.
                let nsub = p.subframes.len();
                let tail = nsub.saturating_sub(1);
                let count = p
                    .subframes
                    .first()
                    .map(|s| s.pulses.len())
                    .unwrap_or(0)
                    .min((1usize << celp::COUNT_BITS) - 1);
                usize::from(FAMILY_BITS)
                    + usize::from(celp::LAG_BITS)
                    + 4 * tail
                    + usize::from(celp::PITCH_GAIN_BITS)
                    + 3 * tail
                    + usize::from(celp::GAIN_BITS)
                    + 4 * tail
                    + usize::from(celp::COUNT_BITS)
                    + nsub
                        * (usize::from(celp::position_bits(count))
                            + count * usize::from(celp::SIGN_BITS))
            }
            Excitation::Noise { .. } => usize::from(FAMILY_BITS) + 1 + 6 + 3,
            Excitation::Tcx { k, gains, indices } => {
                let blocks = indices.len().max(gains.len());
                usize::from(FAMILY_BITS)
                    + 1
                    + usize::from(pvq::K_BITS)
                    + usize::from(celp::GAIN_BITS)
                    + 3 * blocks.saturating_sub(1)
                    + blocks * usize::from(pvq::index_bits(*k))
            }
            Excitation::Ac3(p) => {
                let nsub = p.subframes.len();
                let tail = nsub.saturating_sub(1);
                let per = p.per_track.clamp(1, fcelp::MAX_PER_TRACK);
                usize::from(FAMILY_BITS)
                    + 2
                    + usize::from(fcelp::LAG_Q_BITS)
                    + 4 * tail
                    + usize::from(celp::PITCH_GAIN_BITS)
                    + 3 * tail
                    + usize::from(celp::GAIN_BITS)
                    + 4 * tail
                    + nsub * fcelp::TRACKS * usize::from(fcelp::track_bits(per))
            }
        }
    }

    fn write(&self, w: &mut BitWriter) {
        match self {
            Excitation::Scalar { gain, payload } => {
                w.bits(0, FAMILY_BITS);
                w.bits((gain - vp::GAIN_MIN).clamp(0, 255) as u32, 8);
                w.bits(payload.len().min(4095) as u32, 12);
                w.bytes(payload);
            }
            Excitation::Celp(p) => {
                w.bits(1, FAMILY_BITS);
                let count = p
                    .subframes
                    .first()
                    .map(|s| s.pulses.len())
                    .unwrap_or(0)
                    .min((1usize << celp::COUNT_BITS) - 1);
                // Lag anchor + contour.
                let mut lag = p
                    .subframes
                    .first()
                    .map(|s| s.lag)
                    .unwrap_or(vp::MIN_LAG as i32);
                w.bits(
                    (lag - vp::MIN_LAG as i32).clamp(0, (1 << celp::LAG_BITS) - 1) as u32,
                    celp::LAG_BITS,
                );
                for sub in p.subframes.iter().skip(1) {
                    let d = (sub.lag - lag).clamp(-8, 7);
                    w.bits((d as u32) & 0xF, 4);
                    lag = (lag + d).clamp(vp::MIN_LAG as i32, vp::MAX_LAG as i32);
                }
                // Pitch gain + deltas.
                let mut pg = p.subframes.first().map(|s| s.pitch_gain).unwrap_or(0);
                w.bits(
                    pg.clamp(0, celp::PITCH_GAIN_LEVELS - 1) as u32,
                    celp::PITCH_GAIN_BITS,
                );
                for sub in p.subframes.iter().skip(1) {
                    let d = (sub.pitch_gain - pg).clamp(-4, 3);
                    w.bits((d as u32) & 0x7, 3);
                    pg = (pg + d).clamp(0, celp::PITCH_GAIN_LEVELS - 1);
                }
                // Innovation gain + deltas.
                let mut g = p.subframes.first().map(|s| s.gain).unwrap_or(0);
                w.bits(g.clamp(0, celp::GAIN_LEVELS - 1) as u32, celp::GAIN_BITS);
                for sub in p.subframes.iter().skip(1) {
                    let d = (sub.gain - g).clamp(-8, 7);
                    w.bits((d as u32) & 0xF, 4);
                    g = (g + d).clamp(0, celp::GAIN_LEVELS - 1);
                }
                // One pulse count for the whole frame, then the pulses.
                w.bits(count as u32, celp::COUNT_BITS);
                for sub in &p.subframes {
                    let mut ordered: Vec<(u8, bool)> =
                        sub.pulses.iter().take(count).copied().collect();
                    ordered.sort_unstable_by_key(|&(pos, _)| pos);
                    let positions: Vec<u8> = ordered.iter().map(|&(p, _)| p).collect();
                    let pb = celp::position_bits(count);
                    if pb > 0 {
                        w.bits(celp::rank_positions(&positions) as u32, pb);
                    }
                    for &(_, positive) in &ordered {
                        w.bits(u32::from(positive), celp::SIGN_BITS);
                    }
                }
            }
            Excitation::Noise { gain, seed } => {
                w.bits(2, FAMILY_BITS);
                w.bits(0, 1);
                w.bits((*gain).clamp(0, celp::GAIN_LEVELS - 1) as u32, 6);
                w.bits(u32::from(*seed) & 0x7, 3);
            }
            Excitation::Tcx { k, gains, indices } => {
                w.bits(2, FAMILY_BITS);
                w.bits(1, 1);
                let k = (*k).min(pvq::MAX_PULSES);
                w.bits(k as u32, pvq::K_BITS);
                // One anchor gain plus 3-bit block deltas.
                let anchor = gains.first().copied().unwrap_or(0);
                w.bits(
                    anchor.clamp(0, celp::GAIN_LEVELS - 1) as u32,
                    celp::GAIN_BITS,
                );
                let mut prev = anchor.clamp(0, celp::GAIN_LEVELS - 1);
                for &g in gains.iter().skip(1) {
                    let d = (g - prev).clamp(-4, 3);
                    w.bits((d as u32) & 0x7, 3);
                    prev = (prev + d).clamp(0, celp::GAIN_LEVELS - 1);
                }
                let bits = pvq::index_bits(k);
                for &idx in indices {
                    w.bits(idx, bits);
                }
            }
            Excitation::Ac3(p) => {
                w.bits(3, FAMILY_BITS);
                let per = p.per_track.clamp(1, fcelp::MAX_PER_TRACK);
                w.bits((per - 1) as u32, 2);
                if p.subframes.is_empty() {
                    return;
                }
                // Fractional lag anchor + quarter-sample contour.
                let mut lag = p.subframes[0].lag_q;
                w.bits(
                    (lag - fcelp::LAG_Q_MIN).clamp(0, (1 << fcelp::LAG_Q_BITS) - 1) as u32,
                    fcelp::LAG_Q_BITS,
                );
                for sub in p.subframes.iter().skip(1) {
                    let d = (sub.lag_q - lag).clamp(-8, 7);
                    w.bits((d as u32) & 0xF, 4);
                    lag = (lag + d).clamp(fcelp::LAG_Q_MIN, fcelp::LAG_Q_MAX);
                }
                // Pitch gain + deltas.
                let mut pg = p.subframes[0].pitch_gain;
                w.bits(
                    pg.clamp(0, celp::PITCH_GAIN_LEVELS - 1) as u32,
                    celp::PITCH_GAIN_BITS,
                );
                for sub in p.subframes.iter().skip(1) {
                    let d = (sub.pitch_gain - pg).clamp(-4, 3);
                    w.bits((d as u32) & 0x7, 3);
                    pg = (pg + d).clamp(0, celp::PITCH_GAIN_LEVELS - 1);
                }
                // Innovation gain + deltas.
                let mut g = p.subframes[0].gain;
                w.bits(g.clamp(0, celp::GAIN_LEVELS - 1) as u32, celp::GAIN_BITS);
                for sub in p.subframes.iter().skip(1) {
                    let d = (sub.gain - g).clamp(-8, 7);
                    w.bits((d as u32) & 0xF, 4);
                    g = (g + d).clamp(0, celp::GAIN_LEVELS - 1);
                }
                // One index per track per subframe.
                let tb = fcelp::track_bits(per);
                for sub in &p.subframes {
                    for t in 0..fcelp::TRACKS {
                        let mut poss: Vec<usize> = Vec::new();
                        let mut signs: Vec<bool> = Vec::new();
                        for &(pos, positive) in &sub.pulses {
                            let pos = usize::from(pos);
                            if pos % fcelp::TRACKS == t {
                                poss.push(pos / fcelp::TRACKS);
                                signs.push(positive);
                            }
                        }
                        if poss.len() != per {
                            // Only a well-formed frame is written; a malformed one
                            // is padded so the writer never panics. The encoder
                            // only ever offers well-formed frames.
                            w.bits(0, tb);
                            continue;
                        }
                        w.bits(fcelp::rank_track(&poss, &signs) as u32, tb);
                    }
                }
            }
        }
    }

    fn read(r: &mut BitReader<'_>, frame_len: usize) -> Result<Excitation> {
        match r.bits(FAMILY_BITS)? {
            0 => {
                let gain = i32::from(r.bits(8)? as u8) + vp::GAIN_MIN;
                let len = r.bits(12)? as usize;
                let payload = r.bytes(len)?;
                Ok(Excitation::Scalar { gain, payload })
            }
            1 => {
                let nsub = celp::subframes(frame_len);
                // Lag anchor + contour.
                let anchor = vp::MIN_LAG as i32 + r.bits(celp::LAG_BITS)? as i32;
                let mut lags = vec![anchor.clamp(vp::MIN_LAG as i32, vp::MAX_LAG as i32)];
                for _ in 1..nsub {
                    let d = sign_extend(r.bits(4)?, 4);
                    let prev = *lags.last().unwrap();
                    lags.push((prev + d).clamp(vp::MIN_LAG as i32, vp::MAX_LAG as i32));
                }
                // Pitch gain + deltas.
                let mut pgs = vec![r.bits(celp::PITCH_GAIN_BITS)? as i32];
                for _ in 1..nsub {
                    let d = sign_extend(r.bits(3)?, 3);
                    let prev = *pgs.last().unwrap();
                    pgs.push((prev + d).clamp(0, celp::PITCH_GAIN_LEVELS - 1));
                }
                // Innovation gain + deltas.
                let mut gs = vec![r.bits(celp::GAIN_BITS)? as i32];
                for _ in 1..nsub {
                    let d = sign_extend(r.bits(4)?, 4);
                    let prev = *gs.last().unwrap();
                    gs.push((prev + d).clamp(0, celp::GAIN_LEVELS - 1));
                }
                // One pulse count, then the pulses.
                let count = r.bits(celp::COUNT_BITS)? as usize;
                if count > celp::MAX_PULSES {
                    return Err(Error::malformed("voice.exp2 pulse count above the bound"));
                }
                let pb = celp::position_bits(count);
                let mut p = celp::Params::default();
                for i in 0..nsub {
                    let positions = if pb > 0 {
                        celp::unrank_positions(u64::from(r.bits(pb)?), count)
                    } else {
                        Vec::new()
                    };
                    let mut pulses = Vec::with_capacity(count);
                    for &pos in &positions {
                        let positive = r.bits(celp::SIGN_BITS)? == 1;
                        pulses.push((pos, positive));
                    }
                    p.subframes.push(celp::Subframe {
                        lag: lags[i],
                        pitch_gain: pgs[i],
                        gain: gs[i],
                        pulses,
                    });
                }
                Ok(Excitation::Celp(p))
            }
            2 => {
                if r.bits(1)? == 0 {
                    let gain = r.bits(6)? as i32;
                    let seed = r.bits(3)? as u8;
                    return Ok(Excitation::Noise { gain, seed });
                }
                let blocks = frame_len.div_ceil(pvq::BLOCK);
                if frame_len == 0 || !frame_len.is_multiple_of(pvq::BLOCK) {
                    return Err(Error::malformed(
                        "voice.exp2 escape frame is not a whole number of blocks",
                    ));
                }
                let k = r.bits(pvq::K_BITS)? as usize;
                if k > pvq::MAX_PULSES {
                    return Err(Error::malformed(
                        "voice.exp2 escape pulse count above the bound",
                    ));
                }
                let anchor = r.bits(celp::GAIN_BITS)? as i32;
                let mut gains = vec![anchor.clamp(0, celp::GAIN_LEVELS - 1)];
                for _ in 1..blocks {
                    let d = sign_extend(r.bits(3)?, 3);
                    let prev = *gains.last().unwrap();
                    gains.push((prev + d).clamp(0, celp::GAIN_LEVELS - 1));
                }
                let bits = pvq::index_bits(k);
                let mut indices = Vec::with_capacity(blocks);
                for _ in 0..blocks {
                    indices.push(if bits > 0 { r.bits(bits)? } else { 0 });
                }
                Ok(Excitation::Tcx { k, gains, indices })
            }
            3 => {
                let nsub = celp::subframes(frame_len);
                let per = r.bits(2)? as usize + 1;
                if per > fcelp::MAX_PER_TRACK {
                    return Err(Error::malformed(
                        "voice.exp2 ACELP codebook class above the bound",
                    ));
                }
                // Fractional lag anchor + contour.
                let anchor = fcelp::LAG_Q_MIN + r.bits(fcelp::LAG_Q_BITS)? as i32;
                let mut lags = vec![anchor.clamp(fcelp::LAG_Q_MIN, fcelp::LAG_Q_MAX)];
                for _ in 1..nsub {
                    let d = sign_extend(r.bits(4)?, 4);
                    let prev = *lags.last().unwrap();
                    lags.push((prev + d).clamp(fcelp::LAG_Q_MIN, fcelp::LAG_Q_MAX));
                }
                // Pitch gain + deltas.
                let mut pgs = vec![r.bits(celp::PITCH_GAIN_BITS)? as i32];
                for _ in 1..nsub {
                    let d = sign_extend(r.bits(3)?, 3);
                    let prev = *pgs.last().unwrap();
                    pgs.push((prev + d).clamp(0, celp::PITCH_GAIN_LEVELS - 1));
                }
                // Innovation gain + deltas.
                let mut gs = vec![r.bits(celp::GAIN_BITS)? as i32];
                for _ in 1..nsub {
                    let d = sign_extend(r.bits(4)?, 4);
                    let prev = *gs.last().unwrap();
                    gs.push((prev + d).clamp(0, celp::GAIN_LEVELS - 1));
                }
                // One index per track per subframe.
                let tb = fcelp::track_bits(per);
                let mut p = fcelp::Params {
                    per_track: per,
                    subframes: Vec::with_capacity(nsub),
                };
                for i in 0..nsub {
                    let mut pulses: Vec<(u8, bool)> = Vec::with_capacity(per * fcelp::TRACKS);
                    for t in 0..fcelp::TRACKS {
                        let (local, signs) = fcelp::unrank_track(u64::from(r.bits(tb)?), per);
                        for (k, &lp) in local.iter().enumerate() {
                            let pos = t + lp * fcelp::TRACKS;
                            if pos >= fcelp::SUB_LEN {
                                return Err(Error::malformed(
                                    "voice.exp2 ACELP track position out of range",
                                ));
                            }
                            pulses.push((pos as u8, signs[k]));
                        }
                    }
                    pulses.sort_unstable_by_key(|&(pos, _)| pos);
                    p.subframes.push(fcelp::Subframe {
                        lag_q: lags[i],
                        pitch_gain: pgs[i],
                        gain: gs[i],
                        pulses,
                    });
                }
                Ok(Excitation::Ac3(p))
            }
            _ => Err(Error::new(
                Kind::Unsupported,
                "voice.exp2 excitation family reserved",
            )),
        }
    }
}

/// Sign-extend the low `n` bits of `v`.
fn sign_extend(v: u32, n: u8) -> i32 {
    let shift = 32 - u32::from(n);
    ((v << shift) as i32) >> shift
}

/// A deterministic, packet-local white sample in `[-1, 1)` from a three-bit seed
/// and the sample index. Depends on nothing but its arguments, so a lost packet
/// cannot perturb a later one.
fn noise_sample(seed: u8, n: usize) -> f64 {
    let mut x = 0x9E37_79B9_7F4A_7C15u64
        .wrapping_add(u64::from(seed).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add((n as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x as f64 / u64::MAX as f64) * 2.0 - 1.0
}

/// One `exp2` frame: a tiered spectrum plus one excitation family.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame2 {
    /// Spectral description.
    pub spectral: Spectral,
    /// Excitation description.
    pub excitation: Excitation,
}

impl Frame2 {
    /// Exact information bits of this frame (before byte padding).
    pub fn bits(&self) -> usize {
        self.spectral.bits() + self.excitation.bits()
    }

    /// Serialise the frame and report the exact information bit count beside
    /// the byte artifact, so a caller can enforce a hard bit budget.
    pub fn write(&self) -> (Vec<u8>, usize) {
        let mut w = BitWriter::new();
        self.spectral.write(&mut w);
        self.excitation.write(&mut w);
        let bits = w.len_bits();
        (w.finish(), bits)
    }

    /// Parse a frame for a frame of `frame_len` samples.
    pub fn read(bytes: &[u8], frame_len: usize) -> Result<Frame2> {
        let mut r = BitReader::new(bytes);
        let spectral = Spectral::read(&mut r)?;
        let excitation = Excitation::read(&mut r, frame_len)?;
        Ok(Frame2 {
            spectral,
            excitation,
        })
    }

    /// The synthesis inputs this frame decodes to: reflection codes and the
    /// quantiser width for the shared synthesis loop.
    pub fn synthesis_codes(&self) -> (Vec<i32>, u8) {
        match &self.spectral {
            Spectral::Vq(idx) | Spectral::Vq0(idx) => {
                (vq::decode_index(idx), vq::VQ_INTERNAL_WIDTH)
            }
            Spectral::Scalar { width, codes } => (codes.clone(), *width),
        }
    }

    /// The smallest legal frame: the full-precision spectrum with the
    /// low-information fallback core at its lowest level. Its fitting inside 64
    /// bits is the hard-rate conformance property required by §5 of the
    /// charter — every declared rate must admit at least one legal
    /// representation.
    pub fn minimal() -> Frame2 {
        Frame2 {
            spectral: Spectral::Vq([0; vq::VQ_INDEX_BYTES]),
            excitation: Excitation::Noise { gain: 0, seed: 0 },
        }
    }

    /// Decode this frame into reconstructed samples, advancing `state` exactly
    /// as the `exp1` decoder's shared synthesis loop does. Scalar frames use the
    /// per-subframe gain geometry; CELP frames render the reconstructed shot.
    pub fn decode_into(&self, state: &mut VoiceState, frame_len: usize) -> Result<Vec<i32>> {
        let (k_q, width) = self.synthesis_codes();
        let samples = match &self.excitation {
            Excitation::Scalar { gain, payload } => {
                let symbols = residual::decode(payload, frame_len)?;
                vp::synthesize_gains(
                    state,
                    &k_q,
                    width,
                    vp::RESIDUAL_SUB_LEN,
                    &[*gain],
                    0,
                    0,
                    &symbols,
                )
            }
            Excitation::Celp(params) => {
                let shot = celp::reconstruct(params, frame_len)?;
                shot.render(state, &k_q, width)
            }
            Excitation::Noise { gain, seed } => {
                let level = celp::gain_of(*gain);
                let exc: Vec<f64> = (0..frame_len)
                    .map(|n| level * noise_sample(*seed, n))
                    .collect();
                vp::synthesize_excitation(state, &k_q, width, 0, 0, &exc)
            }
            Excitation::Ac3(params) => {
                let shot = fcelp::reconstruct(params, frame_len)?;
                shot.render(state, &k_q, width)
            }
            Excitation::Tcx { k, gains, indices } => {
                if frame_len == 0 || !frame_len.is_multiple_of(pvq::BLOCK) {
                    return Err(Error::malformed(
                        "voice.exp2 escape frame is not a whole number of blocks",
                    ));
                }
                if *k > pvq::MAX_PULSES {
                    return Err(Error::malformed(
                        "voice.exp2 escape pulse count above the bound",
                    ));
                }
                let blocks = frame_len / pvq::BLOCK;
                if gains.len() != blocks || indices.len() != blocks {
                    return Err(Error::malformed("voice.exp2 escape block count mismatch"));
                }
                let mut coeffs = vec![0.0f64; frame_len];
                for b in 0..blocks {
                    let y = pvq::unrank(u128::from(indices[b]), pvq::BLOCK, *k);
                    let norm = y.iter().map(|v| f64::from(v * v)).sum::<f64>().sqrt();
                    if norm <= 0.0 {
                        continue;
                    }
                    let scale = celp::gain_of(gains[b]) / norm;
                    let block: Vec<f64> = y.iter().map(|v| scale * f64::from(*v)).collect();
                    let r = pvq::idct4(&block);
                    coeffs[b * pvq::BLOCK..(b + 1) * pvq::BLOCK].copy_from_slice(&r);
                }
                vp::synthesize_excitation(state, &k_q, width, 0, 0, &coeffs)
            }
        };
        Ok(samples
            .iter()
            .map(|&x| x.round().clamp(-2.0e9, 2.0e9) as i32)
            .collect())
    }
}

/// The declared 20 ms operating points and their exact frame allowances (bits).
pub const DECLARED_RATES: [(u32, usize); 6] = [
    (3_200, 64),
    (6_000, 120),
    (8_000, 160),
    (9_200, 184),
    (12_000, 240),
    (16_000, 320),
];

/// Exact information bits a family-1 CELP frame costs at `count` pulses per
/// subframe. This is the encoder's real budget authority for the control core:
/// side information is frame-level (7C.2-D), so the old per-subframe overhead
/// model no longer describes the wire and must not be used to derive `maxp`.
fn celp_bits(nsub: usize, count: usize) -> usize {
    let tail = nsub.saturating_sub(1);
    usize::from(FAMILY_BITS)
        + usize::from(celp::LAG_BITS)
        + 4 * tail
        + usize::from(celp::PITCH_GAIN_BITS)
        + 3 * tail
        + usize::from(celp::GAIN_BITS)
        + 4 * tail
        + usize::from(celp::COUNT_BITS)
        + nsub * (usize::from(celp::position_bits(count)) + count * usize::from(celp::SIGN_BITS))
}

/// Diagnostic label for the excitation family a frame selects. Development
/// instrument: it lets an experiment report *which* core won rather than only
/// the resulting distortion.
pub fn family_label(f: &Frame2) -> &'static str {
    match f.excitation {
        Excitation::Scalar { .. } => "scalar",
        Excitation::Celp(_) => "celp",
        Excitation::Noise { .. } => "noise",
        Excitation::Ac3(_) => "acelp",
        Excitation::Tcx { .. } => "tcx",
    }
}

/// Innovation pulse count of a frame (total across subframes). Development
/// instrument.
pub fn pulse_count(f: &Frame2) -> usize {
    match &f.excitation {
        Excitation::Celp(p) => p.subframes.iter().map(|s| s.pulses.len()).sum(),
        Excitation::Ac3(p) => p.subframes.iter().map(|s| s.pulses.len()).sum(),
        Excitation::Tcx { k, indices, .. } => k * indices.len(),
        _ => 0,
    }
}

/// Exact information bits a transform-escape frame costs. Mirrors
/// [`Excitation::bits`] for the escape sub-mode, so the encoder's feasibility
/// check and the wire's accounting cannot disagree.
fn tcx_bits(blocks: usize, k: usize) -> usize {
    usize::from(FAMILY_BITS)
        + 1
        + usize::from(pvq::K_BITS)
        + usize::from(celp::GAIN_BITS)
        + 3 * blocks.saturating_sub(1)
        + blocks * usize::from(pvq::index_bits(k))
}

/// Quantise a precomputed transform of the LPC residual with PVQ at `k` pulses
/// per block.
///
/// The residual is the target minus the short-term filter's zero-input response,
/// which is the same excitation the scalar and noise cores are handed, so all
/// three fallbacks are directly comparable inside the selector. The DCT-IV is
/// orthonormal, so the coefficient norm is the residual norm and the per-block
/// gain means what it says.
///
/// The transform is taken by the caller and shared across `k`: the coefficients
/// depend on the spectral tier but not on the pulse density, so recomputing them
/// per `k` would multiply the most expensive part of the search for nothing.
fn tcx_candidate(
    coeffs: &[f64],
    k: usize,
    spectral: &Spectral,
    frame_bits: usize,
) -> Option<Frame2> {
    let len = coeffs.len();
    if len == 0 || !len.is_multiple_of(pvq::BLOCK) || k == 0 || k > pvq::MAX_PULSES {
        return None;
    }
    let blocks = len / pvq::BLOCK;
    if spectral.bits() + tcx_bits(blocks, k) > frame_bits {
        return None;
    }
    let mut gains = Vec::with_capacity(blocks);
    let mut indices = Vec::with_capacity(blocks);
    for b in 0..blocks {
        let cb = &coeffs[b * pvq::BLOCK..(b + 1) * pvq::BLOCK];
        let norm = cb.iter().map(|v| v * v).sum::<f64>().sqrt();
        gains.push(celp::gain_code(norm));
        let y = pvq::encode_shape(cb, k);
        // `pvq::MAX_PULSES` is chosen so `V(BLOCK, k) ≤ 2^32`, hence the rank fits.
        indices.push(pvq::rank(&y) as u32);
    }
    Some(Frame2 {
        spectral: spectral.clone(),
        excitation: Excitation::Tcx { k, gains, indices },
    })
}

/// Sum of squared error between a frame and its reconstruction.
fn mse(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Samples per envelope subframe of the local perceptual proxy.
const ENV_SUB: usize = 80;
/// The 7C.2-F envelope-proxy weight, recorded for the seal history and used by
/// the test that pins the mechanism.
///
/// Superseded in the codec by [`Options::env_weight`], which is fitted against
/// ViSQOL on the development corpus. At `2.0` the 7C.2-F controlled A/B moved
/// waveform SNR *down* at 6 kbps (+0.24 → −0.69), 8 kbps (+1.04 → +0.73),
/// 9.2 kbps (+1.77 → +1.22) and 16 kbps (+5.04 → +4.80), and up only at
/// 12 kbps (+4.23 → +4.77). That measurement is why the weight became a fitted
/// parameter instead of a constant.
#[cfg(test)]
const ENV_WEIGHT_HISTORY: f64 = 2.0;

/// Temporal-envelope mismatch, in units of energy.
///
/// Raw MSE is the wrong selector for a *stochastic* excitation. When the
/// reconstruction is uncorrelated with the target, which is exactly what the
/// low-rate noise core produces, the MSE-optimal gain is **zero**: silence scores
/// better than correctly-levelled shaped noise. Measured at 3.2 kbps this made
/// the codec output near-silence (SNR 0.00 dB) while claiming to be lossy speech.
/// Adding a short-time envelope term makes a level-matched noisy frame score
/// better than a silent one, which is what the ear hears too.
fn envelope_penalty(target: &[f64], out: &[f64]) -> f64 {
    let n = target.len().min(out.len());
    if n == 0 {
        return 0.0;
    }
    let mut pen = 0.0f64;
    for lo in (0..n).step_by(ENV_SUB) {
        let hi = (lo + ENV_SUB).min(n);
        let m = (hi - lo) as f64;
        let et = (target[lo..hi].iter().map(|x| x * x).sum::<f64>() / m).sqrt();
        let eo = (out[lo..hi].iter().map(|x| x * x).sum::<f64>() / m).sqrt();
        pen += (et - eo) * (et - eo) * m;
    }
    pen
}

/// The selector's local perceptual proxy: waveform error plus a temporal-envelope
/// term. The weights are fitted on the development corpus and then frozen; the
/// external court remains the authority on whether the proxy correlates with it.
fn proxy(target: &[f64], out: &[f64], env_weight: f64) -> f64 {
    mse(target, out) + env_weight * envelope_penalty(target, out)
}

/// Pulse-gain codes spanning the level implied by a residual RMS.
fn noise_gain_codes(rms: f64) -> [i32; 3] {
    let centre = celp::gain_code(rms.max(1.0));
    [
        (centre - 8).clamp(0, celp::GAIN_LEVELS - 1),
        centre.clamp(0, celp::GAIN_LEVELS - 1),
        (centre + 8).clamp(0, celp::GAIN_LEVELS - 1),
    ]
}

/// Which excitation cores the encoder is allowed to offer. The default offers
/// every implemented core; the switches exist so a controlled A/B can attribute
/// a measured delta to one mechanism instead of to the whole phase.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    /// Offer the 7C.2-B/D free-combinatorial CELP core (family 1).
    pub celp: bool,
    /// Offer the 7C.2-E fractional-track ACELP core (family 3).
    pub track_acelp: bool,
    /// Offer the 7C.2-G transform/PVQ escape (fallback family sub-mode).
    pub tcx: bool,
    /// Weight of the temporal-envelope term in the selector proxy.
    ///
    /// `0.0` is plain MSE. The term exists because MSE alone prefers silence to
    /// any reconstruction with correlation below 0.5 (§12–§14). The weight is a
    /// *free parameter* precisely so it can be fitted against a perceptual judge
    /// on the development corpus rather than asserted;
    /// [`DEFAULT_ENV_WEIGHT`] is the fitted value.
    pub env_weight: f64,
}

/// The fitted envelope-proxy weight. See [`Options::env_weight`].
///
/// **Fitted on the development corpus against ViSQOL** (`voice_bench weights`),
/// as §7 prescribes: the proxy's free parameter is fitted on development material
/// and then frozen, and the held-out challenger court is what decides whether it
/// correlates with a perceptual judge. Mean MOS-LQO over five rates:
///
/// ```text
/// w = 0.0   1.250     w = 1.0   1.344     w = 4.0   1.390
/// w = 0.25  1.293     w = 2.0   1.416     w = 8.0   1.345
/// ```
///
/// `2.0` peaks the mean and is best or near-best at *every* rate, whereas `4.0`
/// collapses at 8 kbps (1.248). The fit is from three development cases, so it is
/// recorded as a dev fit rather than a result: **no quality claim attaches to it
/// until the held-out court measures it**, and `voice.exp2` is not yet a live
/// profile. The response is unimodal and shallow, which is what makes the choice
/// defensible rather than a knife edge.
pub const DEFAULT_ENV_WEIGHT: f64 = 2.0;

impl Default for Options {
    fn default() -> Self {
        Options {
            celp: true,
            track_acelp: true,
            // **Measured and disabled** (7C.2-G), following the `SUBFRAME_GAIN`
            // precedent. On the frozen development corpus the
            // escape core is selected in **zero frames** at every declared rate,
            // and offering it costs up to 8864 µs encode p99 at 16 kbps against
            // the 5 ms constitution. A mechanism that wins nothing and breaches
            // the deadline does not ship enabled; the switch is kept so the
            // experiment stays reproducible and so it can be re-judged once the
            // selector objective is fixed.
            tcx: false,
            env_weight: DEFAULT_ENV_WEIGHT,
        }
    }
}

/// The `exp2` frame codec: hard-rate encoding and faithful decoding.
///
/// The encoder is the hard-rate authority. It enumerates representations across
/// the spectral tiers and the excitation families, keeps only those whose
/// **exact** serialised bit count fits the frame allowance, and returns the
/// lowest-distortion survivor. When nothing fits it returns
/// [`Frame2::minimal`], which is legal at every declared rate. It never
/// overshoots, so there is no `OVERSHOOT_LIMIT` in this profile.
pub struct Exp2Codec;

impl Exp2Codec {
    /// Encode one frame into at most `frame_bits` bits; returns the artifact and
    /// the exact information bit count.
    pub fn encode_frame(state: &VoiceState, frame: &[i32], frame_bits: usize) -> (Vec<u8>, usize) {
        Self::encode_frame_with(state, frame, frame_bits, Options::default())
    }

    /// [`Exp2Codec::encode_frame`] with an explicit core selection.
    pub fn encode_frame_with(
        state: &VoiceState,
        frame: &[i32],
        frame_bits: usize,
        sel: Options,
    ) -> (Vec<u8>, usize) {
        let f: Vec<f64> = frame.iter().map(|&v| f64::from(v)).collect();
        let mut best: Option<(f64, usize, Frame2)> = None;
        let cands = vp::analyse(state, frame, 0);
        let nsub = celp::subframes(frame.len()).max(1);
        let rms = (f.iter().map(|v| v * v).sum::<f64>() / f.len().max(1) as f64).sqrt();
        let noise_codes = noise_gain_codes(rms);

        // The fractional-track core is the most expensive analysis, so it is
        // offered exactly once per frame: on the cheapest spectral tier (which is
        // the only one that ever leaves room for pulses) and on the candidate the
        // analysis itself ranks best by residual energy. Offering it on every
        // candidate/tier combination multiplied its cost sixfold for a measured
        // zero quality difference; the encode deadline is a constitution
        // requirement, so the work has to be spent where it is used.
        let best_cand = cands
            .iter()
            .take(3)
            .enumerate()
            .min_by(|a, b| a.1.energy.total_cmp(&b.1.energy))
            .map(|(i, _)| i)
            .unwrap_or(0);

        for (ci, c) in cands.iter().take(3).enumerate() {
            let mut opts: Vec<(Vec<i32>, u8, Spectral)> = Vec::new();
            if c.model.order == vq::VQ_ORDER
                && let Some(lsf) = crate::voice::lsf::reflections_to_lsf(&c.k_raw)
                && let Some(idx) = vq::encode_lsf(&lsf)
            {
                opts.push((
                    vq::decode_index(&idx),
                    vq::VQ_INTERNAL_WIDTH,
                    Spectral::Vq(idx),
                ));
                // 7C.2-F: the nested operating point, offered so the selector can
                // trade spectral precision for transmitted excitation. It is a
                // legal frame the decoder reconstructs exactly, and the encoder
                // measures its consequence through the same round-trip as any
                // other candidate.
                let (s0, s1) = vq::stage0(&idx);
                let idx0 = vq::pack_stage0(s0, s1);
                opts.push((
                    vq::decode_index(&idx0),
                    vq::VQ_INTERNAL_WIDTH,
                    Spectral::Vq0(idx0),
                ));
            }
            opts.push((
                c.k_q.clone(),
                c.model.width,
                Spectral::Scalar {
                    width: c.model.width,
                    codes: c.k_q.clone(),
                },
            ));

            for (si, (k_q, width, spectral)) in opts.into_iter().enumerate() {
                let model = vp::FrameModel { width, ..c.model };
                // Every candidate is scored on its **round-tripped** frame, so the
                // encoder measures exactly what the decoder will reconstruct. This
                // is what makes the differential/contour coding safe: a lag delta
                // the wire clamps cannot desynchronise the two sides.
                let mut consider = |fr: Frame2| {
                    let bits = fr.bits();
                    if bits > frame_bits {
                        return;
                    }
                    let (bytes, _) = fr.write();
                    let Ok(rt) = Frame2::read(&bytes, frame.len()) else {
                        return;
                    };
                    let mut s = state.clone();
                    let Ok(out) = rt.decode_into(&mut s, frame.len()) else {
                        return;
                    };
                    let o: Vec<f64> = out.iter().map(|&v| f64::from(v)).collect();
                    let d = proxy(&f, &o, sel.env_weight);
                    let better = match &best {
                        None => true,
                        Some((bd, bb, _)) => {
                            d < *bd - 1e-9 || ((d - *bd).abs() <= 1e-9 && bits < *bb)
                        }
                    };
                    if better {
                        best = Some((d, bits, fr));
                    }
                };

                // ACELP, within the remaining allowance. The pulse count is
                // derived from the exact serialised size, so it tracks the real
                // budget instead of a stale per-subframe overhead model.
                let maxp = (0..=celp::MAX_PULSES)
                    .filter(|&k| spectral.bits() + celp_bits(nsub, k) <= frame_bits)
                    .max()
                    .unwrap_or(0);
                if sel.celp && spectral.bits() + celp_bits(nsub, maxp) <= frame_bits {
                    let (params, _d, _shot) = celp::analyse(state, &f, &model, &k_q, maxp);
                    consider(Frame2 {
                        spectral: spectral.clone(),
                        excitation: Excitation::Celp(params),
                    });
                }

                // 7C.2-E: the fractional-track core, offered against the control
                // core rather than replacing it. Its pulse count is the codebook
                // class, so each class is one candidate and the selector decides.
                if sel.track_acelp && ci == best_cand && si <= 1 {
                    for per in 1..=fcelp::MAX_PER_TRACK {
                        let dummy = fcelp::Params {
                            per_track: per,
                            subframes: (0..nsub)
                                .map(|_| fcelp::Subframe {
                                    lag_q: fcelp::LAG_Q_MIN,
                                    pitch_gain: 0,
                                    gain: 0,
                                    pulses: Vec::new(),
                                })
                                .collect(),
                        };
                        if spectral.bits() + Excitation::Ac3(dummy).bits() > frame_bits {
                            continue;
                        }
                        let (params, _d, _shot) = fcelp::analyse(state, &f, &model, &k_q, per);
                        consider(Frame2 {
                            spectral: spectral.clone(),
                            excitation: Excitation::Ac3(params),
                        });
                    }
                }

                // Scalar residual across the gain ladder.
                for gain in [-4, 4, 12, 20, 28, 36, 44] {
                    let s = vp::close_loop_gains(
                        state,
                        &f,
                        &model,
                        &k_q,
                        vp::RESIDUAL_SUB_LEN,
                        &[gain],
                    );
                    let payload = residual::encode_best(&s.symbols, &[]);
                    consider(Frame2 {
                        spectral: spectral.clone(),
                        excitation: Excitation::Scalar { gain, payload },
                    });
                }

                // The noise fallback core: level and seed are both chosen.
                for code in noise_codes {
                    for seed in 0..8u8 {
                        consider(Frame2 {
                            spectral: spectral.clone(),
                            excitation: Excitation::Noise { gain: code, seed },
                        });
                    }
                }

                // 7C.2-G: the transform escape, offered at every pulse density
                // that fits. It shares the fallback family, so frames that choose
                // a CELP core pay nothing for its existence. The transform is
                // taken once per tier and shared across the pulse densities.
                if sel.tcx && !f.is_empty() && f.len().is_multiple_of(pvq::BLOCK) {
                    let zeros = vec![0.0f64; f.len()];
                    let mut probe = state.clone();
                    let zir =
                        vp::synthesize_excitation(&mut probe, &k_q, model.width, 0, 0, &zeros);
                    let resid: Vec<f64> = f.iter().zip(&zir).map(|(a, b)| a - b).collect();
                    let coeffs = pvq::forward_blocks(&resid);
                    for k in 1..=pvq::MAX_PULSES {
                        if let Some(fr) = tcx_candidate(&coeffs, k, &spectral, frame_bits) {
                            consider(fr);
                        }
                    }
                }
            }
        }

        let chosen = best.map(|(_, _, fr)| fr).unwrap_or_else(Frame2::minimal);
        chosen.write()
    }

    /// Decode one frame's artifact.
    pub fn decode_frame(
        state: &mut VoiceState,
        bytes: &[u8],
        frame_len: usize,
    ) -> Result<Vec<i32>> {
        Frame2::read(bytes, frame_len)?.decode_into(state, frame_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_celp(n: usize, per: usize) -> celp::Params {
        let subs = celp::subframes(n);
        celp::Params {
            subframes: (0..subs)
                .map(|s| {
                    let mut pulses: Vec<(u8, bool)> = (0..per)
                        .map(|p| (((p * 11 + s * 7) % celp::SUB_LEN) as u8, (s + p) % 2 == 0))
                        .collect();
                    pulses.sort_unstable_by_key(|&(pos, _)| pos);
                    celp::Subframe {
                        lag: 100 + vp::MIN_LAG as i32 + s as i32 * 3,
                        pitch_gain: 8 + s as i32 * 2,
                        gain: 16 + s as i32 * 3,
                        pulses,
                    }
                })
                .collect(),
        }
    }

    #[test]
    fn celp_and_vq_frames_round_trip_exactly() {
        for n in [160usize, 320] {
            for per in 0..=celp::MAX_PULSES {
                let f = Frame2 {
                    spectral: Spectral::Vq([1, 2, 3, 4]),
                    excitation: Excitation::Celp(sample_celp(n, per)),
                };
                let (bytes, bits) = f.write();
                assert_eq!(bits, f.bits(), "accounting must be exact");
                assert!(bytes.len() * 8 >= bits && bytes.len() * 8 < bits + 8);
                assert_eq!(Frame2::read(&bytes, n).unwrap(), f);
            }
        }
    }

    #[test]
    fn decode_reproduces_the_shared_synthesis_loop() {
        // Scalar family: the exp2 decode path must equal `synthesize_gains` on
        // the identical inputs, or the wire would not be behaviour-preserving.
        let k: Vec<f64> = vec![
            0.70, -0.50, 0.40, -0.30, 0.25, -0.20, 0.15, -0.10, 0.08, -0.06, 0.05, -0.04, 0.03,
            -0.02, 0.02, -0.01,
        ];
        let k_q = vp::quantise_k(&k, 6);
        let model = vp::FrameModel {
            estimator: 0,
            order: 16,
            width: 6,
            lag: 0,
            ltpg_q: 0,
        };
        let frame: Vec<f64> = (0..320)
            .map(|i| 400.0 * (0.13 * i as f64).sin() + 90.0 * (0.71 * i as f64).cos())
            .collect();
        let state = VoiceState::new();
        let s = vp::close_loop_gains(&state, &frame, &model, &k_q, vp::RESIDUAL_SUB_LEN, &[16]);
        let payload = residual::encode_best(&s.symbols, &[]);
        let f = Frame2 {
            spectral: Spectral::Vq([9, 8, 7, 6]),
            excitation: Excitation::Scalar {
                gain: 16,
                payload: payload.clone(),
            },
        };
        let (bytes, bits) = f.write();
        assert_eq!(bits, f.bits());
        let back = Frame2::read(&bytes, 320).unwrap();
        let mut a = state.clone();
        let got = back.decode_into(&mut a, 320).unwrap();
        // The same loop run directly on the same inputs and symbols.
        let mut b = state.clone();
        let (codes, width) = f.synthesis_codes();
        let want: Vec<i32> = vp::synthesize_gains(
            &mut b,
            &codes,
            width,
            vp::RESIDUAL_SUB_LEN,
            &[16],
            0,
            0,
            &residual::decode(&payload, 320).unwrap(),
        )
        .iter()
        .map(|&x| x.round() as i32)
        .collect();
        assert_eq!(got, want);
    }

    /// The CELP wire is *differential* (7C.2-D): it sends one frame lag anchor
    /// and a quarter of the subframe parameters as 3–4-bit contour deltas. Those
    /// deltas clamp and the pulse count is shared per frame, so the wire is a
    /// canonicalising map, not the identity. The properties that must actually
    /// hold are therefore:
    ///
    /// 1. **Idempotence** — reading a written frame yields a frame that writes
    ///    to the identical bytes and reads back to itself. A decoder can never
    ///    be handed a frame that re-encodes differently, so a lost packet cannot
    ///    desynchronise the interpretation of a later one.
    /// 2. **Decode fidelity** — decoding the read-back frame equals rendering the
    ///    read-back frame's *own* reconstructed shot through the shared
    ///    synthesis loop. The wire description is behaviour-preserving even
    ///    though the pre-wire search parameters are not.
    #[test]
    fn celp_wire_is_idempotent_and_decodes_its_own_shot() {
        let k = vec![
            0.70, -0.50, 0.40, -0.30, 0.25, -0.20, 0.15, -0.10, 0.08, -0.06, 0.05, -0.04, 0.03,
        ];
        let k_q = vp::quantise_k(&k, 6);
        let model = vp::FrameModel {
            estimator: 0,
            order: 13,
            width: 6,
            lag: 73,
            ltpg_q: 20,
        };
        let mut frame = Vec::with_capacity(320);
        let mut s = 0x2468_ace0_1357_9bdfu64;
        for i in 0..320 {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let noise = ((s >> 40) as i64 % 4001 - 2000) as f64;
            frame.push(noise + if i % 73 < 2 { 800.0 } else { 0.0 });
        }
        let state = VoiceState::new();
        let (params, _d, _shot) = celp::analyse(&state, &frame, &model, &k_q, 4);
        let f = Frame2 {
            spectral: Spectral::Vq([1, 2, 3, 4]),
            excitation: Excitation::Celp(params),
        };
        let (bytes, bits) = f.write();
        assert_eq!(bits, f.bits());
        let back = Frame2::read(&bytes, 320).unwrap();

        // 1. Idempotence: the read-back frame is a wire fixed point.
        let (bytes2, bits2) = back.write();
        assert_eq!(bits2, back.bits());
        assert_eq!(bytes2, bytes, "write must be stable on a canonical frame");
        assert_eq!(Frame2::read(&bytes2, 320).unwrap(), back);

        // 2. Decode fidelity against the read-back frame's own shot.
        let mut a = state.clone();
        let got = back.decode_into(&mut a, 320).unwrap();
        let (codes, width) = back.synthesis_codes();
        let Excitation::Celp(back_params) = &back.excitation else {
            unreachable!("the frame was written as CELP")
        };
        let shot = celp::reconstruct(back_params, 320).unwrap();
        let mut b = state.clone();
        let want: Vec<i32> = shot
            .render(&mut b, &codes, width)
            .iter()
            .map(|&x| x.round() as i32)
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn hard_rate_encoder_never_overshoots_at_any_declared_rate() {
        // A speech-like synthetic frame: a periodic glottal pulse train through a
        // formant-ish resonator plus low-level noise, near 16-bit speech level.
        let mut source = Vec::with_capacity(320 * 40);
        let mut s = 0x1234_5678_9abc_def0u64;
        for i in 0..320 * 40 {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let noise = ((s >> 40) as i64 % 2001 - 1000) as f64;
            let voiced = if i % 78 < 3 { 6000.0 } else { 0.0 };
            let x = voiced + 0.35 * noise + 900.0 * ((i as f64) * 0.09).sin();
            source.push(x.round().clamp(-32768.0, 32767.0) as i32);
        }

        for (bps, bits) in DECLARED_RATES {
            // The encoder reads the decoder's state, so the two cannot drift.
            let mut state = VoiceState::new();
            let mut sig = 0.0;
            let mut err = 0.0;
            let mut frames = 0usize;
            for chunk in source.chunks(320) {
                let enc_state = state.clone();
                let (bytes, nbits) = Exp2Codec::encode_frame(&enc_state, chunk, bits);
                assert!(
                    nbits <= bits,
                    "{bps} bps: {nbits} information bits exceed the {bits}-bit allowance"
                );
                assert!(
                    bytes.len() * 8 <= bits,
                    "{bps} bps: artifact of {} bytes ({}) exceeds the {bits}-bit allowance",
                    bytes.len(),
                    bytes.len() * 8
                );
                let out = Exp2Codec::decode_frame(&mut state, &bytes, 320).unwrap();
                assert_eq!(out.len(), 320);
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
            assert!(
                frames > 0 && err > 0.0,
                "the profile is lossy; exactness is not a claim"
            );
            let snr = 10.0 * (sig / err).log10();
            println!("{bps} bps allowance {bits}: hard-rate SNR {snr:.2} dB over {frames} frames");
        }
    }

    fn sample_ac3(n: usize, per: usize) -> fcelp::Params {
        let nsub = celp::subframes(n);
        fcelp::Params {
            per_track: per,
            subframes: (0..nsub)
                .map(|s| {
                    let mut pulses: Vec<(u8, bool)> = Vec::new();
                    for t in 0..fcelp::TRACKS {
                        for k in 0..per {
                            let pos = t + (k * 3 + s) * fcelp::TRACKS;
                            pulses.push((pos as u8, (s + t + k) % 2 == 0));
                        }
                    }
                    pulses.sort_unstable_by_key(|&(p, _)| p);
                    fcelp::Subframe {
                        lag_q: 160 + s as i32 * 3,
                        pitch_gain: 8 + s as i32 * 2,
                        gain: 16 + s as i32 * 3,
                        pulses,
                    }
                })
                .collect(),
        }
    }

    #[test]
    fn ac3_frames_round_trip_and_are_wire_idempotent() {
        for n in [160usize, 320] {
            for per in 1..=fcelp::MAX_PER_TRACK {
                let f = Frame2 {
                    spectral: Spectral::Vq([1, 2, 3, 4]),
                    excitation: Excitation::Ac3(sample_ac3(n, per)),
                };
                let (bytes, bits) = f.write();
                assert_eq!(bits, f.bits(), "accounting must be exact");
                assert!(bytes.len() * 8 >= bits && bytes.len() * 8 < bits + 8);
                let back = Frame2::read(&bytes, n).unwrap();
                assert_eq!(back, f, "the canonical frame must survive the wire");
                // Wire idempotence, as for the control core.
                let (bytes2, bits2) = back.write();
                assert_eq!(bits2, back.bits());
                assert_eq!(bytes2, bytes);
                assert_eq!(Frame2::read(&bytes2, n).unwrap(), back);
            }
        }
    }

    #[test]
    fn the_two_acelp_cores_are_distinct_wire_families() {
        // The control core and the 7C.2-E core must be distinguishable on the
        // wire, or a decoder could not tell which synthesis loop to run.
        let control = Frame2 {
            spectral: Spectral::Vq([0, 0, 0, 0]),
            excitation: Excitation::Celp(sample_celp(320, 4)),
        };
        let track = Frame2 {
            spectral: Spectral::Vq([0, 0, 0, 0]),
            excitation: Excitation::Ac3(sample_ac3(320, 1)),
        };
        assert_eq!(family_label(&control), "celp");
        assert_eq!(family_label(&track), "acelp");
        assert_eq!(pulse_count(&control), 16);
        assert_eq!(pulse_count(&track), 16);
        let (bc, _) = control.write();
        let (bt, _) = track.write();
        assert_ne!(bc, bt);
        assert_eq!(Frame2::read(&bc, 320).unwrap(), control);
        assert_eq!(Frame2::read(&bt, 320).unwrap(), track);
    }

    #[test]
    fn nested_spectral_tier_round_trips_and_is_exactly_smaller() {
        // The MSVQ is embedded, so stage 0 alone is a legal coarser spectrum.
        let full = [0x5Au8, 0x33, 0xC7, 0x01];
        let (s0, s1) = vq::stage0(&full);
        let nested = vq::pack_stage0(s0, s1);
        let f = Frame2 {
            spectral: Spectral::Vq0(nested),
            excitation: Excitation::Noise { gain: 20, seed: 1 },
        };
        let (bytes, bits) = f.write();
        assert_eq!(bits, f.bits());
        // tier(2) + two 8-bit stage-0 sub-indices + family(2) + sub-mode(1)
        // + gain(6) + seed(3).
        assert_eq!(bits, 2 + 16 + 2 + 1 + 6 + 3);
        // 16 bits cheaper than the full index on the same frame.
        let full_frame = Frame2 {
            spectral: Spectral::Vq(full),
            excitation: Excitation::Noise { gain: 20, seed: 1 },
        };
        assert_eq!(full_frame.bits() - f.bits(), 16);
        let back = Frame2::read(&bytes, 320).unwrap();
        assert_eq!(back, f, "the nested tier must survive the wire");
        // And the two tiers must address the same codebook: the nested decode is
        // the full decode of the same index with stage 1 zeroed.
        assert_eq!(back.synthesis_codes(), f.synthesis_codes());
        let (b2, n2) = back.write();
        assert_eq!((b2, n2), (bytes, bits));
    }

    #[test]
    fn envelope_penalty_rejects_silence_that_raw_mse_accepts() {
        // The measured 7C.2-F finding, pinned so it cannot silently regress: for
        // an uncorrelated reconstruction, MSE is *minimised by silence*, while the
        // envelope term prefers a level-matched frame.
        let target: Vec<f64> = (0..320)
            .map(|i| 1000.0 * (0.2 * i as f64).sin() + 500.0 * (0.9 * i as f64).cos())
            .collect();
        let silent = vec![0.0f64; 320];
        // Uncorrelated, but with the right short-time level.
        let mut s = 0x1234_5678u64;
        let mut noisy = Vec::with_capacity(320);
        for lo in (0..320).step_by(ENV_SUB) {
            let hi = (lo + ENV_SUB).min(320);
            let e = (target[lo..hi].iter().map(|x| x * x).sum::<f64>() / (hi - lo) as f64).sqrt();
            for _ in lo..hi {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                noisy.push(e * (((s >> 40) as i64 % 2001 - 1000) as f64 / 1000.0));
            }
        }
        // Raw MSE prefers silence...
        assert!(mse(&target, &silent) < mse(&target, &noisy));
        // ...while the envelope penalty prefers the level-matched frame.
        assert!(envelope_penalty(&target, &silent) > envelope_penalty(&target, &noisy));
        // With a positive weight the proxy would also prefer it; the shipped
        // weight is fitted separately (see `Options::env_weight`), so assert the
        // mechanism at the recorded 7C.2-F weight rather than at the constant.
        let w = ENV_WEIGHT_HISTORY;
        let d_silent = mse(&target, &silent) + w * envelope_penalty(&target, &silent);
        let d_noisy = mse(&target, &noisy) + w * envelope_penalty(&target, &noisy);
        assert!(d_noisy < d_silent);
    }

    #[test]
    fn tcx_frames_round_trip_exactly_and_are_wire_idempotent() {
        let f = Frame2 {
            spectral: Spectral::Vq([1, 2, 3, 4]),
            excitation: Excitation::Tcx {
                k: 3,
                gains: vec![30, 28, 31, 29],
                indices: vec![1, 2, 3, 4],
            },
        };
        let (bytes, bits) = f.write();
        assert_eq!(bits, f.bits(), "accounting must be exact");
        assert!(bytes.len() * 8 >= bits && bytes.len() * 8 < bits + 8);
        let back = Frame2::read(&bytes, 320).unwrap();
        assert_eq!(back, f, "the canonical escape frame must survive the wire");
        let (bytes2, bits2) = back.write();
        assert_eq!(bytes2, bytes);
        assert_eq!(bits2, bits);
        assert_eq!(Frame2::read(&bytes2, 320).unwrap(), back);
        // A partial frame that is not a whole number of blocks is not a legal
        // escape frame, and must be rejected rather than guessed at.
        assert!(Frame2::read(&bytes, 300).is_err());
        // The escape mode must actually emit signal, not a zero block.
        let mut st = VoiceState::new();
        let out = back.decode_into(&mut st, 320).unwrap();
        assert!(out.iter().any(|&v| v != 0));
    }

    /// The escape core's *mechanism* must be faithful, or "it is never selected"
    /// would be a bug report rather than a measurement. This drives the exact
    /// encode path the selector uses and checks that the decoded block recovers
    /// the shape it was given.
    #[test]
    fn the_escape_core_reconstructs_its_own_quantisation() {
        let len = 320usize;
        let mut resid = vec![0.0f64; len];
        for (i, v) in resid[..pvq::BLOCK].iter_mut().enumerate() {
            let x = i as f64 + 0.5;
            let n = pvq::BLOCK as f64;
            *v = 500.0 * (std::f64::consts::PI * x * 2.5 / n).sin()
                + 300.0 * (std::f64::consts::PI * x * 7.5 / n).cos();
        }
        let coeffs = pvq::forward_blocks(&resid);
        let f = tcx_candidate(&coeffs, 4, &Spectral::Vq([0; 4]), 10_000)
            .expect("the escape candidate must be constructible");
        let (bytes, _) = f.write();
        let back = Frame2::read(&bytes, len).unwrap();
        let Excitation::Tcx { k, gains, indices } = &back.excitation else {
            panic!("the frame was written as an escape frame")
        };
        let y = pvq::unrank(u128::from(indices[0]), pvq::BLOCK, *k);
        let norm = y.iter().map(|v| f64::from(v * v)).sum::<f64>().sqrt();
        assert!(norm > 0.0, "the shape must not be empty");
        let scale = celp::gain_of(gains[0]) / norm;
        let c0 = &coeffs[..pvq::BLOCK];
        let dot: f64 = c0
            .iter()
            .zip(&y)
            .map(|(a, b)| a * scale * f64::from(*b))
            .sum();
        let na: f64 = c0.iter().map(|v| v * v).sum::<f64>().sqrt();
        let nb: f64 = y
            .iter()
            .map(|v| scale * f64::from(*v))
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(
            dot / (na * nb) > 0.5,
            "the escape quantisation lost the shape: correlation {}",
            dot / (na * nb)
        );
    }

    #[test]
    fn every_declared_rate_admits_a_legal_frame() {
        // 20 ms frames: bits per frame for each declared operating rate.
        let rates: [(u32, usize); 6] = [
            (3_200, 64),
            (6_000, 120),
            (8_000, 160),
            (9_200, 184),
            (12_000, 240),
            (16_000, 320),
        ];
        let minimal = Frame2::minimal();
        for (bps, bits) in rates {
            assert!(
                minimal.bits() <= bits,
                "{bps} bps ({bits} bits/frame) admits no legal frame: minimal costs {} bits",
                minimal.bits()
            );
        }
        // And the minimal frame must actually serialise, parse and decode.
        let (bytes, written) = minimal.write();
        assert_eq!(written, minimal.bits());
        assert!(bytes.len() * 8 <= 64, "minimal frame must fit 3.2 kbps");
        let back = Frame2::read(&bytes, 320).unwrap();
        assert_eq!(back, minimal);
        let mut state = VoiceState::new();
        let out = back.decode_into(&mut state, 320).unwrap();
        assert_eq!(out.len(), 320);
        // Deterministic and packet-local: the same frame decodes identically.
        let mut state2 = VoiceState::new();
        assert_eq!(back.decode_into(&mut state2, 320).unwrap(), out);
    }

    #[test]
    fn noise_core_is_seeded_and_deterministic() {
        let a = Frame2 {
            spectral: Spectral::Vq([5, 4, 3, 2]),
            excitation: Excitation::Noise { gain: 20, seed: 3 },
        };
        let b = Frame2 {
            spectral: Spectral::Vq([5, 4, 3, 2]),
            excitation: Excitation::Noise { gain: 20, seed: 5 },
        };
        let (ba, _) = a.write();
        let (bb, _) = b.write();
        assert_eq!(Frame2::read(&ba, 320).unwrap(), a);
        assert_eq!(Frame2::read(&bb, 320).unwrap(), b);
        let mut sa = VoiceState::new();
        let mut sb = VoiceState::new();
        let oa = a.decode_into(&mut sa, 320).unwrap();
        let ob = b.decode_into(&mut sb, 320).unwrap();
        // Different seeds are different realisations, so the encoder has a choice.
        assert_ne!(oa, ob);
    }

    #[test]
    fn scalar_tier_round_trips_and_is_ordered() {
        let f = Frame2 {
            spectral: Spectral::Scalar {
                width: 6,
                codes: vec![3, -7, 0, 12, -1, 5, -5, 2, 0, 0, 1, -1, 4, -2, 6, -3],
            },
            excitation: Excitation::Scalar {
                gain: 17,
                payload: vec![0xab, 0xcd, 0x00, 0x7f],
            },
        };
        let (bytes, bits) = f.write();
        assert_eq!(bits, f.bits());
        assert_eq!(Frame2::read(&bytes, 320).unwrap(), f);
    }

    /// The finding this module exists to make concrete: `exp2` carries the same
    /// 320-sample pitched VQ+CELP frame as `exp1` but does not re-serialise the
    /// generic pitch/gain fields or a residual length, and does not pad to bytes.
    #[test]
    fn exp2_removes_the_exp1_frame_floor() {
        // exp1: flags+base gain 16 + mode 8 + gain delta 8 + lag/LTP 24
        //       + VQ 32 + length 16 + codec id 8 + CELP 4×23.
        let exp1_floor_bits = 16 + 8 + 8 + 24 + 32 + 16 + 8 + 4 * 23;
        assert_eq!(exp1_floor_bits, 204);

        let f = Frame2 {
            spectral: Spectral::Vq([0, 0, 0, 0]),
            excitation: Excitation::Celp(sample_celp(320, 0)),
        };
        let (bytes, bits) = f.write();
        // spectral tier 2 + index 32; family 2 + anchor 9 + contour 3×4
        // + pitch gain 5 + deltas 3×3 + gain 6 + deltas 3×4 + count 3.
        assert_eq!(bits, 2 + 32 + 2 + 9 + 12 + 5 + 9 + 6 + 12 + 3);
        assert_eq!(bits, 92);
        assert_eq!(bytes.len(), 12);
        assert!(bits < exp1_floor_bits);

        // With four pulses the same relation holds: exp1 pays 8 bits/pulse with
        // byte padding; exp2 pays the combinatorial rank and no padding.
        let f4 = Frame2 {
            spectral: Spectral::Vq([0, 0, 0, 0]),
            excitation: Excitation::Celp(sample_celp(320, 4)),
        };
        let (bytes4, bits4) = f4.write();
        let exp1_4pulse = 16 + 8 + 8 + 24 + 32 + 16 + 8 + 4 * (23 + 4 * 8);
        assert_eq!(bits4, 92 + 4 * (21 + 4));
        assert_eq!(bits4, 192);
        assert!(bits4 < exp1_4pulse);
        assert!(bytes4.len() * 8 - bits4 < 8);
    }
}
