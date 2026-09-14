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
use crate::voice::predict as vp;
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
    /// Scalar reflection codes at one ladder width; order is the code count.
    Scalar { width: u8, codes: Vec<i32> },
}

impl Spectral {
    /// Exact information bits for this description.
    pub fn bits(&self) -> usize {
        match self {
            Spectral::Vq(_) => usize::from(TIER_BITS) + vq::VQ_INDEX_BYTES * 8,
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
}

impl Excitation {
    /// Exact information bits for this description.
    pub fn bits(&self) -> usize {
        match self {
            Excitation::Scalar { payload, .. } => {
                usize::from(FAMILY_BITS) + 8 + 12 + payload.len() * 8
            }
            Excitation::Celp(p) => {
                usize::from(FAMILY_BITS)
                    + p.subframes
                        .iter()
                        .map(|s| {
                            usize::from(celp::LAG_BITS)
                                + usize::from(celp::PITCH_GAIN_BITS)
                                + usize::from(celp::GAIN_BITS)
                                + usize::from(celp::COUNT_BITS)
                                + usize::from(celp::position_bits(s.pulses.len()))
                                + s.pulses.len() * usize::from(celp::SIGN_BITS)
                        })
                        .sum::<usize>()
            }
            Excitation::Noise { .. } => usize::from(FAMILY_BITS) + 6 + 3,
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
                for sub in &p.subframes {
                    w.bits(
                        (sub.lag - vp::MIN_LAG as i32).clamp(0, (1 << celp::LAG_BITS) - 1) as u32,
                        celp::LAG_BITS,
                    );
                    w.bits(
                        sub.pitch_gain.clamp(0, celp::PITCH_GAIN_LEVELS - 1) as u32,
                        celp::PITCH_GAIN_BITS,
                    );
                    w.bits(
                        sub.gain.clamp(0, celp::GAIN_LEVELS - 1) as u32,
                        celp::GAIN_BITS,
                    );
                    let count = sub.pulses.len().min((1usize << celp::COUNT_BITS) - 1);
                    w.bits(count as u32, celp::COUNT_BITS);
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
                w.bits((*gain).clamp(0, celp::GAIN_LEVELS - 1) as u32, 6);
                w.bits(u32::from(*seed) & 0x7, 3);
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
                let mut p = celp::Params::default();
                for _ in 0..nsub {
                    let lag = vp::MIN_LAG as i32 + r.bits(celp::LAG_BITS)? as i32;
                    let pitch_gain = r.bits(celp::PITCH_GAIN_BITS)? as i32;
                    let gain = r.bits(celp::GAIN_BITS)? as i32;
                    let count = r.bits(celp::COUNT_BITS)? as usize;
                    if count > celp::MAX_PULSES {
                        return Err(Error::malformed("voice.exp2 pulse count above the bound"));
                    }
                    let pb = celp::position_bits(count);
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
                        lag,
                        pitch_gain,
                        gain,
                        pulses,
                    });
                }
                Ok(Excitation::Celp(p))
            }
            2 => {
                let gain = r.bits(6)? as i32;
                let seed = r.bits(3)? as u8;
                Ok(Excitation::Noise { gain, seed })
            }
            _ => Err(Error::new(
                Kind::Unsupported,
                "voice.exp2 excitation family reserved",
            )),
        }
    }
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
            Spectral::Vq(idx) => (vq::decode_index(idx), vq::VQ_INTERNAL_WIDTH),
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

/// Sum of squared error between a frame and its reconstruction.
fn mse(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
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
        let f: Vec<f64> = frame.iter().map(|&v| f64::from(v)).collect();
        let mut best: Option<(f64, usize, Frame2)> = None;
        let cands = vp::analyse(state, frame, 0);
        let nsub = celp::subframes(frame.len()).max(1);
        let rms = (f.iter().map(|v| v * v).sum::<f64>() / f.len().max(1) as f64).sqrt();
        let noise_codes = noise_gain_codes(rms);

        for c in cands.iter().take(3) {
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
            }
            opts.push((
                c.k_q.clone(),
                c.model.width,
                Spectral::Scalar {
                    width: c.model.width,
                    codes: c.k_q.clone(),
                },
            ));

            for (k_q, width, spectral) in opts {
                let model = vp::FrameModel { width, ..c.model };
                let mut consider = |d: f64, fr: Frame2| {
                    let bits = fr.bits();
                    if bits > frame_bits {
                        return;
                    }
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

                // ACELP, within the remaining allowance.
                let avail = frame_bits.saturating_sub(spectral.bits()) / nsub;
                let maxp = celp::max_pulses_for_bits(avail);
                if maxp > 0 {
                    let (params, _d, shot) = celp::analyse(state, &f, &model, &k_q, maxp);
                    let fr = Frame2 {
                        spectral: spectral.clone(),
                        excitation: Excitation::Celp(params),
                    };
                    if fr.bits() <= frame_bits {
                        let mut s = state.clone();
                        let out = shot.render(&mut s, &k_q, width);
                        consider(mse(&f, &out), fr);
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
                    let fr = Frame2 {
                        spectral: spectral.clone(),
                        excitation: Excitation::Scalar { gain, payload },
                    };
                    if fr.bits() <= frame_bits {
                        consider(mse(&f, &s.samples), fr);
                    }
                }

                // The noise fallback core: level and seed are both chosen.
                for code in noise_codes {
                    for seed in 0..8u8 {
                        let fr = Frame2 {
                            spectral: spectral.clone(),
                            excitation: Excitation::Noise { gain: code, seed },
                        };
                        if fr.bits() > frame_bits {
                            continue;
                        }
                        let mut s = state.clone();
                        if let Ok(out) = fr.decode_into(&mut s, frame.len()) {
                            let out: Vec<f64> = out.iter().map(|&v| f64::from(v)).collect();
                            consider(mse(&f, &out), fr);
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
                        lag: vp::MIN_LAG as i32 + (s as i32 * 17) % 200,
                        pitch_gain: (s as i32 * 5) % celp::PITCH_GAIN_LEVELS,
                        gain: (s as i32 * 9) % celp::GAIN_LEVELS,
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

    #[test]
    fn celp_decode_reproduces_the_reconstructed_shot() {
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
        let (params, _d, shot) = celp::analyse(&state, &frame, &model, &k_q, 4);
        let f = Frame2 {
            spectral: Spectral::Vq([1, 2, 3, 4]),
            excitation: Excitation::Celp(params),
        };
        let (bytes, bits) = f.write();
        assert_eq!(bits, f.bits());
        let back = Frame2::read(&bytes, 320).unwrap();
        let mut a = state.clone();
        let got = back.decode_into(&mut a, 320).unwrap();
        let mut b = state.clone();
        let (codes, width) = f.synthesis_codes();
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
        // tier 2 + index 32 + family 2 + 4 × (9+5+6+3)
        assert_eq!(bits, 2 + 32 + 2 + 4 * 23);
        assert_eq!(bits, 128);
        assert_eq!(bytes.len(), 16);
        assert!(bits < exp1_floor_bits);

        // With four pulses the same relation holds: exp1 pays 8 bits/pulse with
        // byte padding; exp2 pays the combinatorial rank and no padding.
        let f4 = Frame2 {
            spectral: Spectral::Vq([0, 0, 0, 0]),
            excitation: Excitation::Celp(sample_celp(320, 4)),
        };
        let (bytes4, bits4) = f4.write();
        let exp1_4pulse = 16 + 8 + 8 + 24 + 32 + 16 + 8 + 4 * (23 + 4 * 8);
        assert_eq!(bits4, 2 + 32 + 2 + 4 * (23 + 21 + 4));
        assert_eq!(bits4, 228);
        assert!(bits4 < exp1_4pulse);
        assert!(bytes4.len() * 8 - bits4 < 8);
    }
}
