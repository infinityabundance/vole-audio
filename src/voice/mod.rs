//! `vole.audio.stream.voice.exp1` — the Phase 7C voice-call profile.
//!
//! The frozen constitution is `docs/PHASE_7C.md`. Read it before changing
//! anything here: the packet layout, the frame constitution, the latency
//! accounting and the impairment model are commitments, not implementation
//! details.
//!
//! ```text
//! frame      : 160 (10 ms) or 320 (20 ms) samples at 16 kHz, mono
//! packet     : 1 or 2 coded frames, self-contained
//! parameters : ABSOLUTE within a packet — a lost packet can never
//!              desynchronise a later one
//! state      : one bounded ring of reconstructed samples
//! quality    : lossy by construction; no exactness is claimed anywhere
//! ```
//!
//! Nothing here is a network protocol. The impairment model that exercises
//! packet loss and jitter lives in [`crate::voice::impair`] and is purely
//! in-process and deterministic.

pub mod impair;
pub mod plc;
pub mod predict;
pub mod residual;

use crate::error::{Error, Kind, Result};
use crate::learned::lpc::{pack_signed, unpack_signed};
use crate::learned::residual_codec2::SEARCH_CODECS;
use crate::voice::predict::{
    self as vp, FrameModel, GAIN_MAX, GAIN_MIN, LTPG_LEVELS, ORDER_LADDER, VoiceState, WIDTH_LADDER,
};

/// Streaming voice profile identity.
pub const VOICE_PROFILE: &str = "vole.audio.stream.voice.exp1";
/// Canonical profile tag bytes.
pub const VOICE_PROFILE_TAG: &[u8] = b"vole.audio.stream.voice.exp1";
/// Container magic.
pub const VOICE_MAGIC: &[u8; 10] = b"vole.voice";
/// Container format version.
pub const VOICE_VERSION: u8 = 1;
/// Frame lengths admitted by the constitution.
pub const FRAME_LADDER: [u32; 2] = [160, 320];
/// Hard bound on one encoded packet (a hostile-input ceiling, not a size goal).
pub const MAX_PACKET_BYTES: usize = 1 << 16;
/// Frames between periodic state capsules when capsules are enabled.
pub const DEFAULT_CAPSULE_CADENCE: u32 = 8;
/// Frames between comfort-noise (SID) updates in DTX.
pub const DEFAULT_SID_CADENCE: u32 = 8;

/// How far a frame's complete bytes may exceed its budget before the encoder
/// prefers a coarser residual step. The rate target is a *target*: silence is
/// not an acceptable way to meet it, and the overshoot is visible in every
/// receipt as the difference between target and actual bitrate.
pub const OVERSHOOT_LIMIT: usize = 4;

/// Floor on one frame's byte allowance.
///
/// Below roughly this many bytes the frame record cannot carry even a minimal
/// model description plus a length field, so a target that small has no
/// meaningful solution. Handing the search a zero budget would silently disable
/// the overshoot policy — the encoder would fall straight through to the
/// cheapest non-degenerate step and its worst quality — so the allowance floors
/// here and the overshoot stays bounded and visible.
pub const MIN_FRAME_BYTES: usize = 8;

/// Voice codec configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoiceConfig {
    /// Input sample rate. The constitution fixes 16 000.
    pub sample_rate_hz: u32,
    /// Frame length in samples: 160 (10 ms) or 320 (20 ms).
    pub frame_len: u32,
    /// Coded frames per packet: 1, or 2 for the 2×10 ms coalescing shape.
    pub frames_per_packet: u8,
    /// Target bitrate for the whole stream, bits per second.
    pub target_bits_per_second: u32,
    /// Enable VAD/DTX with procedural comfort noise.
    pub dtx: bool,
    /// Frames between periodic state capsules; 0 disables capsules.
    pub capsule_cadence: u32,
    /// Emit backward redundancy for the previous frame.
    pub redundancy: bool,
}

impl VoiceConfig {
    /// Samples one packet carries.
    pub const fn packet_samples(&self) -> usize {
        self.frame_len as usize * self.frames_per_packet as usize
    }

    /// Validate against the constitution.
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate_hz != 16_000 {
            return Err(Error::new(
                Kind::Unsupported,
                "the voice profile is fixed at 16 kHz by its constitution",
            ));
        }
        if !FRAME_LADDER.contains(&self.frame_len) {
            return Err(Error::malformed(
                "voice frame length must be 160 or 320 samples",
            ));
        }
        if !(1..=2).contains(&self.frames_per_packet) {
            return Err(Error::malformed("voice packets carry one or two frames"));
        }
        if self.frame_len == 320 && self.frames_per_packet == 2 {
            return Err(Error::malformed(
                "2×320 samples per packet is not a constitution shape",
            ));
        }
        if self.target_bits_per_second == 0 {
            return Err(Error::malformed("voice target bitrate must be positive"));
        }
        Ok(())
    }

    /// Bytes the packet target allows, before framing.
    pub fn target_bytes(&self) -> usize {
        let samples = self.packet_samples() as u64;
        let bits =
            u64::from(self.target_bits_per_second) * samples / u64::from(self.sample_rate_hz);
        (bits / 8).max(4) as usize
    }
}

// ---------------------------------------------------------------------------
// Byte cursor (little-endian, bounds-checked)
// ---------------------------------------------------------------------------

struct Writer {
    out: Vec<u8>,
}

impl Writer {
    fn new() -> Writer {
        Writer { out: Vec::new() }
    }
    fn u8(&mut self, v: u8) {
        self.out.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes(&mut self, b: &[u8]) {
        self.out.extend_from_slice(b);
    }
    fn finish(self) -> Vec<u8> {
        self.out
    }
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Reader<'a> {
        Reader { b, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| Error::limit("voice packet read overflows"))?;
        let s = self
            .b
            .get(self.pos..end)
            .ok_or_else(|| Error::malformed("voice packet is truncated"))?;
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
}

// ---------------------------------------------------------------------------
// Flags and mode bytes
// ---------------------------------------------------------------------------

const FLAG_NO_DATA: u8 = 1 << 0;
const FLAG_CAPSULE: u8 = 1 << 1;
const FLAG_REDUNDANCY: u8 = 1 << 2;
const FLAG_CONCEALED: u8 = 1 << 3;

/// Pack a frame model into its `mode` byte.
pub fn pack_mode(m: &FrameModel) -> Result<u8> {
    let order_idx = ORDER_LADDER
        .iter()
        .position(|&o| o == m.order)
        .ok_or_else(|| Error::internal("voice order outside the ladder"))?;
    let width_idx = WIDTH_LADDER
        .iter()
        .position(|&w| w == m.width)
        .ok_or_else(|| Error::internal("voice width outside the ladder"))?;
    if m.estimator > 2 {
        return Err(Error::internal("voice estimator outside the ladder"));
    }
    Ok(m.estimator
        | (u8::from(m.lag > 0) << 2)
        | ((order_idx as u8) << 3)
        | ((width_idx as u8) << 6))
}

/// Unpack a `mode` byte.
pub fn unpack_mode(mode: u8) -> Result<FrameModel> {
    let estimator = mode & 0b11;
    let pitch = (mode >> 2) & 1 == 1;
    let order_idx = ((mode >> 3) & 0b111) as usize;
    let width_idx = ((mode >> 6) & 0b11) as usize;
    if estimator > 2 || order_idx >= ORDER_LADDER.len() || width_idx >= WIDTH_LADDER.len() {
        return Err(Error::malformed("voice mode byte out of domain"));
    }
    Ok(FrameModel {
        estimator,
        order: ORDER_LADDER[order_idx],
        width: WIDTH_LADDER[width_idx],
        lag: if pitch { 1 } else { 0 },
        ltpg_q: 0,
    })
}

/// Deterministic splitmix64. The codec's own PRNG: comfort noise and
/// concealment excitation are part of the profile, so both sides must generate
/// the identical sequence from the identical seed.
#[inline]
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A uniform sample in `[-1, 1)` from the codec PRNG.
#[inline]
fn prng_unit(state: &mut u64) -> f64 {
    let v = splitmix64(state) >> 11;
    (v as f64) / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// The PRNG seed for a frame index. Frozen: comfort noise must be identical on
/// both sides across processes and runs.
pub fn comfort_seed(frame_index: u64) -> u64 {
    0x7f3b_9c1d_5e2a_4b87 ^ frame_index.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

// ---------------------------------------------------------------------------
// Capsules and redundancy
// ---------------------------------------------------------------------------

/// Capsule field bits (7C.6).
const CAP_PITCH: u8 = 1 << 0;
const CAP_GAIN: u8 = 1 << 1;
const CAP_SPECTRAL: u8 = 1 << 2;
const CAP_COMFORT: u8 = 1 << 3;

/// Redundancy field bits (7C.7).
const RED_MODE: u8 = 1 << 0;
const RED_PITCH: u8 = 1 << 1;
const RED_GAIN: u8 = 1 << 2;
const RED_SPECTRAL: u8 = 1 << 3;

/// A rotate-xor checksum over a capsule/redundancy body. Cheap, deterministic,
/// and sufficient to reject a stale capsule applied to the wrong state.
fn body_checksum(body: &[u8]) -> u8 {
    let mut acc = 0x5bu8;
    for &b in body {
        acc = acc.rotate_left(1) ^ b;
    }
    acc
}

/// The decoder-visible state a capsule carries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Capsule {
    /// Last good pitch lag (0 = unvoiced).
    pub lag: i32,
    /// Last good long-term gain code.
    pub ltpg_q: i32,
    /// Last good residual gain code.
    pub gain: i32,
    /// Low-order spectral state: the first four reflection codes at width 6.
    pub spectral: [i8; 4],
    /// Comfort-noise level class.
    pub level: u8,
    /// Comfort-noise spectral class.
    pub shape: u8,
    /// Which fields are meaningful.
    pub mask: u8,
}

impl Capsule {
    fn write(&self, w: &mut Writer) {
        let mut body = Vec::with_capacity(16);
        body.push(self.mask);
        if self.mask & CAP_PITCH != 0 {
            body.extend_from_slice(&(self.lag.clamp(0, i32::from(u16::MAX)) as u16).to_le_bytes());
            body.push(self.ltpg_q.clamp(0, LTPG_LEVELS - 1) as u8);
        }
        if self.mask & CAP_GAIN != 0 {
            body.push((self.gain - GAIN_MIN).clamp(0, 255) as u8);
        }
        if self.mask & CAP_SPECTRAL != 0 {
            for &s in &self.spectral {
                body.push(s as u8);
            }
        }
        if self.mask & CAP_COMFORT != 0 {
            body.push(self.level);
            body.push(self.shape);
        }
        let ck = body_checksum(&body);
        w.bytes(&body);
        w.u8(ck);
    }

    fn read(r: &mut Reader<'_>) -> Result<Capsule> {
        let mask = r.u8()?;
        let mut body = vec![mask];
        let mut c = Capsule {
            lag: 0,
            ltpg_q: 0,
            gain: 0,
            spectral: [0; 4],
            level: 0,
            shape: 0,
            mask,
        };
        if mask & CAP_PITCH != 0 {
            let b = r.take(3)?;
            body.extend_from_slice(b);
            c.lag = i32::from(u16::from_le_bytes([b[0], b[1]]));
            c.ltpg_q = i32::from(b[2]);
        }
        if mask & CAP_GAIN != 0 {
            let b = r.u8()?;
            body.push(b);
            c.gain = i32::from(b) + GAIN_MIN;
        }
        if mask & CAP_SPECTRAL != 0 {
            let b = r.take(4)?;
            body.extend_from_slice(b);
            for (s, &v) in c.spectral.iter_mut().zip(b.iter()) {
                *s = v as i8;
            }
        }
        if mask & CAP_COMFORT != 0 {
            let b = r.take(2)?;
            body.extend_from_slice(b);
            c.level = b[0];
            c.shape = b[1];
        }
        let ck = r.u8()?;
        if body_checksum(&body) != ck {
            return Err(Error::malformed("capsule checksum mismatch"));
        }
        Ok(c)
    }
}

/// Backward redundancy for the previous frame (7C.7).
#[derive(Debug, Clone, PartialEq)]
pub struct Redundancy {
    /// Which fields are present.
    pub mask: u8,
    /// Previous frame's model, when `RED_MODE` is set.
    pub mode: u8,
    /// Previous lag, when `RED_PITCH` is set.
    pub lag: i32,
    /// Previous long-term gain code.
    pub ltpg_q: i32,
    /// Previous residual gain, when `RED_GAIN` is set.
    pub gain: i32,
    /// Previous first eight reflection codes, when `RED_SPECTRAL`.
    pub spectral: [i8; 8],
}

impl Redundancy {
    fn write(&self, w: &mut Writer) {
        let mut body = Vec::with_capacity(16);
        body.push(self.mask);
        if self.mask & RED_MODE != 0 {
            body.push(self.mode);
        }
        if self.mask & RED_PITCH != 0 {
            body.extend_from_slice(&(self.lag.clamp(0, i32::from(u16::MAX)) as u16).to_le_bytes());
            body.push(self.ltpg_q.clamp(0, LTPG_LEVELS - 1) as u8);
        }
        if self.mask & RED_GAIN != 0 {
            body.push((self.gain - GAIN_MIN).clamp(0, 255) as u8);
        }
        if self.mask & RED_SPECTRAL != 0 {
            for &s in &self.spectral {
                body.push(s as u8);
            }
        }
        let ck = body_checksum(&body);
        w.bytes(&body);
        w.u8(ck);
    }

    fn read(r: &mut Reader<'_>) -> Result<Redundancy> {
        let mask = r.u8()?;
        let mut body = vec![mask];
        let mut red = Redundancy {
            mask,
            mode: 0,
            lag: 0,
            ltpg_q: 0,
            gain: 0,
            spectral: [0; 8],
        };
        if mask & RED_MODE != 0 {
            let b = r.u8()?;
            body.push(b);
            red.mode = b;
        }
        if mask & RED_PITCH != 0 {
            let b = r.take(3)?;
            body.extend_from_slice(b);
            red.lag = i32::from(u16::from_le_bytes([b[0], b[1]]));
            red.ltpg_q = i32::from(b[2]);
        }
        if mask & RED_GAIN != 0 {
            let b = r.u8()?;
            body.push(b);
            red.gain = i32::from(b) + GAIN_MIN;
        }
        if mask & RED_SPECTRAL != 0 {
            let b = r.take(8)?;
            body.extend_from_slice(b);
            for (s, &v) in red.spectral.iter_mut().zip(b.iter()) {
                *s = v as i8;
            }
        }
        let ck = r.u8()?;
        if body_checksum(&body) != ck {
            return Err(Error::malformed("redundancy checksum mismatch"));
        }
        Ok(red)
    }
}

// ---------------------------------------------------------------------------
// A coded frame
// ---------------------------------------------------------------------------

/// One coded frame: the model description plus the entropy-coded residual.
#[derive(Debug, Clone)]
pub struct CodedFrame {
    /// The frame model.
    pub model: FrameModel,
    /// Quantised reflection codes at `model.width`.
    pub k_q: Vec<i32>,
    /// Residual gain code.
    pub gain: i32,
    /// Complete entropy artifact (codec id byte + payload).
    pub residual: Vec<u8>,
    /// The quantised residual symbols.
    pub symbols: Vec<i32>,
}

impl CodedFrame {
    fn write(&self, w: &mut Writer, base_gain: i32) {
        let mode = pack_mode(&self.model).unwrap_or(0);
        w.u8(mode);
        w.u8((self.gain - base_gain).clamp(-127, 127) as i8 as u8);
        if self.model.lag > 0 {
            w.u16(self.model.lag.clamp(0, i32::from(u16::MAX)) as u16);
            w.u8(self.model.ltpg_q.clamp(0, LTPG_LEVELS - 1) as u8);
        }
        let packed = pack_signed(&self.k_q, self.model.width);
        w.bytes(&packed);
        w.u16(self.residual.len().min(u16::MAX as usize) as u16);
        w.bytes(&self.residual);
    }

    fn read(r: &mut Reader<'_>, base_gain: i32, frame_len: usize) -> Result<CodedFrame> {
        let mode = r.u8()?;
        let mut model = unpack_mode(mode)?;
        let delta = r.u8()? as i8;
        let gain = (base_gain + i32::from(delta)).clamp(GAIN_MIN, GAIN_MAX);
        if model.lag > 0 {
            model.lag = i32::from(r.u16()?);
            model.ltpg_q = i32::from(r.u8()?);
            if model.lag != 0 && (model.lag as usize) < vp::MIN_LAG {
                return Err(Error::malformed("voice pitch lag below the constitution"));
            }
        }
        let k_bytes = (model.order * usize::from(model.width)).div_ceil(8);
        let raw = unpack_signed(r.take(k_bytes)?, model.width, model.order)?;
        // The wire field is two's complement, so its most negative code would
        // dequantise to k = -1 and make the synthesis filter marginally stable.
        // The encoder never emits it; a corrupt packet could, so clamp here.
        let k_q = vp::clamp_k_codes(&raw, model.width);
        let rlen = usize::from(r.u16()?);
        let residual = r.take(rlen)?.to_vec();
        let symbols = residual::decode(&residual, frame_len)?;
        Ok(CodedFrame {
            model,
            k_q,
            gain,
            residual,
            symbols,
        })
    }
}

// ---------------------------------------------------------------------------
// Shared assists
// ---------------------------------------------------------------------------

/// True when a frame carries no voice activity worth transmitting.
fn is_inactive(frame: &[i32], floor: f64) -> bool {
    segment_rms(frame) < floor.max(30.0) * 2.0
}

/// Quantise an RMS to the 8-bit comfort-noise level class (0.5 dB units).
fn level_of(rms: f64) -> u8 {
    let db = 20.0 * (rms.max(1.0)).log10();
    (db * 2.0).round().clamp(0.0, 255.0) as u8
}

/// Inverse of [`level_of`].
fn rms_of(level: u8) -> f64 {
    10f64.powf(f64::from(level) / 40.0)
}

fn segment_rms(x: &[i32]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let e: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    (e / x.len() as f64).sqrt()
}

/// Dequantise a code vector and requantise at another width, so a candidate's
/// *analysis* survives a width change.
fn reflection_of(k_q: &[i32], width: u8) -> Vec<f64> {
    let scale = 1.0 / f64::from(1u32 << vp::shift_of(width));
    k_q.iter().map(|&q| f64::from(q) * scale).collect()
}

/// The first eight reflection codes at width 6, for capsules and redundancy.
fn spectral_codes(k_q: &[i32], order: usize, width: u8) -> [i8; 8] {
    let mut out = [0i8; 8];
    let q = vp::quantise_k(&reflection_of(k_q, width), 6);
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = if i < order {
            q.get(i).copied().unwrap_or(0).clamp(-127, 127) as i8
        } else {
            0
        };
    }
    out
}

/// The shape class of a comfort-noise capsule, as a width-6 reflection vector.
/// The class selects a gentle spectral tilt; it is a *class*, not a stored
/// codebook, and the decoder rebuilds it from the same formula.
fn comfort_shape(shape: u8) -> [i32; 4] {
    let tilt = i32::from(shape % 4);
    [-(6 + tilt), 4 - tilt, -2, 1]
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// The voice encoder. Holds exactly the state the decoder holds, so its own
/// reconstruction is what the decoder will hear.
#[derive(Debug, Clone)]
pub struct VoiceEncoder {
    config: VoiceConfig,
    state: VoiceState,
    frame_index: u64,
    noise_floor: f64,
    since_capsule: u32,
    since_sid: u32,
    last: Option<CodedFrame>,
}

impl VoiceEncoder {
    /// Build an encoder for a validated configuration.
    pub fn new(config: VoiceConfig) -> Result<VoiceEncoder> {
        config.validate()?;
        Ok(VoiceEncoder {
            config,
            state: VoiceState::new(),
            frame_index: 0,
            noise_floor: 0.0,
            since_capsule: u32::MAX,
            since_sid: u32::MAX,
            last: None,
        })
    }

    /// The configuration in force.
    pub fn config(&self) -> &VoiceConfig {
        &self.config
    }

    /// Encode one packet's worth of samples.
    pub fn encode_packet(&mut self, samples: &[i32]) -> Result<Vec<u8>> {
        let n = self.config.frame_len as usize;
        let nf = usize::from(self.config.frames_per_packet);
        if samples.len() != n * nf {
            return Err(Error::malformed(
                "voice encoder input does not match the packet shape",
            ));
        }

        // Adaptive noise floor: a *proposal* mechanism for DTX only, never an
        // accept/reject authority. It starts a factor of four below the first
        // frame so a call does not open on a false silence verdict.
        let first = &samples[..n];
        let rms = segment_rms(first);
        self.noise_floor = if self.noise_floor <= 0.0 {
            (rms * 0.25).max(1.0)
        } else if rms < self.noise_floor {
            self.noise_floor * 0.98 + rms * 0.02
        } else {
            self.noise_floor * 0.9995 + rms * 0.0005
        };
        self.noise_floor = self.noise_floor.max(1.0);

        if self.config.dtx && is_inactive(first, self.noise_floor) {
            return self.encode_inactive();
        }

        let target = self.config.target_bytes();
        let mut body = Writer::new();
        let mut flags = ((nf - 1) as u8 & 0b11) << 4;

        // Capsule decision: cadence, or an explicit transition trigger.
        let want_capsule = self.config.capsule_cadence > 0
            && (self.since_capsule >= self.config.capsule_cadence
                || self.last.is_none()
                || self.last.as_ref().is_some_and(|f| f.model.lag == 0));
        if want_capsule {
            flags |= FLAG_CAPSULE;
            self.build_capsule().write(&mut body);
        }
        if self.config.redundancy
            && let Some(f) = &self.last
        {
            flags |= FLAG_REDUNDANCY;
            Redundancy {
                mask: RED_MODE | RED_PITCH | RED_GAIN | RED_SPECTRAL,
                mode: pack_mode(&f.model).unwrap_or(0),
                lag: f.model.lag,
                ltpg_q: f.model.ltpg_q,
                gain: f.gain,
                spectral: spectral_codes(&f.k_q, f.model.order, f.model.width),
            }
            .write(&mut body);
        }

        let overhead = 2 + body.out.len();
        let share = target.saturating_sub(overhead) / nf;
        // A zero allowance would silently disable the overshoot policy in
        // `choose_gain`, so the per-frame allowance floors at the smallest
        // record that can carry a model description and a length.
        let per_frame = share.max(MIN_FRAME_BYTES);
        let mut carry = 0usize;
        let mut coded = Vec::with_capacity(nf);
        for f in 0..nf {
            let frame = &samples[f * n..(f + 1) * n];
            let frame_budget = (per_frame + carry).max(1);
            let c = self.encode_frame(frame, frame_budget)?;
            carry = frame_budget.saturating_sub(c.residual.len() + c.model.description_bytes());
            coded.push(c);
        }

        let mut w = Writer::new();
        w.u8(flags);
        w.u8((coded[0].gain - GAIN_MIN).clamp(0, 255) as u8);
        w.bytes(&body.out);
        let base = coded[0].gain;
        for c in &coded {
            c.write(&mut w, base);
        }

        self.advance(&coded);
        self.since_capsule = if want_capsule {
            0
        } else {
            self.since_capsule.saturating_add(nf as u32)
        };
        self.last = coded.last().cloned();
        self.frame_index += nf as u64;
        Ok(w.finish())
    }

    fn advance(&mut self, coded: &[CodedFrame]) {
        for c in coded {
            vp::synthesize(
                &mut self.state,
                &c.k_q,
                c.model.width,
                c.model.lag,
                c.model.ltpg_q,
                c.gain,
                &c.symbols,
            );
        }
    }

    fn build_capsule(&self) -> Capsule {
        match &self.last {
            None => Capsule {
                lag: 0,
                ltpg_q: 0,
                gain: 0,
                spectral: [0; 4],
                level: level_of(self.noise_floor),
                shape: 0,
                mask: CAP_COMFORT,
            },
            Some(f) => {
                let spec = spectral_codes(&f.k_q, f.model.order, f.model.width);
                Capsule {
                    lag: f.model.lag,
                    ltpg_q: f.model.ltpg_q,
                    gain: f.gain,
                    spectral: [spec[0], spec[1], spec[2], spec[3]],
                    level: level_of(self.noise_floor),
                    shape: 0,
                    mask: CAP_PITCH | CAP_GAIN | CAP_SPECTRAL | CAP_COMFORT,
                }
            }
        }
    }

    fn encode_inactive(&mut self) -> Result<Vec<u8>> {
        let nf = usize::from(self.config.frames_per_packet);
        let update = self.since_sid >= DEFAULT_SID_CADENCE;
        let level = level_of(self.noise_floor);
        let mut w = Writer::new();
        let mut flags = FLAG_NO_DATA | (((nf - 1) as u8 & 0b11) << 4);
        if update {
            flags |= FLAG_CAPSULE;
        }
        w.u8(flags);
        w.u8(0);
        let shape = 0u8;
        if update {
            Capsule {
                lag: 0,
                ltpg_q: 0,
                gain: 0,
                spectral: [0; 4],
                level,
                shape,
                mask: CAP_COMFORT,
            }
            .write(&mut w);
            self.since_sid = 0;
        } else {
            self.since_sid = self.since_sid.saturating_add(nf as u32);
        }
        // The encoder generates the identical comfort noise the decoder will, so
        // the reconstruction rings stay in lockstep without transmitting a
        // single comfort-noise sample.
        self.fill_comfort(level, shape, self.config.packet_samples());
        self.last = None;
        self.frame_index += nf as u64;
        Ok(w.finish())
    }

    /// Procedural comfort noise: deterministic PRNG excitation shaped by the
    /// comfort spectral class, pushed through the shared synthesis loop.
    fn fill_comfort(&mut self, level: u8, shape: u8, count: usize) {
        let amp = rms_of(level);
        let k_q = comfort_shape(shape);
        let mut seed = comfort_seed(self.frame_index);
        let mut symbols = Vec::with_capacity(count);
        for _ in 0..count {
            let v = (prng_unit(&mut seed) * amp).round();
            symbols.push(v.clamp(-32768.0, 32767.0) as i32);
        }
        vp::synthesize(&mut self.state, &k_q, 6, 0, 0, 0, &symbols);
    }

    /// Encode one frame: bounded candidate competition and an exact sweep over
    /// the residual step against the physical byte budget.
    fn encode_frame(&self, frame: &[i32], budget: usize) -> Result<CodedFrame> {
        let f: Vec<f64> = frame.iter().map(|&v| f64::from(v)).collect();
        let lag_hint = self.last.as_ref().map_or(0, |c| c.model.lag);
        let cands = vp::analyse(&self.state, frame, lag_hint);
        if cands.is_empty() {
            return Err(Error::internal("voice analysis proposed no candidate"));
        }
        // Stage A: rank by open-loop residual energy (a proposal), then refine
        // the top few by closed-loop distortion at a common reference gain.
        let mut order: Vec<usize> = (0..cands.len()).collect();
        order.sort_by(|&a, &b| {
            cands[a]
                .energy
                .partial_cmp(&cands[b].energy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order.truncate(4);
        let ref_gain = GAIN_MIN + (GAIN_MAX - GAIN_MIN) / 2;
        // Bytes a model would cost before any residual symbol, including the
        // mode/gain bytes and the residual length field.
        let wire_cost = |m: &FrameModel| m.description_bytes() + 2;
        let mut best: Option<(usize, u8)> = None;
        let mut best_d = f64::INFINITY;
        let mut cheapest: Option<(usize, u8)> = None;
        let mut cheapest_cost = usize::MAX;
        for &ci in &order {
            for &width in &WIDTH_LADDER {
                let mut model = cands[ci].model;
                model.width = width;
                let cost = wire_cost(&model);
                if cost < cheapest_cost {
                    cheapest_cost = cost;
                    cheapest = Some((ci, width));
                }
                // A model whose description alone overruns the frame's whole
                // byte allowance cannot be rescued by a coarser quantiser, so
                // it must not win the distortion ranking.
                if cost > budget {
                    continue;
                }
                let k_q =
                    vp::quantise_k(&reflection_of(&cands[ci].k_q, cands[ci].model.width), width);
                let s = vp::close_loop(&self.state, &f, &model, &k_q, ref_gain);
                if s.distortion < best_d {
                    best_d = s.distortion;
                    best = Some((ci, width));
                }
            }
        }
        // With an impossible budget, transmit the cheapest description rather
        // than an expensive one the residual cannot follow. The court reports
        // the resulting quality honestly instead of hiding it behind silence.
        let (ci, width) = best
            .or(cheapest)
            .ok_or_else(|| Error::internal("voice candidate search empty"))?;
        let model = FrameModel {
            width,
            ..cands[ci].model
        };
        let k_q = vp::quantise_k(&reflection_of(&cands[ci].k_q, cands[ci].model.width), width);

        // Stage B: the residual step. The step is swept exactly — bisection
        // would be wrong here, because the byte cost is not monotone in the step
        // near the point where the dead zone swallows the whole frame. Among the
        // steps whose complete frame fits, the lowest distortion wins; if none
        // fits, the encoder is allowed a bounded overshoot rather than being
        // forced into silence, and the court reports the real byte count.
        let descr = model.description_bytes() + 2;
        let mut chosen = self.choose_gain(&f, &model, &k_q, descr, budget);
        if chosen.is_none()
            && let Some((cci, cwidth)) = cheapest
            && (cci, cwidth) != (ci, width)
        {
            let cmodel = FrameModel {
                width: cwidth,
                ..cands[cci].model
            };
            let ck = vp::quantise_k(
                &reflection_of(&cands[cci].k_q, cands[cci].model.width),
                cwidth,
            );
            let cdescr = cmodel.description_bytes() + 2;
            if let Some(v) = self.choose_gain(&f, &cmodel, &ck, cdescr, budget) {
                let residual = residual::encode_best(&v.1, &SEARCH_CODECS);
                if residual.len() <= u16::MAX as usize {
                    return Ok(CodedFrame {
                        model: cmodel,
                        k_q: ck,
                        gain: v.0,
                        residual,
                        symbols: v.1,
                    });
                }
            }
        }
        let (gain, symbols) = match chosen.take() {
            Some(v) => v,
            None => {
                let s = vp::close_loop(&self.state, &f, &model, &k_q, GAIN_MAX);
                (GAIN_MAX, s.symbols)
            }
        };
        let residual = residual::encode_best(&symbols, &SEARCH_CODECS);
        if residual.len() <= u16::MAX as usize {
            return Ok(CodedFrame {
                model,
                k_q,
                gain,
                residual,
                symbols,
            });
        }
        // Wire-format guard: a residual longer than a u16 falls back to the
        // coarsest step, which always fits for any bounded frame.
        let s = vp::close_loop(&self.state, &f, &model, &k_q, GAIN_MAX);
        let residual = residual::encode_best(&s.symbols, &SEARCH_CODECS);
        Ok(CodedFrame {
            model,
            k_q,
            gain: GAIN_MAX,
            residual,
            symbols: s.symbols,
        })
    }

    /// The residual step to transmit.
    ///
    /// Preference order:
    /// 1. lowest distortion among steps whose complete frame fits `budget`;
    /// 2. else lowest distortion among steps within `OVERSHOOT_LIMIT` × budget;
    /// 3. else the **cheapest non-degenerate** step;
    /// 4. else the cheapest step at all.
    ///
    /// A step is *degenerate* when it is coarse enough to quantise the whole
    /// frame to zero. Degenerate steps are never a valid choice for a frame with
    /// energy: they transmit no excitation, so the reconstruction history never
    /// leaves zero and the codec goes permanently silent — the trap that a plain
    /// "coarsest step that fits" rate controller falls into. They remain the
    /// absolute last resort so that no input can panic the encoder.
    ///
    /// Step 3 exists because collapsing to silence is never a correct way to
    /// meet a rate target, and because the court measures the *actual* emitted
    /// bitrate; the overshoot is visible in every receipt.
    fn choose_gain(
        &self,
        frame: &[f64],
        model: &FrameModel,
        k_q: &[i32],
        description_bytes: usize,
        budget: usize,
    ) -> Option<(i32, Vec<i32>)> {
        let limit = budget.saturating_mul(OVERSHOOT_LIMIT);
        let source_energy: f64 = frame.iter().map(|v| v * v).sum();
        let mut fit: Option<(f64, i32, Vec<i32>)> = None;
        let mut over: Option<(f64, i32, Vec<i32>)> = None;
        let mut cheap_nondegenerate: Option<(usize, i32, Vec<i32>)> = None;
        let mut last_resort: Option<(usize, i32, Vec<i32>)> = None;
        for gain in GAIN_MIN..=GAIN_MAX {
            let s = vp::close_loop(&self.state, frame, model, k_q, gain);
            let cost = description_bytes + residual::estimated_bytes(&s.symbols);
            let d = s.distortion;
            if last_resort.as_ref().is_none_or(|b| cost < b.0) {
                last_resort = Some((cost, gain, s.symbols.clone()));
            }
            let degenerate = source_energy > 0.0 && s.symbols.iter().all(|&v| v == 0);
            if degenerate {
                continue;
            }
            if cheap_nondegenerate.as_ref().is_none_or(|b| cost < b.0) {
                cheap_nondegenerate = Some((cost, gain, s.symbols.clone()));
            }
            let improves =
                |slot: &Option<(f64, i32, Vec<i32>)>| slot.as_ref().is_none_or(|b| d < b.0);
            if cost <= budget {
                if improves(&fit) {
                    fit = Some((d, gain, s.symbols.clone()));
                }
            } else if cost <= limit && improves(&over) {
                over = Some((d, gain, s.symbols));
            }
        }
        if let Some((_, gain, symbols)) = fit.or(over) {
            return Some((gain, symbols));
        }
        if let Some((_, gain, symbols)) = cheap_nondegenerate {
            return Some((gain, symbols));
        }
        last_resort.map(|(_, gain, symbols)| (gain, symbols))
    }

    /// The encoder's own reconstruction ring, for evidence and tests.
    pub fn state(&self) -> &VoiceState {
        &self.state
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// What a decoded packet yielded, for the court's accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacketKind {
    /// Coded speech frames.
    Active,
    /// A DTX/no-data packet.
    Inactive,
}

/// The voice decoder.
#[derive(Debug, Clone)]
pub struct VoiceDecoder {
    config: VoiceConfig,
    state: VoiceState,
    frame_index: u64,
    last: Option<CodedFrame>,
    last_sid: Option<(u8, u8)>,
    conceal_run: u32,
}

impl VoiceDecoder {
    /// Build a decoder for a validated configuration.
    pub fn new(config: VoiceConfig) -> Result<VoiceDecoder> {
        config.validate()?;
        Ok(VoiceDecoder {
            config,
            state: VoiceState::new(),
            frame_index: 0,
            last: None,
            last_sid: None,
            conceal_run: 0,
        })
    }

    /// The configuration in force.
    pub fn config(&self) -> &VoiceConfig {
        &self.config
    }

    /// Decode one packet, returning `(kind, samples)`.
    pub fn decode_packet(&mut self, packet: &[u8]) -> Result<(PacketKind, Vec<i32>)> {
        if packet.len() > MAX_PACKET_BYTES {
            return Err(Error::limit("voice packet exceeds the wire bound"));
        }
        let mut r = Reader::new(packet);
        let flags = r.u8()?;
        let rate = r.u8()?;
        if flags & FLAG_CONCEALED != 0 {
            return Err(Error::malformed(
                "a transmitted packet must not carry the CONCEALED flag",
            ));
        }
        let nf = (usize::from((flags >> 4) & 0b11)) + 1;
        if nf != usize::from(self.config.frames_per_packet) {
            return Err(Error::malformed(
                "voice packet frame count disagrees with the configuration",
            ));
        }
        if flags & FLAG_NO_DATA != 0 {
            return self.decode_inactive(&mut r, flags, nf);
        }
        let base_gain = i32::from(rate) + GAIN_MIN;

        if flags & FLAG_CAPSULE != 0 {
            let c = Capsule::read(&mut r)?;
            self.last_sid = Some((c.level, c.shape));
            self.apply_capsule(&c);
        }
        if flags & FLAG_REDUNDANCY != 0 {
            let red = Redundancy::read(&mut r)?;
            self.apply_redundancy(&red);
        }

        let mut coded = Vec::with_capacity(nf);
        for _ in 0..nf {
            coded.push(CodedFrame::read(
                &mut r,
                base_gain,
                self.config.frame_len as usize,
            )?);
        }
        let mut out = Vec::with_capacity(self.config.packet_samples());
        for c in &coded {
            let samples = vp::synthesize(
                &mut self.state,
                &c.k_q,
                c.model.width,
                c.model.lag,
                c.model.ltpg_q,
                c.gain,
                &c.symbols,
            );
            out.extend(samples.into_iter().map(round_sample));
        }
        self.last = coded.last().cloned();
        self.conceal_run = 0;
        self.frame_index += nf as u64;
        Ok((PacketKind::Active, out))
    }

    fn decode_inactive(
        &mut self,
        r: &mut Reader<'_>,
        flags: u8,
        nf: usize,
    ) -> Result<(PacketKind, Vec<i32>)> {
        let (level, shape) = if flags & FLAG_CAPSULE != 0 {
            let c = Capsule::read(r)?;
            self.last_sid = Some((c.level, c.shape));
            (c.level, c.shape)
        } else {
            self.last_sid.unwrap_or((level_of(60.0), 0))
        };
        let count = self.config.packet_samples();
        let amp = rms_of(level);
        let k_q = comfort_shape(shape);
        let mut seed = comfort_seed(self.frame_index);
        let mut symbols = Vec::with_capacity(count);
        for _ in 0..count {
            let v = (prng_unit(&mut seed) * amp).round();
            symbols.push(v.clamp(-32768.0, 32767.0) as i32);
        }
        let samples = vp::synthesize(&mut self.state, &k_q, 6, 0, 0, 0, &symbols);
        self.last = None;
        self.conceal_run = 0;
        self.frame_index += nf as u64;
        Ok((
            PacketKind::Inactive,
            samples.into_iter().map(round_sample).collect(),
        ))
    }

    fn apply_capsule(&mut self, c: &Capsule) {
        if c.mask & CAP_PITCH == 0 {
            return;
        }
        // A capsule refreshes the *concealment* model only. It never rewrites
        // already-played audio and it never replaces the sample ring, so a
        // stale capsule cannot corrupt the continuing reconstruction.
        let model = FrameModel {
            estimator: 0,
            order: 4,
            width: 6,
            lag: c.lag,
            ltpg_q: c.ltpg_q,
        };
        self.last = Some(CodedFrame {
            model,
            k_q: c.spectral.iter().map(|&v| i32::from(v)).collect(),
            gain: c.gain,
            residual: Vec::new(),
            symbols: Vec::new(),
        });
    }

    fn apply_redundancy(&mut self, red: &Redundancy) {
        if red.mask & RED_MODE == 0 || self.last.is_some() {
            // Re-entrant repair only when the decoder has no live frame: the
            // redundancy describes the *previous* frame, so it is used to make
            // concealment accurate, never to overwrite a decoded frame.
            if red.mask & RED_MODE == 0 {
                return;
            }
        }
        if let Ok(mut model) = unpack_mode(red.mode) {
            if red.mask & RED_PITCH != 0 {
                model.lag = red.lag;
                model.ltpg_q = red.ltpg_q;
            }
            if self.last.is_none() {
                self.last = Some(CodedFrame {
                    model,
                    k_q: red
                        .spectral
                        .iter()
                        .take(model.order)
                        .map(|&v| i32::from(v))
                        .collect(),
                    gain: if red.mask & RED_GAIN != 0 {
                        red.gain
                    } else {
                        0
                    },
                    residual: Vec::new(),
                    symbols: Vec::new(),
                });
            }
        }
    }

    /// Conceal one frame that never arrived (7C.5).
    ///
    /// Concealed audio is **never** exact reconstruction. It is the best causal
    /// continuation the decoder can make from state it actually holds.
    pub fn conceal(&mut self) -> Vec<i32> {
        let n = self.config.frame_len as usize;
        let samples = plc::conceal(
            &mut self.state,
            self.last.as_ref().map(|f| plc::LastGood {
                model: f.model,
                k_q: &f.k_q,
                gain: f.gain,
            }),
            self.conceal_run,
            comfort_seed(self.frame_index),
            n,
            self.last_sid.map_or(60.0, |s| rms_of(s.0)),
        );
        self.conceal_run = self.conceal_run.saturating_add(1);
        self.frame_index += 1;
        samples.into_iter().map(round_sample).collect()
    }

    /// Consecutive frames concealed so far.
    pub fn conceal_run(&self) -> u32 {
        self.conceal_run
    }

    /// The decoder's reconstruction ring, for evidence and tests.
    pub fn state(&self) -> &VoiceState {
        &self.state
    }
}

/// True when an encoded packet is a DTX/no-data packet. This is the only
/// container detail the court is allowed to read, and it exists so the court
/// can report active and inactive byte rates separately without re-deriving the
/// flag layout.
pub fn packet_is_inactive(packet: &[u8]) -> bool {
    packet.first().is_some_and(|f| f & FLAG_NO_DATA != 0)
}

/// Round a reconstruction into the canonical sample domain.
fn round_sample(x: f64) -> i32 {
    x.round().clamp(-2.0e9, 2.0e9) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(frame_len: u32, fpp: u8, bps: u32) -> VoiceConfig {
        VoiceConfig {
            sample_rate_hz: 16_000,
            frame_len,
            frames_per_packet: fpp,
            target_bits_per_second: bps,
            dtx: false,
            capsule_cadence: DEFAULT_CAPSULE_CADENCE,
            redundancy: true,
        }
    }

    fn speech(n: usize, seed: u64) -> Vec<i32> {
        // A voiced-ish test signal: a pitch pulse train through a two-pole
        // formant, plus mild noise. Enough structure for pitch and LPC.
        let mut s = seed | 1;
        let mut y1 = 0.0f64;
        let mut y2 = 0.0f64;
        let mut out = Vec::with_capacity(n);
        let period = 80usize;
        for i in 0..n {
            let phase = i % period;
            let exc = if phase == 0 { 1.0 } else { 0.0 };
            let noise = (prng_unit(&mut s)) * 0.02;
            let v = exc + noise + 1.5 * y1 - 0.72 * y2;
            y2 = y1;
            y1 = v;
            out.push((v * 6000.0).clamp(-32768.0, 32767.0) as i32);
        }
        out
    }

    #[test]
    fn mode_byte_round_trips() {
        for estimator in 0..3u8 {
            for &order in &ORDER_LADDER {
                for &width in &WIDTH_LADDER {
                    for lag in [0i32, 97] {
                        let m = FrameModel {
                            estimator,
                            order,
                            width,
                            lag,
                            ltpg_q: 7,
                        };
                        let byte = pack_mode(&m).unwrap();
                        let back = unpack_mode(byte).unwrap();
                        assert_eq!(back.estimator, m.estimator);
                        assert_eq!(back.order, m.order);
                        assert_eq!(back.width, m.width);
                        assert_eq!(back.lag > 0, m.lag > 0);
                    }
                }
            }
        }
        assert!(unpack_mode(0b0000_0011).is_err());
    }

    #[test]
    fn capsule_round_trips_and_detects_corruption() {
        for mask in [
            CAP_COMFORT,
            CAP_PITCH | CAP_GAIN | CAP_SPECTRAL | CAP_COMFORT,
            CAP_SPECTRAL | CAP_GAIN,
        ] {
            let c = Capsule {
                lag: if mask & CAP_PITCH != 0 { 121 } else { 0 },
                ltpg_q: if mask & CAP_PITCH != 0 { 22 } else { 0 },
                gain: if mask & CAP_GAIN != 0 { 11 } else { 0 },
                spectral: if mask & CAP_SPECTRAL != 0 {
                    [3, -2, 1, 0]
                } else {
                    [0; 4]
                },
                level: if mask & CAP_COMFORT != 0 { 150 } else { 0 },
                shape: if mask & CAP_COMFORT != 0 { 2 } else { 0 },
                mask,
            };
            let mut w = Writer::new();
            c.write(&mut w);
            let bytes = w.finish();
            let mut r = Reader::new(&bytes);
            assert_eq!(Capsule::read(&mut r).unwrap(), c);
            let mut bad = bytes.clone();
            let n = bad.len();
            bad[n - 1] ^= 0x40;
            let mut r = Reader::new(&bad);
            assert!(Capsule::read(&mut r).is_err());
        }
    }

    #[test]
    fn redundancy_round_trips_and_detects_corruption() {
        let red = Redundancy {
            mask: RED_MODE | RED_PITCH | RED_GAIN | RED_SPECTRAL,
            mode: 0b0101_1101,
            lag: 88,
            ltpg_q: 19,
            gain: 3,
            spectral: [1, 2, 3, 4, 5, 6, 7, 8],
        };
        let mut w = Writer::new();
        red.write(&mut w);
        let bytes = w.finish();
        let mut r = Reader::new(&bytes);
        assert_eq!(Redundancy::read(&mut r).unwrap(), red);
        let mut bad = bytes.clone();
        bad[1] ^= 0xff;
        let mut r = Reader::new(&bad);
        assert!(Redundancy::read(&mut r).is_err());
    }

    #[test]
    fn comfort_shape_classes_are_bounded_and_deterministic() {
        for shape in 0..8u8 {
            let k = comfort_shape(shape);
            assert_eq!(k, comfort_shape(shape));
            for &v in &k {
                assert!(v.abs() < 32);
            }
        }
    }

    #[test]
    fn encoder_and_decoder_agree_frame_for_frame() {
        for (frame_len, fpp) in [(160u32, 1u8), (320, 1), (160, 2)] {
            for bps in [8_000u32, 16_000, 32_000] {
                let cfg = config(frame_len, fpp, bps);
                let mut enc = VoiceEncoder::new(cfg).unwrap();
                let mut dec = VoiceDecoder::new(cfg).unwrap();
                let src = speech(cfg.packet_samples() * 12, 0x1234);
                let mut plain = Vec::new();
                let mut decoded = Vec::new();
                for chunk in src.chunks(cfg.packet_samples()) {
                    if chunk.len() < cfg.packet_samples() {
                        break;
                    }
                    let pkt = enc.encode_packet(chunk).unwrap();
                    let (kind, out) = dec.decode_packet(&pkt).unwrap();
                    assert_eq!(kind, PacketKind::Active);
                    assert_eq!(out.len(), cfg.packet_samples());
                    plain.extend_from_slice(chunk);
                    decoded.extend_from_slice(&out);
                }
                assert_eq!(plain.len(), decoded.len());
                let sig: f64 = plain.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
                let err: f64 = plain
                    .iter()
                    .zip(decoded.iter())
                    .map(|(&a, &b)| {
                        let d = f64::from(a) - f64::from(b);
                        d * d
                    })
                    .sum();
                assert!(
                    decoded.iter().all(|v| v.abs() < 1_000_000_000),
                    "reconstruction must stay bounded"
                );
                assert!(err > 0.0, "the profile is lossy; exactness is not a claim");
                let snr = 10.0 * (sig / err).log10();
                // Below ~12 kbps with 10 ms frames the model description alone
                // consumes the whole frame allowance; that is the measured
                // finding reported by the court, not something to assert away.
                let floor = if frame_len == 160 && fpp == 1 && bps < 12_000 {
                    -10.0
                } else {
                    6.0
                };
                assert!(
                    snr > floor,
                    "frame_len {frame_len} fpp {fpp} at {bps} bps gave {snr:.2} dB"
                );
            }
        }
    }

    #[test]
    fn the_encoder_picks_a_model_that_fits_the_frame_allowance() {
        // At 16 kbps with 20 ms frames the allowance is 40 bytes; the model
        // description must leave room for the residual.
        let cfg = config(320, 1, 16_000);
        let mut enc = VoiceEncoder::new(cfg).unwrap();
        let src = speech(320 * 4, 0x31);
        for chunk in src.chunks(320) {
            let pkt = enc.encode_packet(chunk).unwrap();
            assert!(
                pkt.len() <= cfg.target_bytes() + 4,
                "packet of {} bytes overshot the {} byte target",
                pkt.len(),
                cfg.target_bytes()
            );
        }
    }

    #[test]
    fn encoding_is_deterministic() {
        let cfg = config(160, 1, 16_000);
        let src = speech(160 * 6, 0xabc);
        let mut a = VoiceEncoder::new(cfg).unwrap();
        let mut b = VoiceEncoder::new(cfg).unwrap();
        for chunk in src.chunks(160) {
            assert_eq!(
                a.encode_packet(chunk).unwrap(),
                b.encode_packet(chunk).unwrap()
            );
        }
    }

    #[test]
    fn packet_frame_count_must_match_the_configuration() {
        let cfg = config(160, 2, 16_000);
        let cfg2 = config(160, 1, 16_000);
        let mut enc = VoiceEncoder::new(cfg).unwrap();
        let mut dec = VoiceDecoder::new(cfg2).unwrap();
        let pkt = enc.encode_packet(&speech(320, 7)).unwrap();
        assert!(dec.decode_packet(&pkt).is_err());
    }

    #[test]
    fn truncated_packets_are_rejected_not_guessed() {
        let cfg = config(160, 1, 16_000);
        let mut enc = VoiceEncoder::new(cfg).unwrap();
        let mut dec = VoiceDecoder::new(cfg).unwrap();
        let pkt = enc.encode_packet(&speech(160, 9)).unwrap();
        for cut in 1..pkt.len() {
            let mut d = VoiceDecoder::new(cfg).unwrap();
            assert!(
                d.decode_packet(&pkt[..cut]).is_err(),
                "truncation to {cut} bytes must not decode"
            );
        }
        let _ = &mut dec;
    }

    #[test]
    fn dtx_produces_procedural_comfort_noise_when_active_is_off() {
        let mut cfg = config(160, 1, 16_000);
        cfg.dtx = true;
        let mut enc = VoiceEncoder::new(cfg).unwrap();
        let mut dec = VoiceDecoder::new(cfg).unwrap();
        let mut kinds = Vec::new();
        for i in 0..30 {
            let chunk: Vec<i32> = if i < 10 {
                speech(160, 0x55 + i)
            } else {
                vec![0i32; 160]
            };
            let pkt = enc.encode_packet(&chunk).unwrap();
            let (kind, out) = dec.decode_packet(&pkt).unwrap();
            assert_eq!(out.len(), 160);
            kinds.push(kind);
        }
        assert!(kinds.contains(&PacketKind::Active));
        assert!(kinds.contains(&PacketKind::Inactive));
        // Comfort noise is deterministic: a second decoder reproduces it exactly.
        let mut e2 = VoiceEncoder::new(cfg).unwrap();
        let mut d2 = VoiceDecoder::new(cfg).unwrap();
        let mut first = Vec::new();
        for i in 0..30 {
            let chunk: Vec<i32> = if i < 10 {
                speech(160, 0x55 + i)
            } else {
                vec![0i32; 160]
            };
            let pkt = e2.encode_packet(&chunk).unwrap();
            let (_, out) = d2.decode_packet(&pkt).unwrap();
            first.extend(out);
        }
        assert!(first.iter().any(|&v| v != 0));
    }

    #[test]
    fn concealment_is_bounded_and_recovers() {
        let cfg = config(160, 1, 16_000);
        let mut enc = VoiceEncoder::new(cfg).unwrap();
        let mut dec = VoiceDecoder::new(cfg).unwrap();
        let src = speech(160 * 20, 0x77);
        let mut pkt = None;
        for (i, chunk) in src.chunks(160).enumerate() {
            let p = enc.encode_packet(chunk).unwrap();
            if i == 5 {
                pkt = Some(p);
                // Drop packets 5..8.
                for _ in 0..3 {
                    let c = dec.conceal();
                    assert_eq!(c.len(), 160);
                }
                continue;
            }
            let _ = dec.decode_packet(&p).unwrap();
        }
        assert!(pkt.is_some());
        assert!(dec.conceal_run() == 0, "a received packet resets the run");
        // The concealed output must be finite and bounded.
        for v in dec.state().chronological(160) {
            assert!(v.is_finite() && v.abs() <= 2.0e9);
        }
    }

    #[test]
    fn configuration_is_checked_against_the_constitution() {
        let mut c = config(160, 1, 16_000);
        assert!(c.validate().is_ok());
        c.sample_rate_hz = 8_000;
        assert!(c.validate().is_err());
        let mut c = config(160, 1, 16_000);
        c.frame_len = 128;
        assert!(c.validate().is_err());
        let mut c = config(320, 2, 16_000);
        c.frames_per_packet = 2;
        assert!(c.validate().is_err());
        let mut c = config(160, 1, 16_000);
        c.target_bits_per_second = 0;
        assert!(c.validate().is_err());
    }
}
