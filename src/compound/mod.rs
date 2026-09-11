//! Native procedural composition (optimization Track A), **experimental
//! profile** `vole.audio.compound.exp1`.
//!
//! ## Why a new profile (the spec audit)
//!
//! `docs/U1_SPEC.md` is the normative authority for `u1/v1`. It contains **no**
//! Compound payload syntax and **no** Compound observation semantics: the
//! `Representation::Compound = 0x0A` tag exists in code with only a one-line
//! descriptor comment. Implementing a payload behind `0x0A` would therefore be
//! a silent semantic break of the frozen profile.
//!
//! Instead this module defines a **separate, explicitly versioned experimental
//! profile** whose bytes can never be mistaken for a `u1/v1` `SampleObject`.
//! Nothing here changes `u1/v1`, the literal fallback, or any existing tag.
//!
//! ## What it is
//!
//! A bounded deterministic composition graph over already-exact primitives:
//! silence, constant, and the frozen DDS oscillator (the frozen sine table and
//! `eff_incr` law), combined with gain, integer delay, exact linear ADSR
//! (the frozen `EnvelopeParams` law) and an exact i64 `Add` that saturates once
//! at the output boundary. Every node is a total deterministic function of the
//! graph; evaluation is integer-only.
//!
//! This is deliberately **not** an open-ended DSP language: the vocabulary is
//! small, every node has bounded cost, and no transcendental function appears.

use crate::error::{Error, Result};
use crate::hash::sha256::Sha256;
use crate::sampler::envelope::EnvelopeParams;
use crate::sampler::procedural::{eff_incr, osc_phase, osc_sample};
use crate::universe::arithmetic::{rnd_shift, sat_i32};

/// Experimental composition profile identity.
pub const COMPOUND_PROFILE: &str = "vole.audio.u1/vole.audio.compound.exp1";

/// Canonical profile tag bytes.
pub const COMPOUND_PROFILE_TAG: &[u8] = b"vole.audio.u1/vole.audio.compound.exp1";

/// Container magic.
pub const COMPOUND_MAGIC: &[u8; 13] = b"vole.compound";

/// Container format version.
pub const COMPOUND_FORMAT_VERSION: u8 = 1;

/// Maximum nodes in one composition graph.
pub const MAX_COMPOUND_NODES: u32 = 256;

/// Maximum children of one `Add` node.
pub const MAX_COMPOUND_ARITY: u32 = 64;

/// One node operation (frozen opcode ids in this experimental profile).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompoundOp {
    /// Endless silence.
    Silence,
    /// Constant code-domain level.
    Constant { level: i32 },
    /// Frozen DDS oscillator (mono).
    Oscillator {
        freq_hz: u32,
        amp_q16: i32,
        phase0: u64,
    },
    /// Multiply the single child by a Q16 gain.
    Gain { q16: i32 },
    /// Shift the single child in time (positive = later).
    Delay { frames: i64 },
    /// Exact i64 sum of children, saturated once.
    Add,
    /// Multiply the single child by the frozen analytic ADSR level.
    Envelope {
        attack_frames: u32,
        decay_frames: u32,
        sustain_q16: i32,
        release_frames: u32,
        t_on: i64,
        t_off: Option<i64>,
    },
}

impl CompoundOp {
    pub const fn tag(&self) -> u8 {
        match self {
            CompoundOp::Silence => 0,
            CompoundOp::Constant { .. } => 1,
            CompoundOp::Oscillator { .. } => 2,
            CompoundOp::Gain { .. } => 3,
            CompoundOp::Delay { .. } => 4,
            CompoundOp::Add => 5,
            CompoundOp::Envelope { .. } => 6,
        }
    }
    pub const fn arity(&self) -> usize {
        match self {
            CompoundOp::Gain { .. } | CompoundOp::Delay { .. } | CompoundOp::Envelope { .. } => 1,
            _ => 0,
        }
    }
}

/// One node: an operation and its (earlier-indexed) children.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompoundNode {
    pub op: CompoundOp,
    pub children: Vec<u16>,
}

/// A bounded deterministic composition graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompoundGraph {
    pub channels: u8,
    pub frames: u64,
    pub sample_rate_hz: u32,
    /// Topological order: a node may only reference earlier indices.
    pub nodes: Vec<CompoundNode>,
}

impl CompoundGraph {
    fn root(&self) -> usize {
        self.nodes.len() - 1
    }

    /// Validate structure, arity, ordering and bounds.
    pub fn validate(&self) -> Result<()> {
        if self.channels != 1 {
            return Err(Error::new(
                crate::error::Kind::Unsupported,
                "compound composition is mono-first in this build",
            ));
        }
        if self.frames == 0 || self.frames > crate::limits::MAX_OBJECT_FRAMES {
            return Err(Error::limit("compound frame count exceeds the bound"));
        }
        if self.sample_rate_hz == 0 || self.sample_rate_hz > crate::limits::MAX_SAMPLE_RATE_HZ {
            return Err(Error::malformed("compound sample rate out of domain"));
        }
        if self.nodes.is_empty() || self.nodes.len() as u32 > MAX_COMPOUND_NODES {
            return Err(Error::limit("compound node count out of range"));
        }
        for (i, n) in self.nodes.iter().enumerate() {
            if n.children.len() as u32 > MAX_COMPOUND_ARITY {
                return Err(Error::limit("compound node arity exceeds the bound"));
            }
            match &n.op {
                CompoundOp::Add => {
                    if n.children.is_empty() {
                        return Err(Error::malformed("compound Add has no children"));
                    }
                }
                op => {
                    if n.children.len() != op.arity() {
                        return Err(Error::malformed("compound node arity mismatch"));
                    }
                }
            }
            for &c in &n.children {
                if usize::from(c) >= i {
                    return Err(Error::malformed(
                        "compound children must reference earlier nodes",
                    ));
                }
            }
            if let CompoundOp::Envelope {
                sustain_q16,
                attack_frames,
                decay_frames,
                release_frames,
                ..
            } = &n.op
                && EnvelopeParams::new(*attack_frames, *decay_frames, *sustain_q16, *release_frames)
                    .is_none()
            {
                return Err(Error::malformed(
                    "compound envelope parameters out of domain",
                ));
            }
        }
        Ok(())
    }

    /// Evaluate every node into a per-node code-domain buffer.
    fn evaluate_nodes(&self) -> Vec<Vec<i32>> {
        let n = self.frames as usize;
        let mut bufs: Vec<Vec<i32>> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let out = match &node.op {
                CompoundOp::Silence => vec![0i32; n],
                CompoundOp::Constant { level } => vec![*level; n],
                CompoundOp::Oscillator {
                    freq_hz,
                    amp_q16,
                    phase0,
                } => {
                    let incr = eff_incr(*freq_hz, 1 << 24, self.sample_rate_hz);
                    (0..n)
                        .map(|t| osc_sample(osc_phase(*phase0, incr, 0, t as i64), *amp_q16))
                        .collect()
                }
                CompoundOp::Gain { q16 } => {
                    let child = &bufs[usize::from(node.children[0])];
                    child
                        .iter()
                        .map(|&v| sat_i32(rnd_shift(i64::from(v) * i64::from(*q16), 16)))
                        .collect()
                }
                CompoundOp::Delay { frames } => {
                    let child = &bufs[usize::from(node.children[0])];
                    let mut out = vec![0i32; n];
                    for (t, slot) in out.iter_mut().enumerate() {
                        let src = t as i64 - *frames;
                        if src >= 0 && (src as usize) < n {
                            *slot = child[src as usize];
                        }
                    }
                    out
                }
                CompoundOp::Add => {
                    let mut acc = vec![0i64; n];
                    for &c in &node.children {
                        for (a, &v) in acc.iter_mut().zip(bufs[usize::from(c)].iter()) {
                            *a += i64::from(v);
                        }
                    }
                    acc.into_iter().map(sat_i32).collect()
                }
                CompoundOp::Envelope {
                    attack_frames,
                    decay_frames,
                    sustain_q16,
                    release_frames,
                    t_on,
                    t_off,
                } => {
                    let env = EnvelopeParams::new(
                        *attack_frames,
                        *decay_frames,
                        *sustain_q16,
                        *release_frames,
                    )
                    .expect("validated");
                    let child = &bufs[usize::from(node.children[0])];
                    (0..n)
                        .map(|t| {
                            let level = env.level_at(*t_on, *t_off, t as i64);
                            sat_i32(rnd_shift(i64::from(child[t]) * i64::from(level), 16))
                        })
                        .collect()
                }
            };
            bufs.push(out);
        }
        bufs
    }

    /// Materialize the canonical intrinsic signal (mono, `frames` samples).
    pub fn materialize(&self) -> Result<Vec<i32>> {
        self.validate()?;
        let bufs = self.evaluate_nodes();
        Ok(bufs[self.root()].clone())
    }

    /// Canonical bytes:
    /// `MAGIC(12) VERSION(1) PROFILE_LEN(1) PROFILE CHANNELS(1) FRAMES(u64)
    ///  RATE(u32) NODES(u32) [op children payload] DIGEST(32)`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(COMPOUND_MAGIC);
        out.push(COMPOUND_FORMAT_VERSION);
        out.push(COMPOUND_PROFILE_TAG.len() as u8);
        out.extend_from_slice(COMPOUND_PROFILE_TAG);
        out.push(self.channels);
        out.extend_from_slice(&self.frames.to_le_bytes());
        out.extend_from_slice(&self.sample_rate_hz.to_le_bytes());
        out.extend_from_slice(&(self.nodes.len() as u32).to_le_bytes());
        for n in &self.nodes {
            out.push(n.op.tag());
            out.extend_from_slice(&(n.children.len() as u16).to_le_bytes());
            for &c in &n.children {
                out.extend_from_slice(&c.to_le_bytes());
            }
            match &n.op {
                CompoundOp::Silence | CompoundOp::Add => {}
                CompoundOp::Constant { level } => out.extend_from_slice(&level.to_le_bytes()),
                CompoundOp::Oscillator {
                    freq_hz,
                    amp_q16,
                    phase0,
                } => {
                    out.extend_from_slice(&freq_hz.to_le_bytes());
                    out.extend_from_slice(&amp_q16.to_le_bytes());
                    out.extend_from_slice(&phase0.to_le_bytes());
                }
                CompoundOp::Gain { q16 } => out.extend_from_slice(&q16.to_le_bytes()),
                CompoundOp::Delay { frames } => out.extend_from_slice(&frames.to_le_bytes()),
                CompoundOp::Envelope {
                    attack_frames,
                    decay_frames,
                    sustain_q16,
                    release_frames,
                    t_on,
                    t_off,
                } => {
                    out.extend_from_slice(&attack_frames.to_le_bytes());
                    out.extend_from_slice(&decay_frames.to_le_bytes());
                    out.extend_from_slice(&sustain_q16.to_le_bytes());
                    out.extend_from_slice(&release_frames.to_le_bytes());
                    out.extend_from_slice(&t_on.to_le_bytes());
                    match t_off {
                        Some(off) => {
                            out.push(1);
                            out.extend_from_slice(&off.to_le_bytes());
                        }
                        None => out.push(0),
                    }
                }
            }
        }
        let digest = Sha256::digest(&out);
        out.extend_from_slice(&digest);
        out
    }

    /// Content identity: SHA-256 of the canonical bytes.
    pub fn content_id(&self) -> [u8; 32] {
        Sha256::digest(&self.canonical_bytes())
    }

    /// Parse and validate canonical bytes.
    pub fn parse(bytes: &[u8]) -> Result<CompoundGraph> {
        let mut r = CompoundReader::new(bytes);
        r.take(COMPOUND_MAGIC.len())?;
        let version = r.u8()?;
        if version != COMPOUND_FORMAT_VERSION {
            return Err(Error::new(
                crate::error::Kind::Unsupported,
                format!("unsupported compound version {version}"),
            ));
        }
        let plen = r.u8()? as usize;
        let profile = r.take(plen)?;
        if profile != COMPOUND_PROFILE_TAG {
            return Err(Error::malformed("compound profile tag mismatch"));
        }
        let channels = r.u8()?;
        let frames = r.u64()?;
        let sample_rate_hz = r.u32()?;
        let node_count = r.u32()?;
        if node_count == 0 || node_count > MAX_COMPOUND_NODES {
            return Err(Error::limit("compound node count out of range"));
        }
        let mut nodes = Vec::with_capacity(node_count as usize);
        for _ in 0..node_count {
            let tag = r.u8()?;
            let cc = r.u16()? as usize;
            if cc > MAX_COMPOUND_ARITY as usize {
                return Err(Error::limit("compound node arity exceeds the bound"));
            }
            let mut children = Vec::with_capacity(cc);
            for _ in 0..cc {
                children.push(r.u16()?);
            }
            let op = match tag {
                0 => CompoundOp::Silence,
                1 => CompoundOp::Constant { level: r.i32()? },
                2 => CompoundOp::Oscillator {
                    freq_hz: r.u32()?,
                    amp_q16: r.i32()?,
                    phase0: r.u64()?,
                },
                3 => CompoundOp::Gain { q16: r.i32()? },
                4 => CompoundOp::Delay { frames: r.i64()? },
                5 => CompoundOp::Add,
                6 => {
                    let attack_frames = r.u32()?;
                    let decay_frames = r.u32()?;
                    let sustain_q16 = r.i32()?;
                    let release_frames = r.u32()?;
                    let t_on = r.i64()?;
                    let has_off = r.u8()?;
                    let t_off = if has_off == 1 { Some(r.i64()?) } else { None };
                    CompoundOp::Envelope {
                        attack_frames,
                        decay_frames,
                        sustain_q16,
                        release_frames,
                        t_on,
                        t_off,
                    }
                }
                other => {
                    return Err(Error::new(
                        crate::error::Kind::Unsupported,
                        format!("unknown compound opcode {other}"),
                    ));
                }
            };
            nodes.push(CompoundNode { op, children });
        }
        if r.remaining() != 32 {
            return Err(Error::malformed("compound digest framing mismatch"));
        }
        let stated = r.array32()?;
        let body_end = bytes.len() - 32;
        if Sha256::digest(&bytes[..body_end]) != stated {
            return Err(Error::new(
                crate::error::Kind::Integrity,
                "compound digest mismatch",
            ));
        }
        let g = CompoundGraph {
            channels,
            frames,
            sample_rate_hz,
            nodes,
        };
        g.validate()?;
        Ok(g)
    }

    /// Complete stored bytes (canonical bytes; framing is inside them).
    pub fn complete_bytes(&self) -> u64 {
        self.canonical_bytes().len() as u64
    }

    /// Exact closure: materialize and compare to a target signal.
    pub fn closes_to(&self, target: &[i32]) -> bool {
        match self.materialize() {
            Ok(v) => v == target,
            Err(_) => false,
        }
    }
}

struct CompoundReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> CompoundReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        CompoundReader { bytes, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| Error::limit("compound read overflows"))?;
        let s = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| Error::malformed("compound bytes are truncated"))?;
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
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn array32(&mut self) -> Result<[u8; 32]> {
        Ok(self.take(32)?.try_into().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oscillator(freq: u32, amp: i32) -> CompoundNode {
        CompoundNode {
            op: CompoundOp::Oscillator {
                freq_hz: freq,
                amp_q16: amp,
                phase0: 0,
            },
            children: Vec::new(),
        }
    }

    #[test]
    fn polyphonic_sum_materializes_and_round_trips() {
        let n = 4096u64;
        let g = CompoundGraph {
            channels: 1,
            frames: n,
            sample_rate_hz: 48_000,
            nodes: vec![
                oscillator(220, 1 << 14),
                oscillator(330, 1 << 14),
                oscillator(440, 1 << 14),
                CompoundNode {
                    op: CompoundOp::Add,
                    children: vec![0, 1, 2],
                },
            ],
        };
        let s = g.materialize().unwrap();
        assert_eq!(s.len(), n as usize);
        assert!(s.iter().any(|&v| v != 0));
        let bytes = g.canonical_bytes();
        assert!(bytes.len() < 240, "compound bytes {}", bytes.len());
        let back = CompoundGraph::parse(&bytes).unwrap();
        assert_eq!(back, g);
        assert_eq!(back.materialize().unwrap(), s);
        assert!(back.closes_to(&s));
    }

    #[test]
    fn envelope_and_gain_are_exact_and_bounded() {
        let g = CompoundGraph {
            channels: 1,
            frames: 2048,
            sample_rate_hz: 48_000,
            nodes: vec![
                CompoundNode {
                    op: CompoundOp::Constant { level: 1_000_000 },
                    children: Vec::new(),
                },
                CompoundNode {
                    op: CompoundOp::Envelope {
                        attack_frames: 100,
                        decay_frames: 200,
                        sustain_q16: 1 << 15,
                        release_frames: 300,
                        t_on: 0,
                        t_off: Some(1000),
                    },
                    children: vec![0],
                },
                CompoundNode {
                    op: CompoundOp::Gain { q16: 1 << 15 },
                    children: vec![1],
                },
            ],
        };
        let s = g.materialize().unwrap();
        assert_eq!(s[0], 0);
        assert!(s[150] > 0);
        // Release starts at the level at the note-off frame and then falls.
        assert_eq!(s[1000], s[900]);
        assert!(s[1001] < s[900]);
        assert_eq!(s[2000], 0);
        assert!(g.closes_to(&s));
    }

    #[test]
    fn delay_shifts_into_the_window() {
        let g = CompoundGraph {
            channels: 1,
            frames: 16,
            sample_rate_hz: 48_000,
            nodes: vec![
                CompoundNode {
                    op: CompoundOp::Constant { level: 7 },
                    children: Vec::new(),
                },
                CompoundNode {
                    op: CompoundOp::Delay { frames: 4 },
                    children: vec![0],
                },
            ],
        };
        let s = g.materialize().unwrap();
        assert_eq!(&s[..4], &[0, 0, 0, 0]);
        assert_eq!(&s[4..], &[7; 12]);
    }

    #[test]
    fn malformed_graphs_are_rejected() {
        let bad = CompoundGraph {
            channels: 1,
            frames: 8,
            sample_rate_hz: 48_000,
            nodes: vec![
                CompoundNode {
                    op: CompoundOp::Gain { q16: 1 },
                    children: vec![1],
                },
                CompoundNode {
                    op: CompoundOp::Silence,
                    children: Vec::new(),
                },
            ],
        };
        assert!(bad.validate().is_err());
        let good = CompoundGraph {
            channels: 1,
            frames: 8,
            sample_rate_hz: 48_000,
            nodes: vec![CompoundNode {
                op: CompoundOp::Silence,
                children: Vec::new(),
            }],
        };
        let mut bytes = good.canonical_bytes();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        assert!(CompoundGraph::parse(&bytes).is_err());
    }
}
