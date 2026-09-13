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
                        // Saturating subtraction keeps extreme `Delay` offsets
                        // total and panic-free; in the ordinary domain it is
                        // identical to plain subtraction.
                        let src = (t as i64).saturating_sub(*frames);
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

    /// Materialize only the absolute frame window `[start, start + frames)`.
    ///
    /// This is the **bounded** materialization path. It evaluates the requested
    /// interval only: oscillator phase and envelope level are derived from the
    /// *absolute* frame index, `Delay` requests its child at a translated
    /// position, and `Add`/`Gain`/`Envelope` transform the same bounded window.
    ///
    /// Working memory and traversal work scale with the requested window and
    /// the graph's structural depth (`MAX_COMPOUND_NODES`), **not** with the
    /// graph's total duration. The result is bit-identical to
    /// `materialize()?[start .. start + frames]`.
    pub fn materialize_range(&self, start: u64, frames: u64) -> Result<Vec<i32>> {
        Ok(self.materialize_range_profiled(start, frames)?.0)
    }

    /// As [`materialize_range`](Self::materialize_range), additionally returning
    /// the measured traversal profile so callers can prove the bound.
    pub fn materialize_range_profiled(
        &self,
        start: u64,
        frames: u64,
    ) -> Result<(Vec<i32>, RangeProfile)> {
        self.validate()?;
        let end = start
            .checked_add(frames)
            .ok_or_else(|| Error::limit("compound window end overflows"))?;
        if end > self.frames {
            return Err(Error::limit("compound window exceeds the graph extent"));
        }
        let mut prof = RangeProfile::default();
        let mut out = Vec::with_capacity(frames as usize);
        let root = self.root();
        for i in 0..frames {
            let t = (start + i) as i64;
            let mut depth = 0u32;
            out.push(self.eval_frame(root, t, &mut depth, &mut prof));
        }
        Ok((out, prof))
    }

    /// Pull a single absolute output frame through the graph.
    ///
    /// Depth is bounded by the longest child path, which [`validate`] restricts
    /// to `MAX_COMPOUND_NODES` because every child references an earlier index.
    /// A diamond DAG is therefore re-visited once per edge, never exponentially:
    /// each frame costs `O(edges)` work and `O(depth)` stack.
    ///
    /// [`validate`]: Self::validate
    fn eval_frame(&self, node: usize, t: i64, depth: &mut u32, prof: &mut RangeProfile) -> i32 {
        *depth += 1;
        prof.peak_depth = prof.peak_depth.max(*depth);
        prof.node_visits += 1;
        let nd = &self.nodes[node];
        let n = self.frames as i64;
        let v = match &nd.op {
            CompoundOp::Silence => 0,
            CompoundOp::Constant { level } => *level,
            CompoundOp::Oscillator {
                freq_hz,
                amp_q16,
                phase0,
            } => {
                let incr = eff_incr(*freq_hz, 1 << 24, self.sample_rate_hz);
                osc_sample(osc_phase(*phase0, incr, 0, t), *amp_q16)
            }
            CompoundOp::Gain { q16 } => {
                let c = self.eval_frame(usize::from(nd.children[0]), t, depth, prof);
                sat_i32(rnd_shift(i64::from(c) * i64::from(*q16), 16))
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
                let c = self.eval_frame(usize::from(nd.children[0]), t, depth, prof);
                let level = env.level_at(*t_on, *t_off, t);
                sat_i32(rnd_shift(i64::from(c) * i64::from(level), 16))
            }
            CompoundOp::Delay { frames: d } => {
                let src = t.saturating_sub(*d);
                if src >= 0 && src < n {
                    self.eval_frame(usize::from(nd.children[0]), src, depth, prof)
                } else {
                    0
                }
            }
            CompoundOp::Add => {
                let mut acc = 0i64;
                for &ch in &nd.children {
                    acc += i64::from(self.eval_frame(usize::from(ch), t, depth, prof));
                }
                sat_i32(acc)
            }
        };
        *depth -= 1;
        v
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

/// Measured cost of one bounded-materialization pass.
///
/// `peak_depth` is the deepest simultaneous child path touched while pulling a
/// single output frame; it is bounded by the graph's node count and is
/// independent of the requested window and of the graph's total duration.
/// `node_visits` counts node evaluations across the whole requested window.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RangeProfile {
    pub peak_depth: u32,
    pub node_visits: u64,
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

    /// A graph whose evaluator exercises every operation and every boundary
    /// class a bounded window can split: oscillator phase, envelope attack /
    /// decay / sustain / release, gain, and a two-hop delay chain.
    fn boundary_rich_graph(frames: u64) -> CompoundGraph {
        CompoundGraph {
            channels: 1,
            frames,
            sample_rate_hz: 44_100,
            nodes: vec![
                oscillator(220, 1 << 14), // 0
                oscillator(329, 1 << 13), // 1
                CompoundNode {
                    op: CompoundOp::Add, // 2
                    children: vec![0, 1],
                },
                CompoundNode {
                    op: CompoundOp::Envelope {
                        attack_frames: 137,
                        decay_frames: 211,
                        sustain_q16: 1 << 15,
                        release_frames: 353,
                        t_on: 97,
                        t_off: Some((frames / 2) as i64),
                    },
                    children: vec![2],
                },
                CompoundNode {
                    op: CompoundOp::Gain { q16: 1 << 15 }, // 4
                    children: vec![3],
                },
                CompoundNode {
                    op: CompoundOp::Delay { frames: 64 }, // 5
                    children: vec![4],
                },
                CompoundNode {
                    op: CompoundOp::Delay { frames: 19 }, // 6
                    children: vec![5],
                },
                CompoundNode {
                    op: CompoundOp::Constant { level: 3 }, // 7
                    children: Vec::new(),
                },
                CompoundNode {
                    op: CompoundOp::Add, // 8
                    children: vec![6, 7],
                },
            ],
        }
    }

    /// Small deterministic LCG so the window fuzz is reproducible without a
    /// rand dependency.
    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    #[test]
    fn bounded_materialization_matches_the_reference_slice() {
        let g = boundary_rich_graph(50_000);
        let reference = g.materialize().unwrap();
        // Deterministic randomized windows of varied length and offset, plus
        // the exact operation boundaries.
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut checked = 0u32;
        for _ in 0..400 {
            let start = lcg(&mut state) % g.frames;
            let remaining = g.frames - start;
            let len = 1 + lcg(&mut state) % remaining.min(9000);
            let got = g.materialize_range(start, len).unwrap();
            let want = &reference[start as usize..(start + len) as usize];
            assert_eq!(got.as_slice(), want, "window {start}+{len}");
            checked += 1;
        }
        // Boundary-crossing windows: enveloped t_on=97, the two delay edges at
        // 64 and 64+19, the decay end at 97+137+211, and the release start at
        // frames/2.
        for edge in [0u64, 64, 83, 97, 234, 445, 25_000] {
            for pre in 0..8 {
                let start = edge.saturating_sub(pre);
                let len = 17u64.min(g.frames - start);
                let got = g.materialize_range(start, len).unwrap();
                assert_eq!(
                    got.as_slice(),
                    &reference[start as usize..(start + len) as usize]
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 456);
    }

    #[test]
    fn bounded_materialization_rejects_out_of_extent_windows() {
        let g = boundary_rich_graph(1000);
        assert!(g.materialize_range(0, 1000).is_ok());
        assert!(g.materialize_range(1000, 0).is_ok());
        assert!(g.materialize_range(999, 2).is_err());
        assert!(g.materialize_range(1001, 0).is_err());
        assert!(g.materialize_range(u64::MAX, 1).is_err());
    }

    #[test]
    fn bounded_scratch_is_independent_of_duration() {
        let short = boundary_rich_graph(1_000);
        let long = boundary_rich_graph(4_000_000);
        let (sv_s, prof_s) = short.materialize_range_profiled(500, 64).unwrap();
        let reference_short = short.materialize().unwrap();
        assert_eq!(sv_s.as_slice(), &reference_short[500..564]);
        // Same topology => same structural depth, regardless of extent.
        let (_, prof_long) = long.materialize_range_profiled(3_000_000, 64).unwrap();
        assert_eq!(prof_s.peak_depth, prof_long.peak_depth);
        // Traversal work scales with the requested window, not the duration.
        assert_eq!(prof_s.node_visits, prof_long.node_visits);
        // Depth is bounded by the node count, never by the frame count.
        assert!(prof_s.peak_depth as usize <= long.nodes.len());
        // A 64-frame window cannot visit the whole graph per frame repeatedly;
        // the per-frame cost is exactly the reachable edge count.
        assert!(prof_s.node_visits <= 64 * long.nodes.len() as u64);
    }
}
