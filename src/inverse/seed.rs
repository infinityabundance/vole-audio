//! **Entropy-seed representation** (Phase 7A.3 / 7A.4).
//!
//! ```text
//! X = reconstruct(H, R)
//! ```
//!
//! A [`SeedObject`] persists
//!
//! * `H` — a deterministic [`CompoundGraph`] explanation, itself stored as an
//!   **entropy-coded state stream** (opcodes, child-reference deltas, frequency,
//!   amplitude, phase, gain, delay and ADSR fields, each in its own integer
//!   stream and each coded with the strongest existing metadata codec);
//! * `R` — the **entropy-coded exact residual** that `H` failed to explain,
//!   coded with the repository's mature residual-codec family.
//!
//! The residual is never stored as raw PCM. High-entropy material produces a
//! large residual, which is valid; it does not abandon the entropy-seed form.
//!
//! H's state coding reuses the Phase-2B integer metadata machinery
//! ([`crate::entropy::repcode`]) — a three-entry recent-value cache, a
//! baseline+varint delta stream, or a raw varint stream, whichever is smallest
//! for the stream. No parallel entropy subsystem is introduced: the residual
//! uses [`crate::learned::residual_codec2`] and the state uses the native
//! metadata coders.

use crate::compound::{CompoundGraph, CompoundNode, CompoundOp, MAX_COMPOUND_ARITY};
use crate::entropy::repcode::{decode_baseline, decode_repcode, encode_baseline, encode_repcode};
use crate::error::{Error, Result};
use crate::learned::residual_codec2::{ResidualCodecV2, ResidualEncodingV2, encode_best_v3};

/// Fixed header of the seed container (version byte).
const SEED_VERSION: u8 = 1;

/// The two general-Golomb codecs encode the residual quotient as unary, so a
/// large-magnitude outlier makes them `O(|r|)` per symbol. They are skipped when
/// the residual holds any value above this magnitude; the skip is deterministic
/// and depends only on the residual. (Both codecs remain in the family for the
/// small-magnitude streams they are designed for.)
const GOLOMB_MAGNITUDE_LIMIT: u32 = 1 << 16;

/// Encode the seed residual with the repository residual-codec family, skipping
/// the two unbounded Golomb codecs when a large-magnitude symbol is present.
fn encode_best_seed(residual: &[i32]) -> ResidualEncodingV2 {
    let max_mag = residual.iter().map(|r| r.unsigned_abs()).max().unwrap_or(0);
    let skip_golomb = max_mag > GOLOMB_MAGNITUDE_LIMIT;
    let mut best: Option<ResidualEncodingV2> = None;
    for codec in ResidualCodecV2::ALL_V3 {
        if skip_golomb
            && matches!(
                codec,
                ResidualCodecV2::Golomb | ResidualCodecV2::CenteredGolomb
            )
        {
            continue;
        }
        let payload = codec.encode(residual);
        let mut bytes = Vec::with_capacity(payload.len() + 1);
        bytes.push(codec.id());
        bytes.extend_from_slice(&payload);
        let candidate = ResidualEncodingV2 { codec, bytes };
        if best
            .as_ref()
            .is_none_or(|b| candidate.complete_bytes() < b.complete_bytes())
        {
            best = Some(candidate);
        }
    }
    // `ALL_V3` has no skip when magnitudes are small; the only all-skipped case
    // is impossible because the skip set is a strict subset of the family.
    match best {
        Some(b) => b,
        None => encode_best_v3(residual),
    }
}

/// Number of integer streams in the H state.
const STREAM_COUNT: usize = 11;

// Stream indices.
const S_HEADER: usize = 0;
const S_OPCODES: usize = 1;
const S_CHILD_COUNTS: usize = 2;
const S_CHILD_OFFSETS: usize = 3;
const S_CONST_LEVEL: usize = 4;
const S_OSC_FREQ: usize = 5;
const S_OSC_AMP: usize = 6;
const S_OSC_PHASE: usize = 7;
const S_GAIN: usize = 8;
const S_DELAY: usize = 9;
const S_ENV: usize = 10;

fn zz(x: i64) -> u64 {
    ((x << 1) ^ (x >> 63)) as u64
}

fn unzz(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

fn put_uvarint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            break;
        }
        out.push(byte | 0x80);
    }
}

fn read_uvarint(bytes: &[u8], pos: &mut usize) -> Result<u64> {
    let mut v = 0u64;
    let mut shift = 0u32;
    loop {
        let b = *bytes
            .get(*pos)
            .ok_or_else(|| Error::malformed("seed varint is truncated"))?;
        *pos += 1;
        v |= u64::from(b & 0x7F) << shift;
        if b & 0x80 == 0 {
            return Ok(v);
        }
        shift += 7;
        if shift >= 64 {
            return Err(Error::malformed("seed varint overflows"));
        }
    }
}

/// Encode one integer stream with the smallest existing metadata codec.
/// Codec ids: `0` raw varint, `1` recent-value cache, `2` baseline+delta.
fn encode_stream(values: &[u64]) -> Vec<u8> {
    let mut raw = Vec::new();
    for &v in values {
        put_uvarint(&mut raw, v);
    }
    let mut best: (u8, Vec<u8>) = (0, raw);

    let rep = encode_repcode(values);
    if decode_repcode(&rep).map(|d| d == values).unwrap_or(false) && rep.len() < best.1.len() {
        best = (1, rep);
    }
    let base = encode_baseline(values);
    if decode_baseline(&base).map(|d| d == values).unwrap_or(false) && base.len() < best.1.len() {
        best = (2, base);
    }
    let (codec, payload) = best;
    let mut out = Vec::with_capacity(payload.len() + 8);
    out.push(codec);
    put_uvarint(&mut out, values.len() as u64);
    out.extend_from_slice(&payload);
    out
}

fn decode_stream(bytes: &[u8], pos: &mut usize) -> Result<Vec<u64>> {
    let codec = *bytes
        .get(*pos)
        .ok_or_else(|| Error::malformed("seed stream is truncated"))?;
    *pos += 1;
    let count = read_uvarint(bytes, pos)? as usize;
    let values = match codec {
        0 => {
            let mut v = Vec::with_capacity(count.min(1 << 20));
            for _ in 0..count {
                v.push(read_uvarint(bytes, pos)?);
            }
            v
        }
        1 | 2 => {
            // The repcode/baseline payloads are self-terminating only through
            // their declared count, so decode from the remaining slice and then
            // advance the cursor by re-encoding the decoded values.
            let decoded = if codec == 1 {
                decode_repcode(&bytes[*pos..])?
            } else {
                decode_baseline(&bytes[*pos..])?
            };
            if decoded.len() != count {
                return Err(Error::malformed("seed stream count mismatch"));
            }
            let consumed = if codec == 1 {
                encode_repcode(&decoded).len()
            } else {
                encode_baseline(&decoded).len()
            };
            *pos += consumed;
            decoded
        }
        other => {
            return Err(Error::new(
                crate::error::Kind::Unsupported,
                format!("unknown seed stream codec {other}"),
            ));
        }
    };
    Ok(values)
}

/// Decompose a validated mono Compound graph into the eleven canonical streams.
fn build_streams(g: &CompoundGraph) -> Vec<Vec<u64>> {
    let mut s: Vec<Vec<u64>> = vec![Vec::new(); STREAM_COUNT];
    s[S_HEADER] = vec![
        u64::from(g.channels),
        g.frames,
        u64::from(g.sample_rate_hz),
        g.nodes.len() as u64,
    ];
    for (i, n) in g.nodes.iter().enumerate() {
        s[S_OPCODES].push(u64::from(n.op.tag()));
        s[S_CHILD_COUNTS].push(n.children.len() as u64);
        for &c in &n.children {
            // Children always reference earlier nodes, so the offset is >= 1 and
            // small for a topologically compact graph.
            s[S_CHILD_OFFSETS].push((i as u64) - u64::from(c));
        }
        match &n.op {
            CompoundOp::Silence | CompoundOp::Add => {}
            CompoundOp::Constant { level } => s[S_CONST_LEVEL].push(zz(i64::from(*level))),
            CompoundOp::Oscillator {
                freq_hz,
                amp_q16,
                phase0,
            } => {
                s[S_OSC_FREQ].push(u64::from(*freq_hz));
                s[S_OSC_AMP].push(zz(i64::from(*amp_q16)));
                s[S_OSC_PHASE].push(*phase0);
            }
            CompoundOp::Gain { q16 } => s[S_GAIN].push(zz(i64::from(*q16))),
            CompoundOp::Delay { frames } => s[S_DELAY].push(zz(*frames)),
            CompoundOp::Envelope {
                attack_frames,
                decay_frames,
                sustain_q16,
                release_frames,
                t_on,
                t_off,
            } => {
                s[S_ENV].push(u64::from(*attack_frames));
                s[S_ENV].push(u64::from(*decay_frames));
                s[S_ENV].push(zz(i64::from(*sustain_q16)));
                s[S_ENV].push(u64::from(*release_frames));
                s[S_ENV].push(zz(*t_on));
                match t_off {
                    Some(off) => {
                        s[S_ENV].push(1);
                        s[S_ENV].push(zz(*off));
                    }
                    None => s[S_ENV].push(0),
                }
            }
        }
    }
    s
}

fn take(stream: &[u64], cursor: &mut usize) -> Result<u64> {
    let v = *stream
        .get(*cursor)
        .ok_or_else(|| Error::malformed("seed H state stream is exhausted"))?;
    *cursor += 1;
    Ok(v)
}

fn rebuild_graph(s: &[Vec<u64>]) -> Result<CompoundGraph> {
    if s[S_HEADER].len() != 4 {
        return Err(Error::malformed("seed H state header is malformed"));
    }
    let channels = u8::try_from(s[S_HEADER][0])
        .map_err(|_| Error::malformed("seed H state channel count out of domain"))?;
    let frames = s[S_HEADER][1];
    let sample_rate_hz = u32::try_from(s[S_HEADER][2])
        .map_err(|_| Error::malformed("seed H state sample rate out of domain"))?;
    let node_count = s[S_HEADER][3] as usize;
    if node_count == 0 || node_count > crate::compound::MAX_COMPOUND_NODES as usize {
        return Err(Error::limit("seed H state node count out of range"));
    }
    if s[S_OPCODES].len() != node_count || s[S_CHILD_COUNTS].len() != node_count {
        return Err(Error::malformed("seed H state node streams disagree"));
    }
    let mut cursors = [0usize; STREAM_COUNT];
    cursors[S_HEADER] = 4;
    let mut nodes: Vec<CompoundNode> = Vec::with_capacity(node_count);
    for i in 0..node_count {
        let tag = take(&s[S_OPCODES], &mut cursors[S_OPCODES])?;
        let cc = take(&s[S_CHILD_COUNTS], &mut cursors[S_CHILD_COUNTS])? as usize;
        if cc > MAX_COMPOUND_ARITY as usize {
            return Err(Error::limit("seed H state arity exceeds the bound"));
        }
        let mut children = Vec::with_capacity(cc);
        for _ in 0..cc {
            let off = take(&s[S_CHILD_OFFSETS], &mut cursors[S_CHILD_OFFSETS])?;
            if off == 0 || off > i as u64 {
                return Err(Error::malformed("seed H state child reference is invalid"));
            }
            children.push((i as u64 - off) as u16);
        }
        let op = match tag {
            0 => CompoundOp::Silence,
            1 => CompoundOp::Constant {
                level: unzz(take(&s[S_CONST_LEVEL], &mut cursors[S_CONST_LEVEL])?) as i32,
            },
            2 => {
                let freq_hz = u32::try_from(take(&s[S_OSC_FREQ], &mut cursors[S_OSC_FREQ])?)
                    .map_err(|_| Error::malformed("seed oscillator frequency out of domain"))?;
                let amp_q16 = unzz(take(&s[S_OSC_AMP], &mut cursors[S_OSC_AMP])?);
                let phase0 = take(&s[S_OSC_PHASE], &mut cursors[S_OSC_PHASE])?;
                CompoundOp::Oscillator {
                    freq_hz,
                    amp_q16: i32::try_from(amp_q16)
                        .map_err(|_| Error::malformed("seed oscillator amplitude out of domain"))?,
                    phase0,
                }
            }
            3 => CompoundOp::Gain {
                q16: i32::try_from(unzz(take(&s[S_GAIN], &mut cursors[S_GAIN])?))
                    .map_err(|_| Error::malformed("seed gain out of domain"))?,
            },
            4 => CompoundOp::Delay {
                frames: unzz(take(&s[S_DELAY], &mut cursors[S_DELAY])?),
            },
            5 => CompoundOp::Add,
            6 => {
                let attack_frames = u32::try_from(take(&s[S_ENV], &mut cursors[S_ENV])?)
                    .map_err(|_| Error::malformed("seed envelope attack out of domain"))?;
                let decay_frames = u32::try_from(take(&s[S_ENV], &mut cursors[S_ENV])?)
                    .map_err(|_| Error::malformed("seed envelope decay out of domain"))?;
                let sustain_q16 = i32::try_from(unzz(take(&s[S_ENV], &mut cursors[S_ENV])?))
                    .map_err(|_| Error::malformed("seed envelope sustain out of domain"))?;
                let release_frames = u32::try_from(take(&s[S_ENV], &mut cursors[S_ENV])?)
                    .map_err(|_| Error::malformed("seed envelope release out of domain"))?;
                let t_on = unzz(take(&s[S_ENV], &mut cursors[S_ENV])?);
                let has_off = take(&s[S_ENV], &mut cursors[S_ENV])?;
                let t_off = match has_off {
                    0 => None,
                    1 => Some(unzz(take(&s[S_ENV], &mut cursors[S_ENV])?)),
                    _ => return Err(Error::malformed("seed envelope off flag is invalid")),
                };
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
                    format!("unknown seed opcode {other}"),
                ));
            }
        };
        nodes.push(CompoundNode { op, children });
    }
    // Every stream must be exactly consumed: a seed is a canonical encoding, so
    // trailing slack is malformed rather than ignored.
    if cursors
        .iter()
        .enumerate()
        .any(|(i, &c)| i != S_HEADER && c != s[i].len())
    {
        return Err(Error::malformed("seed H state has trailing stream values"));
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

/// Encode a validated mono Compound graph as an entropy-coded H state.
pub fn encode_h_state(g: &CompoundGraph) -> Result<Vec<u8>> {
    g.validate()?;
    let streams = build_streams(g);
    let mut out = Vec::new();
    out.push(SEED_VERSION);
    out.extend_from_slice(&(STREAM_COUNT as u16).to_le_bytes());
    for stream in &streams {
        let enc = encode_stream(stream);
        out.extend_from_slice(&(enc.len() as u32).to_le_bytes());
        out.extend_from_slice(&enc);
    }
    Ok(out)
}

/// Decode an entropy-coded H state back to its exact graph.
pub fn decode_h_state(bytes: &[u8]) -> Result<CompoundGraph> {
    let mut pos = 0usize;
    let version = *bytes
        .first()
        .ok_or_else(|| Error::malformed("seed H state is empty"))?;
    if version != SEED_VERSION {
        return Err(Error::new(
            crate::error::Kind::Unsupported,
            format!("unsupported seed H state version {version}"),
        ));
    }
    pos += 1;
    let declared = u16::from_le_bytes(
        bytes
            .get(pos..pos + 2)
            .ok_or_else(|| Error::malformed("seed H state framing is truncated"))?
            .try_into()
            .unwrap(),
    );
    pos += 2;
    if declared as usize != STREAM_COUNT {
        return Err(Error::malformed("seed H state stream count mismatch"));
    }
    let mut streams: Vec<Vec<u64>> = Vec::with_capacity(STREAM_COUNT);
    for _ in 0..STREAM_COUNT {
        let len = u32::from_le_bytes(
            bytes
                .get(pos..pos + 4)
                .ok_or_else(|| Error::malformed("seed H state framing is truncated"))?
                .try_into()
                .unwrap(),
        ) as usize;
        pos += 4;
        let end = pos
            .checked_add(len)
            .ok_or_else(|| Error::limit("seed H state length overflows"))?;
        let chunk = bytes
            .get(pos..end)
            .ok_or_else(|| Error::malformed("seed H state stream is truncated"))?;
        pos = end;
        let mut p = 0usize;
        streams.push(decode_stream(chunk, &mut p)?);
        if p != chunk.len() {
            return Err(Error::malformed("seed H state stream has trailing bytes"));
        }
    }
    if pos != bytes.len() {
        return Err(Error::malformed("seed H state has trailing bytes"));
    }
    rebuild_graph(&streams)
}

/// A deterministic explanation `H` plus its entropy-coded exact residual `R`.
#[derive(Debug, Clone)]
pub struct SeedObject {
    graph: CompoundGraph,
    h_state: Vec<u8>,
    residual: ResidualEncodingV2,
    residual_len: usize,
}

impl SeedObject {
    /// Build the canonical seed for `graph` closing `target` exactly, or `None`
    /// when the residual is not representable as an i32 delta sequence (the
    /// candidate is dropped, never approximated).
    pub fn from_graph_target(graph: CompoundGraph, target: &[i32]) -> Result<Option<SeedObject>> {
        let Some(residual) = residual_only(&graph, target)? else {
            return Ok(None);
        };
        Self::from_graph_residual(graph, target.len(), residual)
    }

    /// Build a seed from an already-computed exact residual (no re-materialize).
    pub fn from_graph_residual(
        graph: CompoundGraph,
        target_len: usize,
        residual: Vec<i32>,
    ) -> Result<Option<SeedObject>> {
        if graph.channels != 1 || residual.len() != target_len {
            return Ok(None);
        }
        let residual_enc = encode_best_seed(&residual);
        let h_state = encode_h_state(&graph)?;
        Ok(Some(SeedObject {
            graph,
            h_state,
            residual: residual_enc,
            residual_len: residual.len(),
        }))
    }

    /// The deterministic explanation `H`.
    pub fn graph(&self) -> &CompoundGraph {
        &self.graph
    }

    /// Decode `H` from its entropy-coded state (the decoder-side view).
    pub fn decoded_graph(&self) -> Result<CompoundGraph> {
        decode_h_state(&self.h_state)
    }

    /// Entropy-coded H state bytes.
    pub fn h_bytes(&self) -> u64 {
        self.h_state.len() as u64
    }

    /// Entropy-coded residual bytes (including the residual codec id byte).
    pub fn residual_bytes(&self) -> u64 {
        self.residual.complete_bytes()
    }

    /// The residual codec actually selected.
    pub fn residual_codec(&self) -> &'static str {
        residual_codec_name(self.residual.codec)
    }

    /// Complete physical bytes: one version byte, the H state, then R.
    pub fn complete_bytes(&self) -> u64 {
        1 + self.h_state.len() as u64 + self.residual.complete_bytes()
    }

    /// Decode `H` from its entropy-coded state and reconstruct `X = H + R`
    /// exactly.
    pub fn materialize(&self) -> Result<Vec<i32>> {
        let graph = self.decoded_graph()?;
        let base = graph.materialize()?;
        let residual = self.residual.decode(self.residual_len)?;
        if base.len() != residual.len() {
            return Err(Error::malformed("seed residual length disagrees with H"));
        }
        let mut out = Vec::with_capacity(base.len());
        for (&b, &r) in base.iter().zip(residual.iter()) {
            out.push(saturating_add(b, r));
        }
        Ok(out)
    }

    /// Exact closure against the canonical target.
    pub fn closes_to(&self, target: &[i32]) -> bool {
        self.materialize().map(|v| v == target).unwrap_or(false)
    }
}

fn saturating_add(a: i32, b: i32) -> i32 {
    let v = i64::from(a) + i64::from(b);
    v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

/// The exact `target - H` residual, or `None` when it is not representable as
/// an i32 delta sequence. This is the cheap ranking primitive: callers can order
/// candidate explanations by residual energy before paying for the full
/// residual-codec search.
///
/// The residual is *exact* by construction: `H` is materialized through the
/// deterministic evaluator and the difference is taken in i64 before narrowing.
pub fn residual_only(graph: &CompoundGraph, target: &[i32]) -> Result<Option<Vec<i32>>> {
    if graph.channels != 1 {
        return Ok(None);
    }
    let signal = graph.materialize()?;
    if signal.len() != target.len() {
        return Ok(None);
    }
    let mut residual = Vec::with_capacity(target.len());
    for (&t, &s) in target.iter().zip(signal.iter()) {
        let d = i64::from(t) - i64::from(s);
        match i32::try_from(d) {
            Ok(v) => residual.push(v),
            Err(_) => return Ok(None),
        }
    }
    Ok(Some(residual))
}

fn residual_codec_name(codec: crate::learned::residual_codec2::ResidualCodecV2) -> &'static str {
    // The canonical codec label is derived from the debug spelling; this avoids
    // adding a second name table.
    codec.name()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compound::{CompoundNode, CompoundOp};

    fn osc(freq: u32, amp: i32, phase0: u64) -> CompoundNode {
        CompoundNode {
            op: CompoundOp::Oscillator {
                freq_hz: freq,
                amp_q16: amp,
                phase0,
            },
            children: Vec::new(),
        }
    }

    fn sample_graph() -> CompoundGraph {
        CompoundGraph {
            channels: 1,
            frames: 2048,
            sample_rate_hz: 48_000,
            nodes: vec![
                osc(220, 62, 12_345),
                osc(440, 31, 0),
                CompoundNode {
                    op: CompoundOp::Add,
                    children: vec![0, 1],
                },
                CompoundNode {
                    op: CompoundOp::Gain { q16: 256 },
                    children: vec![2],
                },
                CompoundNode {
                    op: CompoundOp::Envelope {
                        attack_frames: 10,
                        decay_frames: 20,
                        sustain_q16: 1 << 15,
                        release_frames: 30,
                        t_on: 0,
                        t_off: Some(1_000),
                    },
                    children: vec![3],
                },
            ],
        }
    }

    #[test]
    fn h_state_round_trips_exactly() {
        let g = sample_graph();
        let bytes = encode_h_state(&g).unwrap();
        let back = decode_h_state(&bytes).unwrap();
        assert_eq!(back, g);
        assert!(bytes.len() < g.canonical_bytes().len());
    }

    #[test]
    fn h_state_is_deterministic() {
        let g = sample_graph();
        assert_eq!(encode_h_state(&g).unwrap(), encode_h_state(&g).unwrap());
    }

    #[test]
    fn seed_closes_exactly_on_its_own_graph() {
        let g = sample_graph();
        let target = g.materialize().unwrap();
        let seed = SeedObject::from_graph_target(g, &target).unwrap().unwrap();
        assert!(seed.closes_to(&target));
        assert_eq!(seed.materialize().unwrap(), target);
        // A self-closing seed has a tiny (near-free) residual.
        assert!(seed.residual_bytes() < 16);
    }

    #[test]
    fn seed_absorbs_nonzero_residual_exactly() {
        let g = sample_graph();
        let mut target = g.materialize().unwrap();
        for (i, v) in target.iter_mut().enumerate() {
            *v = v.wrapping_add((i as i32 % 7) - 3);
        }
        let seed = SeedObject::from_graph_target(g, &target).unwrap().unwrap();
        assert!(seed.closes_to(&target));
        assert!(seed.residual_bytes() > 1);
    }

    #[test]
    fn malformed_h_state_is_rejected() {
        let g = sample_graph();
        let mut bytes = encode_h_state(&g).unwrap();
        bytes[0] = 9;
        assert!(decode_h_state(&bytes).is_err());
        let mut bytes = encode_h_state(&g).unwrap();
        let n = bytes.len();
        bytes[n - 1] ^= 0xFF;
        assert!(decode_h_state(&bytes).is_err());
    }
}
