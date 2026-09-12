//! Stateful syntax parsing (Phase 6, mechanism 1; experimental profile Exp3).
//!
//! A signal that alternates between a small number of regimes punishes a plain
//! segmented representation: every segment repays the full canonical bytes of
//! its model, even when an identical model was already used a segment earlier.
//!
//! This module makes the **model syntax itself stateful**. A decoder-synchronized
//! carousel holds the three most recently used model tuples in move-to-front
//! order. Each segment is coded as a two-bit symbol
//!
//! ```text
//! 00  reuse carousel slot 0 (MRU)
//! 01  reuse carousel slot 1
//! 10  reuse carousel slot 2
//! 11  new tuple: a varint length and the canonical model bytes follow
//! ```
//!
//! followed by a move-to-front update. A reused tuple therefore costs two bits
//! instead of a full model payload.
//!
//! The encoding decision is **path dependent**: choosing tuple `A` now changes
//! the price of the same tuple several segments later, because it reorders the
//! carousel. The parser is therefore a shortest-path search whose state is
//! `(position, carousel)` — exactly the mechanism the fourth-pass report calls
//! `StatefulSyntaxParse`. The search keeps a deterministic bounded beam; the
//! unit tests measure the beam width against exhaustive enumeration.
//!
//! This is **not** a mixture: each segment reconstructs through exactly one
//! model, and the object is a plain concatenation of independently
//! materializable segments (the same exact-closure contract as
//! [`crate::learned::segmented`]).
//!
//! ```text
//! X_hat[t] = sat_i32(H_{Theta_{s(t)}}(X_hat history)[t] + R[t])   and   X_hat == X
//! ```

use crate::error::{Error, Result};
use crate::learned::model::LearnedModel;
use std::collections::HashMap;

/// Carousel capacity: slots `MRU[0..=2]`, with symbol `3` meaning "new tuple".
pub const CAROUSEL_SLOTS: usize = 3;

/// Hard ceiling on the number of boundary nodes one parse may visit. A grid
/// whose finest rung produces more nodes than this is rejected rather than
/// explored, so a hostile or accidental grid cannot allocate unbounded state.
pub const MAX_PARSE_NODES: usize = 8192;

/// Hard ceiling on the number of retained `(position, carousel)` states per
/// boundary. The search is an exact shortest path after dominance deduplication
/// (a cheaper path to the same carousel dominates a dearer one), so this
/// ceiling is a denial-of-service guard, not part of the mechanism; it is never
/// reached by the frozen fixtures.
pub const MAX_PARSE_STATES_PER_NODE: usize = 4096;

/// Hard ceiling on the model alternatives considered for one region.
pub const MAX_REGION_OPTIONS: usize = 256;

/// The symbol that introduces a fresh tuple payload.
pub const NEW_TUPLE_SYMBOL: u8 = 3;

/// One segment: a frame count and the model tuple that explains it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxSegment {
    /// Frames covered by this segment (`>= 1`).
    pub frames: u32,
    /// The model tuple for this segment (never itself a syntax/segment wrapper).
    pub model: Box<LearnedModel>,
}

/// A sequence of model tuples coded with a move-to-front carousel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatefulSyntaxModel {
    pub channels: u8,
    pub segments: Vec<SyntaxSegment>,
}

impl StatefulSyntaxModel {
    /// Total frames across all segments.
    pub fn frames(&self) -> u64 {
        self.segments.iter().map(|s| u64::from(s.frames)).sum()
    }

    /// Validate geometry, nesting and the complexity ceilings.
    pub fn validate(&self) -> Result<()> {
        let c = usize::from(self.channels);
        if c == 0 || c > crate::limits::MAX_CHANNELS as usize {
            return Err(Error::malformed(
                "stateful syntax channel count out of range",
            ));
        }
        if self.segments.is_empty() {
            return Err(Error::malformed("stateful syntax model has no segments"));
        }
        if self.segments.len() as u32 > crate::limits::MAX_LEARNED_SEGMENTS {
            return Err(Error::limit("stateful syntax has too many segments"));
        }
        for s in &self.segments {
            if s.frames == 0 {
                return Err(Error::malformed("stateful syntax has an empty segment"));
            }
            // A segment tuple may itself be a segmented model (the speech
            // portfolio's forward/reverse direction choice is one), because a
            // segment's hypothesis is always evaluated block-locally. Only
            // self-nesting is refused, to keep the wrapper tree finite and the
            // complexity bounds obviously computable.
            if matches!(&*s.model, LearnedModel::StatefulSyntax(_)) {
                return Err(Error::malformed(
                    "stateful syntax segments may not nest another stateful syntax model",
                ));
            }
            s.model.validate()?;
            if s.model.channels() != self.channels {
                return Err(Error::malformed(
                    "segment channel count disagrees with the stateful syntax model",
                ));
            }
        }
        Ok(())
    }

    /// Global frame index to `(segment_index, local_frame)`.
    pub fn locate(&self, global_frame: usize) -> Result<(usize, usize)> {
        let mut remaining = global_frame;
        for (i, s) in self.segments.iter().enumerate() {
            let f = s.frames as usize;
            if remaining < f {
                return Ok((i, remaining));
            }
            remaining -= f;
        }
        Err(Error::malformed(
            "stateful syntax frame index exceeds the extent",
        ))
    }

    /// Declared receptive field (maximum over segments).
    pub fn receptive_field(&self) -> u64 {
        self.segments
            .iter()
            .map(|s| s.model.receptive_field())
            .max()
            .unwrap_or(0)
    }

    /// Abstract operations per output sample (maximum over segments).
    pub fn ops_per_sample(&self) -> u64 {
        self.segments
            .iter()
            .map(|s| s.model.ops_per_sample())
            .max()
            .unwrap_or(0)
    }

    /// Persistent state bytes (sum over segments).
    pub fn state_bytes(&self) -> u64 {
        self.segments.iter().map(|s| s.model.state_bytes()).sum()
    }

    /// Declared checkpoints (sum over segments).
    pub fn checkpoint_count(&self) -> u32 {
        self.segments
            .iter()
            .map(|s| s.model.checkpoint_count())
            .sum()
    }

    /// Frames replayed to serve a seek: the local offset inside the segment.
    pub fn replay_frames(&self, start: usize) -> usize {
        match self.locate(start) {
            Ok((_, local)) => local,
            Err(_) => start,
        }
    }

    /// Canonical model bytes with the move-to-front syntax applied:
    /// `kind(19) || channels || count(u32) || frames[u32] || symbols(2-bit packed)
    ///  || payload(varint length + model bytes per fresh tuple)`.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let count = self.segments.len();
        let mut out = Vec::new();
        out.push(19); // model kind 19 = stateful syntax
        out.push(self.channels);
        out.extend_from_slice(&(count as u32).to_le_bytes());
        for s in &self.segments {
            out.extend_from_slice(&s.frames.to_le_bytes());
        }
        let sym_len = count.div_ceil(4);

        // Carousel pass: assign each segment a symbol and accumulate the payload
        // of every fresh tuple in first-use order.
        let mut vocab: Vec<Vec<u8>> = Vec::new();
        let mut car: Vec<usize> = Vec::with_capacity(CAROUSEL_SLOTS);
        let mut symbols = vec![0u8; count];
        let mut payload: Vec<u8> = Vec::new();
        for (i, s) in self.segments.iter().enumerate() {
            let bytes = s.model.canonical_bytes();
            if let Some(k) = car.iter().position(|&id| vocab[id] == bytes) {
                symbols[i] = k as u8;
                let id = car.remove(k);
                car.insert(0, id);
            } else {
                symbols[i] = NEW_TUPLE_SYMBOL;
                put_uvarint(&mut payload, bytes.len() as u64);
                payload.extend_from_slice(&bytes);
                let id = vocab.len();
                vocab.push(bytes);
                car.insert(0, id);
                car.truncate(CAROUSEL_SLOTS);
            }
        }
        let mut packed = vec![0u8; sym_len];
        for (i, &s) in symbols.iter().enumerate() {
            packed[i / 4] |= (s & 3) << ((i % 4) * 2);
        }
        out.extend_from_slice(&packed);
        out.extend_from_slice(&payload);
        out
    }

    /// Parse canonical bytes written by [`StatefulSyntaxModel::canonical_bytes`].
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<StatefulSyntaxModel> {
        if bytes.len() < 6 || bytes[0] != 19 {
            return Err(Error::malformed("stateful syntax model header mismatch"));
        }
        let channels = bytes[1];
        let count = u32::from_le_bytes(bytes[2..6].try_into().unwrap());
        if count == 0 || count > crate::limits::MAX_LEARNED_SEGMENTS {
            return Err(Error::limit(
                "stateful syntax segment count exceeds the bound",
            ));
        }
        let count = count as usize;
        let mut at = 6usize;
        let frames_len = count
            .checked_mul(4)
            .ok_or_else(|| Error::limit("stateful syntax frame table overflows"))?;
        let frames_bytes = bytes
            .get(at..at + frames_len)
            .ok_or_else(|| Error::malformed("stateful syntax model is truncated"))?;
        at += frames_len;
        let frames: Vec<u32> = frames_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        let sym_len = count.div_ceil(4);
        let packed = bytes
            .get(at..at + sym_len)
            .ok_or_else(|| Error::malformed("stateful syntax model is truncated"))?
            .to_vec();
        at += sym_len;

        // Replay the carousel exactly as the encoder did.
        let mut vocab: Vec<LearnedModel> = Vec::new();
        let mut car: Vec<usize> = Vec::with_capacity(CAROUSEL_SLOTS);
        let mut segments = Vec::with_capacity(count);
        for i in 0..count {
            let symbol = (packed[i / 4] >> ((i % 4) * 2)) & 3;
            let model = if symbol == NEW_TUPLE_SYMBOL {
                let (len, used) = get_uvarint(bytes, at)?;
                at += used;
                let len = usize::try_from(len)
                    .map_err(|_| Error::limit("stateful syntax tuple length exceeds usize"))?;
                if len > crate::limits::MAX_LEARNED_WEIGHT_BYTES.saturating_mul(4) as usize {
                    return Err(Error::limit(
                        "stateful syntax tuple length exceeds the bound",
                    ));
                }
                let mb = bytes
                    .get(at..at + len)
                    .ok_or_else(|| Error::malformed("stateful syntax model is truncated"))?;
                at += len;
                let m = LearnedModel::from_canonical_bytes(mb)?;
                vocab.push(m);
                let id = vocab.len() - 1;
                car.insert(0, id);
                car.truncate(CAROUSEL_SLOTS);
                vocab[id].clone()
            } else {
                let slot = symbol as usize;
                let id = *car.get(slot).ok_or_else(|| {
                    Error::malformed("stateful syntax references an empty carousel slot")
                })?;
                let m = vocab[id].clone();
                car.remove(slot);
                car.insert(0, id);
                m
            };
            segments.push(SyntaxSegment {
                frames: frames[i],
                model: Box::new(model),
            });
        }
        if at != bytes.len() {
            return Err(Error::malformed("stateful syntax model has trailing bytes"));
        }
        let m = StatefulSyntaxModel { channels, segments };
        m.validate()?;
        Ok(m)
    }

    /// Compute the hypothesis over the whole extent, open-loop from a source.
    /// `source` is the concatenated source (segment-major).
    pub fn hypothesis_from_source(&self, source: &[i32], frames: usize) -> Result<Vec<i32>> {
        self.validate()?;
        let c = usize::from(self.channels);
        if source.len() != frames * c {
            return Err(Error::malformed("stateful syntax source length mismatch"));
        }
        if self.frames() != frames as u64 {
            return Err(Error::malformed(
                "stateful syntax frame count disagrees with the extent",
            ));
        }
        let mut out = Vec::with_capacity(frames * c);
        let mut base = 0usize;
        for s in &self.segments {
            let f = s.frames as usize;
            let h = s
                .model
                .hypothesis_from_source(&source[base * c..(base + f) * c], f)?;
            out.extend_from_slice(&h);
            base += f;
        }
        Ok(out)
    }

    /// Reconstruct `[start, start+len)` exactly from the concatenated residual.
    pub fn evaluate_range(
        &self,
        residual: &[i32],
        frames: usize,
        start: usize,
        len: usize,
    ) -> Result<Vec<i32>> {
        self.validate()?;
        let c = usize::from(self.channels);
        if residual.len() != frames * c {
            return Err(Error::malformed("stateful syntax residual length mismatch"));
        }
        if self.frames() != frames as u64 {
            return Err(Error::malformed(
                "stateful syntax frame count disagrees with the extent",
            ));
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::limit("stateful syntax range overflows"))?;
        if end > frames {
            return Err(Error::malformed("stateful syntax range exceeds the extent"));
        }
        let mut out = vec![0i32; len * c];
        if len == 0 {
            return Ok(out);
        }
        let mut base = 0usize;
        for s in &self.segments {
            let f = s.frames as usize;
            let seg_start = base;
            let seg_end = base + f;
            let lo = start.max(seg_start);
            let hi = end.min(seg_end);
            if lo < hi {
                let local_start = lo - seg_start;
                let local_len = hi - lo;
                let seg_residual = &residual[seg_start * c..seg_end * c];
                let vals = s
                    .model
                    .evaluate_range(seg_residual, f, local_start, local_len)?;
                out[(lo - start) * c..(hi - start) * c].copy_from_slice(&vals);
            }
            base = seg_end;
            if base >= end {
                break;
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Small varint helpers, shared by the syntax coder and the parser's price
// model. They are deliberately local: the syntax payload is the only consumer.
// ---------------------------------------------------------------------------

fn put_uvarint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Number of bytes `put_uvarint(v)` writes.
pub fn uvarint_len(mut v: u64) -> u64 {
    let mut n = 1u64;
    while v >= 0x80 {
        v >>= 7;
        n += 1;
    }
    n
}

/// Read a varint at `at`, returning `(value, bytes_consumed)`.
fn get_uvarint(bytes: &[u8], at: usize) -> Result<(u64, usize)> {
    let mut v = 0u64;
    let mut shift = 0u32;
    let mut used = 0usize;
    loop {
        let b = *bytes
            .get(at + used)
            .ok_or_else(|| Error::malformed("stateful syntax varint is truncated"))?;
        used += 1;
        if shift >= 64 {
            return Err(Error::limit("stateful syntax varint overflows"));
        }
        v |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return Ok((v, used));
        }
        shift += 7;
    }
}

// ---------------------------------------------------------------------------
// Region price model
// ---------------------------------------------------------------------------

/// A region alternative handed to the syntax parser: exactly one model plus the
/// price of its residual over that region.
#[derive(Debug, Clone)]
pub struct RegionOption {
    pub model: LearnedModel,
    /// Regions covered (`b - a`), for caller convenience.
    pub frames: u32,
    /// Code-length proxy of the residual under this model, in bits.
    pub residual_bits: u64,
}

/// Exact residual of `source` under `model` when the model is evaluated
/// block-locally over the region (`source.len() == frames`).
pub fn residual_under(model: &LearnedModel, source: &[i32], frames: usize) -> Result<Vec<i32>> {
    if source.len() != frames {
        return Err(Error::malformed("region source length mismatch"));
    }
    let h = model.hypothesis_from_source(source, frames)?;
    if h.len() != frames {
        return Err(Error::malformed("region hypothesis length mismatch"));
    }
    let mut r = vec![0i32; frames];
    for i in 0..frames {
        let d = i64::from(source[i]) - i64::from(h[i]);
        if d < i64::from(i32::MIN) || d > i64::from(i32::MAX) {
            return Err(Error::new(
                crate::error::Kind::Integrity,
                "region residual does not fit i32",
            ));
        }
        r[i] = d as i32;
    }
    Ok(r)
}

/// Build a [`RegionOption`] from an already-evaluated residual.
pub fn option_from_residual(model: LearnedModel, residual: &[i32]) -> RegionOption {
    RegionOption {
        model,
        frames: residual.len() as u32,
        residual_bits: proxy_residual_bits(residual),
    }
}

/// A deterministic, fast code-length proxy for a residual slice.
///
/// Planning must not pay the full 24-codec search at every grid edge; the
/// assembled object is measured exactly afterwards. The proxy is the smaller of
/// the canonical Exp-Golomb length and the best Rice length, plus a sign bit per
/// sample. It is an honest integer code length, never a variance.
pub fn proxy_residual_bits(residual: &[i32]) -> u64 {
    let mut eg: u64 = residual.len() as u64; // one sign/zero bit per sample
    for &r in residual {
        eg = eg.saturating_add(eg0_bits(u64::from(r.unsigned_abs())));
    }
    rice_bits(residual).min(eg)
}

/// Exp-Golomb order-0 bits for the value `v` (the frozen binarization uses
/// `v = |r|`; a zero residual codes as one bit here plus the sign bit).
fn eg0_bits(v: u64) -> u64 {
    let x = v + 1;
    let bl = 64 - x.leading_zeros() as u64;
    2 * bl - 1
}

/// Best Rice-`k` length (including a 5-bit `k` header) over a bounded `k` ladder.
fn rice_bits(residual: &[i32]) -> u64 {
    let mut best = u64::MAX;
    for k in 0..=24u32 {
        let mut total = 5u64; // k header
        for &r in residual {
            let v = u64::from(r.unsigned_abs());
            total = total.saturating_add((v >> k) + 1 + u64::from(k));
            if total >= best {
                break;
            }
        }
        if total < best {
            best = total;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// The path-dependent parser
// ---------------------------------------------------------------------------

/// The chosen parse: ordered regions, their models, and the carousel statistics.
#[derive(Debug, Clone)]
pub struct ParseOutcome {
    /// `(start, end)` frame bounds of each chosen region, in order.
    pub regions: Vec<(usize, usize)>,
    /// The model chosen for each region (parallel to `regions`).
    pub models: Vec<LearnedModel>,
    /// Proxy cost of the chosen parse, in bits.
    pub cost_bits: u64,
    /// Segments whose tuple was a carousel hit (symbols 0..=2).
    pub mru_hits: u64,
    /// Segments that transmitted a fresh tuple (symbol 3).
    pub new_tuples: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct State {
    cost_bits: u64,
    segments: u32,
    car: [u64; CAROUSEL_SLOTS],
    via_edge: usize,
    via_opt: usize,
    back_node: usize,
    back_state: usize,
    has_back: bool,
}

#[derive(Clone)]
struct EdgeOption {
    model: LearnedModel,
    residual_bits: u64,
    tuple_id: u64,
    new_bits: u64,
}

struct Edge {
    a: usize,
    b: usize,
    ai: usize,
    bi: usize,
    opts: Vec<EdgeOption>,
}

/// Shortest-path syntax parse over a bounded boundary grid.
///
/// `options(a, b)` returns the model alternatives for the region `[a, b)`,
/// already priced with [`proxy_residual_bits`]. `beam` bounds the number of
/// `(position, carousel)` states kept per boundary; pass `usize::MAX` for
/// exhaustive enumeration on small fixtures. Returns `None` when no parse
/// reaches the end.
pub fn parse_stateful<F>(
    frames: usize,
    grid: &[usize],
    beam: usize,
    mut options: F,
) -> Option<ParseOutcome>
where
    F: FnMut(usize, usize) -> Vec<RegionOption>,
{
    if frames == 0 || grid.is_empty() {
        return None;
    }
    let unit = grid.iter().copied().min().unwrap_or(1).max(1);
    let max_rung = grid.iter().copied().max().unwrap_or(unit).max(unit);

    // Boundary nodes: every grid multiple strictly below `frames`, then `frames`.
    // The loop is bounded by `MAX_PARSE_NODES` so a pathological grid cannot
    // allocate an unbounded position table before the guard below fires.
    let mut pos: Vec<usize> = Vec::new();
    let mut p = 0usize;
    while p < frames && pos.len() <= MAX_PARSE_NODES {
        pos.push(p);
        p += unit;
    }
    if pos.len() > MAX_PARSE_NODES {
        return None;
    }
    if pos.last().copied() != Some(frames) {
        pos.push(frames);
    }
    let nodes = pos.len();
    if !(2..=MAX_PARSE_NODES).contains(&nodes) {
        return None;
    }

    // Enumerate edges and intern tuples once, so the DP only compares integers.
    let mut vocab: Vec<Vec<u8>> = Vec::new();
    let mut vocab_ids: HashMap<Vec<u8>, u64> = HashMap::new();
    let mut edges: Vec<Edge> = Vec::new();
    for i in 0..nodes {
        for j in (i + 1)..nodes {
            let len = pos[j] - pos[i];
            // Grid rungs, plus one unrestricted tail into the terminal node.
            if len > max_rung && j != nodes - 1 {
                continue;
            }
            let region = options(pos[i], pos[j]);
            if region.is_empty() {
                continue;
            }
            let mut opts = Vec::with_capacity(region.len());
            for o in region.into_iter().take(MAX_REGION_OPTIONS) {
                let bytes = o.model.canonical_bytes();
                let len = bytes.len() as u64;
                let tuple_id = match vocab_ids.get(&bytes) {
                    Some(&id) => id,
                    None => {
                        let id = vocab.len() as u64;
                        vocab_ids.insert(bytes.clone(), id);
                        vocab.push(bytes);
                        id
                    }
                };
                let payload = uvarint_len(len) + len;
                opts.push(EdgeOption {
                    model: o.model,
                    residual_bits: o.residual_bits,
                    tuple_id,
                    new_bits: payload.saturating_mul(8),
                });
            }
            edges.push(Edge {
                a: pos[i],
                b: pos[j],
                ai: i,
                bi: j,
                opts,
            });
        }
    }
    if edges.is_empty() {
        return None;
    }
    // Edges grouped by source node.
    let mut by_src: Vec<Vec<usize>> = vec![Vec::new(); nodes];
    for (e, edge) in edges.iter().enumerate() {
        by_src[edge.ai].push(e);
    }

    let empty_car = [u64::MAX; CAROUSEL_SLOTS];
    let mut states: Vec<Vec<State>> = vec![Vec::new(); nodes];
    states[0].push(State {
        cost_bits: 0,
        segments: 0,
        car: empty_car,
        via_edge: 0,
        via_opt: 0,
        back_node: 0,
        back_state: 0,
        has_back: false,
    });

    for i in 0..nodes {
        prune_states(&mut states[i], beam);
        if states[i].is_empty() {
            continue;
        }
        let current = states[i].clone();
        let mut touched: Vec<usize> = Vec::new();
        for (si, st) in current.iter().enumerate() {
            for &e in &by_src[i] {
                let edge = &edges[e];
                let j = edge.bi;
                if !touched.contains(&j) {
                    touched.push(j);
                }
                for (oi, opt) in edge.opts.iter().enumerate() {
                    let (add, car) = price_symbol(st.car, opt);
                    states[j].push(State {
                        cost_bits: st.cost_bits.saturating_add(add),
                        segments: st.segments.saturating_add(1),
                        car,
                        via_edge: e,
                        via_opt: oi,
                        back_node: i,
                        back_state: si,
                        has_back: true,
                    });
                }
            }
        }
        // Exact dominance deduplication after every source expansion keeps the
        // intermediate state pools bounded (a cheaper path to the same carousel
        // dominates a dearer one), so an unbounded path enumeration can never
        // allocate without limit.
        for j in touched {
            dedupe_states(&mut states[j]);
        }
    }

    prune_states(&mut states[nodes - 1], beam);
    let terminal = states[nodes - 1]
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| state_order(a, b))
        .map(|(idx, _)| idx)?;

    // Walk the back-pointers.
    let mut regions = Vec::new();
    let mut models = Vec::new();
    let mut mru_hits = 0u64;
    let mut new_tuples = 0u64;
    let mut node = nodes - 1;
    let mut idx = terminal;
    loop {
        let st = states[node][idx];
        if !st.has_back {
            break;
        }
        let edge = &edges[st.via_edge];
        let opt = &edge.opts[st.via_opt];
        regions.push((edge.a, edge.b));
        models.push(opt.model.clone());
        let parent = states[st.back_node][st.back_state];
        if parent.car.contains(&opt.tuple_id) {
            mru_hits += 1;
        } else {
            new_tuples += 1;
        }
        node = st.back_node;
        idx = st.back_state;
    }
    regions.reverse();
    models.reverse();
    let cost_bits = states[nodes - 1][terminal].cost_bits;
    Some(ParseOutcome {
        regions,
        models,
        cost_bits,
        mru_hits,
        new_tuples,
    })
}

/// Add the price of coding `opt` and return the resulting carousel.
fn price_symbol(car: [u64; CAROUSEL_SLOTS], opt: &EdgeOption) -> (u64, [u64; CAROUSEL_SLOTS]) {
    let sym_bits = 2u64;
    let mut next = car;
    if let Some(k) = car.iter().position(|&id| id == opt.tuple_id) {
        // Move to front.
        for m in (1..=k).rev() {
            next[m] = car[m - 1];
        }
        next[0] = opt.tuple_id;
        (opt.residual_bits.saturating_add(sym_bits), next)
    } else {
        for m in (1..CAROUSEL_SLOTS).rev() {
            next[m] = car[m - 1];
        }
        next[0] = opt.tuple_id;
        (
            opt.residual_bits
                .saturating_add(sym_bits)
                .saturating_add(opt.new_bits),
            next,
        )
    }
}

/// Deterministic state order: cheaper first, then fewer segments, then the
/// carousel tuple ids, then the back-pointer.
fn state_order(a: &State, b: &State) -> std::cmp::Ordering {
    a.cost_bits
        .cmp(&b.cost_bits)
        .then(a.segments.cmp(&b.segments))
        .then(a.car.cmp(&b.car))
        .then(a.via_edge.cmp(&b.via_edge))
        .then(a.via_opt.cmp(&b.via_opt))
}

fn prune_states(states: &mut Vec<State>, beam: usize) {
    dedupe_states(states);
    let cap = beam.clamp(1, MAX_PARSE_STATES_PER_NODE);
    if states.len() > cap {
        states.truncate(cap);
    }
}

/// Keep the cheapest state per carousel. Needs no prior sort: first occurrence
/// after a full ordering by `state_order` is the cheapest, and equal carousels
/// may be non-adjacent, so a set is used rather than `dedup_by`.
fn dedupe_states(states: &mut Vec<State>) {
    if states.len() < 2 {
        return;
    }
    states.sort_by(state_order);
    let mut seen: std::collections::HashSet<[u64; CAROUSEL_SLOTS]> =
        std::collections::HashSet::with_capacity(states.len());
    states.retain(|s| seen.insert(s.car));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::arithmetic::quantize_weight;
    use crate::learned::finite_field::LinearPredictor;
    use crate::learned::object::LearnedObject;

    fn ar1(channels: u8, weight: f64) -> LearnedModel {
        LearnedModel::Linear(LinearPredictor {
            channels,
            taps: 1,
            weights: vec![quantize_weight(weight)],
            bias: vec![0],
            block_frames: None,
        })
    }

    fn lcg(state: &mut u64) -> i32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 33) & 0xffff) as i32 - 32768
    }

    fn signal(n: usize, weight: f64, seed: u64) -> Vec<i32> {
        let mut s = seed | 1;
        let mut x = 0i64;
        let w = (weight * 4096.0) as i64;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let noise = i64::from(lcg(&mut s)) >> 2;
            x = (w * x) / 4096 + noise;
            x = x.clamp(i64::from(i16::MIN) * 256, i64::from(i16::MAX) * 256);
            out.push(x as i32);
        }
        out
    }

    #[test]
    fn canonical_round_trip_replays_the_carousel() {
        let a = ar1(1, 0.9);
        let b = ar1(1, -0.5);
        let m = StatefulSyntaxModel {
            channels: 1,
            segments: vec![
                SyntaxSegment {
                    frames: 100,
                    model: Box::new(a.clone()),
                },
                SyntaxSegment {
                    frames: 100,
                    model: Box::new(b.clone()),
                },
                SyntaxSegment {
                    frames: 100,
                    model: Box::new(a.clone()),
                },
            ],
        };
        let bytes = m.canonical_bytes();
        let back = StatefulSyntaxModel::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn move_to_front_saves_payload_on_a_repeated_tuple() {
        let a = ar1(1, 0.9);
        let b = ar1(1, -0.5);
        let repeated = StatefulSyntaxModel {
            channels: 1,
            segments: vec![
                SyntaxSegment {
                    frames: 64,
                    model: Box::new(a.clone()),
                },
                SyntaxSegment {
                    frames: 64,
                    model: Box::new(b.clone()),
                },
                SyntaxSegment {
                    frames: 64,
                    model: Box::new(a.clone()),
                },
            ],
        };
        let distinct = StatefulSyntaxModel {
            channels: 1,
            segments: vec![
                SyntaxSegment {
                    frames: 64,
                    model: Box::new(a.clone()),
                },
                SyntaxSegment {
                    frames: 64,
                    model: Box::new(b.clone()),
                },
                SyntaxSegment {
                    frames: 64,
                    model: Box::new(ar1(1, 0.2)),
                },
            ],
        };
        assert!(repeated.canonical_bytes().len() < distinct.canonical_bytes().len());
    }

    /// The core path-dependence claim: exhaustive and bounded-beam parses agree
    /// on every fixture once the beam is wide enough.
    #[test]
    fn beam_matches_exhaustive_on_repeated_regimes() {
        let grid = [512usize, 256];
        let models = [ar1(1, 0.95), ar1(1, -0.6), ar1(1, 0.3), ar1(1, 0.0)];
        // Alternating regimes A B A B, then A B C A B C.
        let mut source = Vec::new();
        source.extend(signal(512, 0.95, 11));
        source.extend(signal(512, -0.6, 22));
        source.extend(signal(512, 0.95, 33));
        source.extend(signal(512, -0.6, 44));
        let n = source.len();
        let opts = |a: usize, b: usize| -> Vec<RegionOption> {
            if b > n {
                return Vec::new();
            }
            let mut out = Vec::new();
            for m in &models {
                if let Ok(r) = residual_under(m, &source[a..b], b - a) {
                    out.push(option_from_residual(m.clone(), &r));
                }
            }
            out
        };
        let exhaustive = parse_stateful(n, &grid, usize::MAX, opts).unwrap();
        assert_eq!(
            exhaustive.regions.iter().map(|(a, b)| b - a).sum::<usize>(),
            n
        );
        // A repeated tuple must have been seen as a carousel hit.
        assert!(exhaustive.mru_hits >= 1, "expected tuple reuse");
        for beam in [1usize, 2, 4, 8, 16] {
            let bounded = parse_stateful(n, &grid, beam, opts).unwrap();
            assert!(
                bounded.cost_bits >= exhaustive.cost_bits,
                "beam {beam} beat exhaustive"
            );
            assert_eq!(
                bounded.cost_bits, exhaustive.cost_bits,
                "beam {beam} should find the optimum"
            );
        }
        assert_eq!(exhaustive.models.len(), exhaustive.regions.len());
    }

    #[test]
    fn parser_is_deterministic() {
        let grid = [256usize, 128];
        let models = [ar1(1, 0.9), ar1(1, -0.4), ar1(1, 0.1)];
        let mut source = Vec::new();
        for (i, w) in [0.9, -0.4, 0.1, -0.4].iter().enumerate() {
            source.extend(signal(256, *w, 100 + i as u64));
        }
        let n = source.len();
        let opts = |a: usize, b: usize| -> Vec<RegionOption> {
            if b > n {
                return Vec::new();
            }
            models
                .iter()
                .filter_map(|m| {
                    residual_under(m, &source[a..b], b - a)
                        .ok()
                        .map(|r| option_from_residual(m.clone(), &r))
                })
                .collect()
        };
        let x = parse_stateful(n, &grid, 8, opts).unwrap();
        let y = parse_stateful(n, &grid, 8, opts).unwrap();
        assert_eq!(x.cost_bits, y.cost_bits);
        assert_eq!(x.regions, y.regions);
        assert_eq!(x.models, y.models);
    }

    #[test]
    fn stateful_syntax_object_round_trips_and_closes_exactly() {
        let a: Vec<i32> = signal(512, 0.95, 7);
        let b: Vec<i32> = signal(512, -0.5, 9);
        let mut source = a.clone();
        source.extend_from_slice(&b);
        source.extend_from_slice(&a);
        let m = StatefulSyntaxModel {
            channels: 1,
            segments: vec![
                SyntaxSegment {
                    frames: 512,
                    model: Box::new(ar1(1, 0.95)),
                },
                SyntaxSegment {
                    frames: 512,
                    model: Box::new(ar1(1, -0.5)),
                },
                SyntaxSegment {
                    frames: 512,
                    model: Box::new(ar1(1, 0.95)),
                },
            ],
        };
        let o = LearnedObject::from_intrinsic_exp3(
            LearnedModel::StatefulSyntax(m),
            1,
            1536,
            48_000,
            Vec::new(),
            &source,
        )
        .unwrap();
        assert!(o.verify(&source));
        let bytes = o.canonical_bytes();
        let back = LearnedObject::parse(&bytes).unwrap();
        assert_eq!(back.materialize().unwrap(), source);
    }

    #[test]
    fn nested_segmented_tuple_is_allowed_and_closes() {
        // The speech portfolio's direction-choice model is itself segmented, so
        // a stateful-syntax segment must accept it (block-local evaluation).
        let source = signal(256, 0.9, 77);
        let inner = crate::learned::segmented::SegmentedModel {
            channels: 1,
            segments: vec![
                crate::learned::segmented::Segment {
                    frames: 128,
                    model: Box::new(ar1(1, 0.9)),
                },
                crate::learned::segmented::Segment {
                    frames: 128,
                    model: Box::new(ar1(1, -0.5)),
                },
            ],
        };
        let m = StatefulSyntaxModel {
            channels: 1,
            segments: vec![SyntaxSegment {
                frames: 256,
                model: Box::new(LearnedModel::Segmented(inner)),
            }],
        };
        let o = LearnedObject::from_intrinsic_exp3(
            LearnedModel::StatefulSyntax(m),
            1,
            256,
            48_000,
            Vec::new(),
            &source,
        )
        .unwrap();
        assert!(o.verify(&source));
        let cost = crate::learned::accounting::LearnedCost::of(&o).unwrap();
        assert!(cost.decomposition_is_consistent());
        assert!(cost.model_decomposition_is_consistent());
        assert_eq!(cost.complete_bytes, o.canonical_bytes().len() as u64);
        let bytes = o.canonical_bytes();
        let back = LearnedObject::parse(&bytes).unwrap();
        assert_eq!(back.materialize().unwrap(), source);
    }

    #[test]
    fn parser_rejects_a_grid_with_unbounded_nodes() {
        // A frame-scale grid would need a billion nodes; the guard must reject
        // it without allocating, and it must not loop a billion times either.
        let opts = |_a: usize, _b: usize| -> Vec<RegionOption> { Vec::new() };
        assert!(parse_stateful(1_000_000_000, &[1, 2], usize::MAX, opts).is_none());
    }

    #[test]
    fn rejects_hostile_syntax_bytes() {
        let a = ar1(1, 0.9);
        let m = StatefulSyntaxModel {
            channels: 1,
            segments: vec![SyntaxSegment {
                frames: 64,
                model: Box::new(a),
            }],
        };
        let bytes = m.canonical_bytes();
        // Truncations must never panic.
        for cut in 0..bytes.len() {
            let _ = StatefulSyntaxModel::from_canonical_bytes(&bytes[..cut]);
        }
        // A reference into an empty carousel is rejected, not accepted.
        let mut bad = bytes.clone();
        // Byte layout: 19, channels, count(4), frames(4), symbols (2-bit packed).
        bad[10] = 0x01; // first symbol = 1 (carousel slot 1) with an empty carousel
        assert!(StatefulSyntaxModel::from_canonical_bytes(&bad).is_err());
    }
}
