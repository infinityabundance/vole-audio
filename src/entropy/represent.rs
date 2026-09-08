//! Physical (canonical) SampleObject representations with page-bounded
//! entropy materialization (H.2.1/H.2.4/H.2.6/H.2.13).
//!
//! Two representations are provided:
//!
//! * [`RepresentedLiteral`] — entropy-coded literal samples reconstructing
//!   the canonical U1 intrinsic sample codes **exactly**;
//! * [`RepresentedResidual`] — the existing exact Phase-E residual
//!   (`object::residual::Residual`), with its semantic model unchanged,
//!   whose sparse records are entropy-coded per page.
//!
//! Both are physical representations: the semantic descriptor is carried
//! unchanged and semantics/identity of the semantic object are untouched.
//! Pages are independently decodable; partial materialization decodes only
//! the pages intersecting the requested frame range
//! (`partial == same slice of full` is a tested invariant).
//!
//! RAW fallback (H.2.5): every page compares complete RANS bytes against the
//! RAW alternative and keeps the smaller.
//!
//! Container bytes (Phase-N-ready; canonical, versioned, self-delimiting;
//! grammar not frozen beyond H.2):
//!
//! ```text
//! "vole.entropy.p1"   15 B  profile tag
//! version              1 B  (= 1)
//! kind                 1 B  (1 = literal, 2 = residual)
//! model_mode           1 B  (0 = inline, 1 = shared pool)
//! reserved             1 B  (= 0)
//! page_frames          u32 LE
//! symbolization        1 B   (literal objects; 0 for residual)
//! semantic header      canonical_header_bytes(descriptor)  (fixed length)
//! model block          kind residual only: hypothesis length u32 LE + bytes
//! [pool]               shared mode: model_count u32 LE + canonical models
//! [page index]         page_count u32 LE + 16 B records:
//!                      start_frame u64 LE, frames u32 LE, kind u8, pad 3
//! [page bodies]        kind RANS: N self-delimiting blocks (N = streams)
//!                      kind RAW: raw payload bytes (samples for literal,
//!                      per-page canonical records for residual)
//! ```

use crate::entropy::accounting::CompleteCost;
use crate::entropy::block::{self, Block, BlockPayload, ModelRef};
use crate::entropy::model::SymbolModel;
use crate::entropy::symbol::{self, Symbolization};
use crate::error::{Error, Result};
use crate::hash::sha256::Sha256;
use crate::object::descriptor::{ObjectDescriptor, canonical_header_bytes};
use crate::object::id::ContentId;
use crate::object::residual::{Residual, ResidualModel, ResidualRecord};
use crate::universe::layout::Layout;

pub const FORMAT_TAG: &[u8; 15] = b"vole.entropy.p1";
pub const FORMAT_VERSION: u8 = 1;
pub const KIND_LITERAL: u8 = 1;
pub const KIND_RESIDUAL: u8 = 2;
/// Page index record size in the container.
pub const PAGE_INDEX_RECORD_BYTES: u64 = 16;

/// Page payload kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PageKind {
    Rans = 0,
    Raw = 1,
}

impl PageKind {
    pub const fn code(self) -> u8 {
        self as u8
    }
    pub const fn from_code(c: u8) -> Option<PageKind> {
        match c {
            0 => Some(PageKind::Rans),
            1 => Some(PageKind::Raw),
            _ => None,
        }
    }
}

/// Model policy for an entropy-coded object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ModelMode {
    /// Every stream carries its inline model inside its block.
    Inline = 0,
    /// Streams reference a content-addressable pool stored once per object.
    Shared = 1,
}

impl ModelMode {
    pub const fn code(self) -> u8 {
        self as u8
    }
    pub const fn from_code(c: u8) -> Option<ModelMode> {
        match c {
            0 => Some(ModelMode::Inline),
            1 => Some(ModelMode::Shared),
            _ => None,
        }
    }
}

/// One entropy page of samples (literal representation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiteralPage {
    pub start_frame: u64,
    pub frames: u32,
    pub kind: PageKind,
    /// RANS: one block per symbolization stream. RAW: empty.
    pub blocks: Vec<Block>,
    /// RAW sample bytes (canonical LE codes) when `kind == Raw`.
    pub raw: Vec<u8>,
}

/// Entropy-coded literal SampleObject representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepresentedLiteral {
    /// Semantic descriptor (unchanged; tag `Literal`).
    pub descriptor: ObjectDescriptor,
    pub page_frames: u32,
    pub symbolization: Symbolization,
    pub model_mode: ModelMode,
    /// Shared model pool (used only in `Shared` mode; content-addressed by
    /// canonical model bytes).
    pub pool: Vec<SymbolModel>,
    /// One page per grid cell (last page may be partial).
    pub pages: Vec<LiteralPage>,
}

/// One entropy page of residual records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualPage {
    pub start_frame: u64,
    pub frames: u32,
    pub kind: PageKind,
    /// RANS: channel masks (C streams) followed by 4 delta-lane streams.
    pub blocks: Vec<Block>,
    /// RAW: per-page canonical record bytes.
    pub raw: Vec<u8>,
}

/// Entropy-coded residual representation of a PredictorResidual object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepresentedResidual {
    /// Semantic descriptor (tag `PredictorResidual`, unchanged).
    pub descriptor: ObjectDescriptor,
    /// Semantic hypothesis (unchanged; its canonical bytes are
    /// `hypothesis_bytes` in cost accounting).
    pub model: ResidualModel,
    pub page_frames: u32,
    pub model_mode: ModelMode,
    pub pool: Vec<SymbolModel>,
    pub pages: Vec<ResidualPage>,
}

// ---------------------------------------------------------------------------
// Grid helpers
// ---------------------------------------------------------------------------

/// Grid cell count and last-cell frames for `extent` frames at `page_frames`.
pub fn grid(extent: u64, page_frames: u32) -> (usize, u32) {
    let pf = u64::from(page_frames);
    let cells = extent.div_ceil(pf);
    let last = if extent.is_multiple_of(pf) {
        page_frames
    } else {
        (extent % pf) as u32
    };
    (cells as usize, last)
}

/// Enforce the page-domain ceilings shared by both representations.
fn check_page_domain(page_frames: u32, extent: u64) -> Result<(usize, u32)> {
    if page_frames == 0 || page_frames > crate::limits::MAX_ENTROPY_PAGE_FRAMES {
        return Err(Error::limit("page_frames out of domain"));
    }
    let (cells, last) = grid(extent, page_frames);
    if cells as u32 > crate::limits::MAX_ENTROPY_PAGES_PER_OBJECT {
        return Err(Error::limit("page count exceeds object ceiling"));
    }
    Ok((cells, last))
}

// ---------------------------------------------------------------------------
// RepresentedLiteral encode / decode
// ---------------------------------------------------------------------------

impl RepresentedLiteral {
    /// Encode canonical interleaved samples into an entropy-coded literal.
    ///
    /// Per page, RAW is chosen when its complete bytes do not exceed the RANS
    /// form (H.2.5).
    pub fn encode(
        descriptor: ObjectDescriptor,
        samples: &[i32],
        page_frames: u32,
        symbolization: Symbolization,
        model_mode: ModelMode,
        integrity: bool,
    ) -> Result<RepresentedLiteral> {
        let channels = usize::from(descriptor.layout.count());
        let extent = descriptor.extent_frames;
        let expect = usize::try_from(extent)
            .map_err(|_| Error::limit("extent exceeds host usize"))?
            .checked_mul(channels)
            .ok_or_else(|| Error::limit("extent x channels overflows"))?;
        if samples.len() != expect {
            return Err(Error::malformed(
                "sample count != extent x channels for the descriptor",
            ));
        }
        let (cells, last) = check_page_domain(page_frames, extent)?;

        let mut pool: Vec<SymbolModel> = Vec::new();
        let mut pages = Vec::with_capacity(cells);
        for cell in 0..cells {
            let start = cell as u64 * u64::from(page_frames);
            let frames = if cell + 1 == cells { last } else { page_frames };
            let lo = (start as usize) * channels;
            let hi = lo + frames as usize * channels;
            let page_samples = &samples[lo..hi];

            let streams = symbol::symbolize(symbolization, page_samples, channels)?;
            let mut blocks = Vec::with_capacity(streams.len());
            let mut rans_total: u64 = 0;
            for stream in &streams {
                let blk = encode_stream_block(
                    symbolization,
                    descriptor.layout.count(),
                    stream,
                    &mut pool,
                    model_mode,
                    integrity,
                )?;
                rans_total += block::block_bytes(&blk)?.len() as u64;
                blocks.push(blk);
            }

            // RAW alternative: raw LE sample bytes of the page.
            let mut raw = Vec::with_capacity(page_samples.len() * 4);
            for s in page_samples {
                raw.extend_from_slice(&s.to_le_bytes());
            }
            let kind = if rans_total < raw.len() as u64 {
                PageKind::Rans
            } else {
                PageKind::Raw
            };
            pages.push(LiteralPage {
                start_frame: start,
                frames,
                kind,
                blocks: if kind == PageKind::Rans {
                    blocks
                } else {
                    Vec::new()
                },
                raw: if kind == PageKind::Raw {
                    raw
                } else {
                    Vec::new()
                },
            });
        }
        Ok(RepresentedLiteral {
            descriptor,
            page_frames,
            symbolization,
            model_mode,
            pool,
            pages,
        })
    }

    /// Decode one page's samples (interleaved i32 codes).
    pub fn page_samples(&self, page: &LiteralPage) -> Result<Vec<i32>> {
        let channels = usize::from(self.descriptor.layout.count());
        let total = page.frames as usize * channels;
        match page.kind {
            PageKind::Raw => {
                if page.raw.len() != total * 4 {
                    return Err(Error::malformed("RAW page length mismatch"));
                }
                let mut out = Vec::with_capacity(total);
                for w in page.raw.as_chunks::<4>().0 {
                    out.push(i32::from_le_bytes([w[0], w[1], w[2], w[3]]));
                }
                Ok(out)
            }
            PageKind::Rans => {
                let streams = decode_blocks(&page.blocks, &self.pool)?;
                symbol::desymbolize(self.symbolization, &streams, channels)
            }
        }
    }

    /// Materialize the canonical interleaved codes for `[start_frame,
    /// start_frame + frames)`; decodes only intersecting pages.
    pub fn materialize(&self, start_frame: u64, frames: u32) -> Result<Vec<i32>> {
        if frames == 0 {
            return Ok(Vec::new());
        }
        let extent = self.descriptor.extent_frames;
        if start_frame >= extent || start_frame + u64::from(frames) > extent {
            return Err(Error::malformed("materialize range outside object extent"));
        }
        let channels = usize::from(self.descriptor.layout.count());
        let end = start_frame + u64::from(frames);
        let mut out = Vec::with_capacity(frames as usize * channels);
        for page in &self.pages {
            let ps = page.start_frame;
            let pe = ps + u64::from(page.frames);
            if ps >= end || pe <= start_frame {
                continue;
            }
            let samples = self.page_samples(page)?;
            let take_lo = start_frame.saturating_sub(ps) as usize * channels;
            let take_hi = (end.min(pe) - ps) as usize * channels;
            out.extend_from_slice(&samples[take_lo..take_hi]);
        }
        debug_assert_eq!(out.len(), frames as usize * channels);
        Ok(out)
    }

    /// Full materialization (equality anchor for partial == full slices).
    pub fn materialize_full(&self) -> Result<Vec<i32>> {
        let mut out = Vec::with_capacity(
            usize::try_from(self.descriptor.extent_frames).unwrap_or(0)
                * usize::from(self.descriptor.layout.count()),
        );
        for page in &self.pages {
            out.extend_from_slice(&self.page_samples(page)?);
        }
        Ok(out)
    }

    /// Pages intersecting `[start, start + frames)` (observation halo info).
    pub fn pages_touching(&self, start_frame: u64, frames: u32) -> Vec<u64> {
        let end = start_frame + u64::from(frames);
        self.pages
            .iter()
            .filter(|p| {
                let ps = p.start_frame;
                let pe = ps + u64::from(p.frames);
                ps < end && pe > start_frame
            })
            .map(|p| p.start_frame)
            .collect()
    }

    /// Complete-cost breakdown of this representation.
    pub fn cost(&self, canonical_literal_bytes: u64) -> Result<CompleteCost> {
        let mut model_bytes = 0u64;
        let mut payload_bytes = 0u64;
        let mut integrity_bytes = 0u64;
        for page in &self.pages {
            match page.kind {
                PageKind::Rans => {
                    for b in &page.blocks {
                        match &b.payload {
                            BlockPayload::Rans { model, bytes, .. } => {
                                payload_bytes += bytes.len() as u64;
                                if let ModelRef::Inline(m) = model {
                                    model_bytes += m.canonical_bytes().len() as u64;
                                }
                            }
                            BlockPayload::Raw { bytes } => payload_bytes += bytes.len() as u64,
                        }
                        if b.integrity.is_some() {
                            integrity_bytes += 33;
                        }
                    }
                }
                PageKind::Raw => payload_bytes += page.raw.len() as u64,
            }
        }
        if self.model_mode == ModelMode::Shared {
            for m in &self.pool {
                model_bytes += m.canonical_bytes().len() as u64;
            }
        }
        let raw = self.descriptor.extent_frames * u64::from(self.descriptor.layout.count()) * 4;
        let mut c = CompleteCost {
            metadata_bytes: (15 + 1 + 1 + 1 + 1 + 1 + 4 + 1 + 46) as u64,
            hypothesis_bytes: 0,
            model_bytes,
            payload_bytes,
            index_bytes: PAGE_INDEX_RECORD_BYTES * self.pages.len() as u64,
            dependency_bytes: 0,
            integrity_bytes,
            complete_bytes: 0,
            raw_sample_bytes: raw,
            canonical_literal_bytes,
            source_wav_bytes: 0,
        };
        c.compute();
        Ok(c)
    }
}

// ---------------------------------------------------------------------------
// RepresentedResidual encode / decode
// ---------------------------------------------------------------------------

impl RepresentedResidual {
    /// Encode an existing exact residual into per-page entropy-coded form.
    /// The semantic model and record semantics are unchanged; reconstruction
    /// returns a byte-identical `Residual`.
    pub fn encode(
        descriptor: ObjectDescriptor,
        residual: &Residual,
        page_frames: u32,
        model_mode: ModelMode,
        integrity: bool,
    ) -> Result<RepresentedResidual> {
        let channels = usize::from(descriptor.layout.count());
        let extent = descriptor.extent_frames;
        if Residual::new(
            &descriptor,
            residual.model.clone(),
            residual.records.clone(),
        )
        .is_none()
        {
            return Err(Error::malformed("residual out of domain for descriptor"));
        }
        let (cells, last) = check_page_domain(page_frames, extent)?;

        let mut pool: Vec<SymbolModel> = Vec::new();
        let mut pages = Vec::with_capacity(cells);
        for cell in 0..cells {
            let start = cell as u64 * u64::from(page_frames);
            let frames = if cell + 1 == cells { last } else { page_frames };
            let page = encode_residual_page(
                residual, channels, start, frames, &mut pool, model_mode, integrity,
            )?;
            pages.push(page);
        }
        Ok(RepresentedResidual {
            descriptor,
            model: residual.model.clone(),
            page_frames,
            model_mode,
            pool,
            pages,
        })
    }

    /// Decode one page into residual records (absolute frames).
    pub fn page_records(&self, page: &ResidualPage) -> Result<Vec<ResidualRecord>> {
        let channels = usize::from(self.descriptor.layout.count());
        match page.kind {
            PageKind::Raw => parse_raw_records(&page.raw),
            PageKind::Rans => {
                let streams = decode_blocks(&page.blocks, &self.pool)?;
                if streams.len() != channels + 4 {
                    return Err(Error::malformed("residual page stream count mismatch"));
                }
                let (mask_streams, delta_streams) = streams.split_at(channels);
                residual_from_streams(
                    page.start_frame,
                    page.frames,
                    channels,
                    mask_streams,
                    delta_streams,
                )
            }
        }
    }

    /// Reconstruct the full semantic residual (equality anchor).
    pub fn reconstruct_full(&self) -> Result<Residual> {
        let mut records = Vec::new();
        for page in &self.pages {
            records.extend_from_slice(&self.page_records(page)?);
        }
        Residual::new(&self.descriptor, self.model.clone(), records)
            .ok_or_else(|| Error::malformed("reconstructed residual out of domain"))
    }

    /// Exact closure samples for `[start_frame, start_frame + frames)`:
    /// `sat_i32(H(f, ch) + R(f, ch))`, decoding only intersecting pages.
    /// Equal to the same slice of the full closure by construction.
    pub fn materialize_closure(&self, start_frame: u64, frames: u32) -> Result<Vec<i32>> {
        if frames == 0 {
            return Ok(Vec::new());
        }
        let extent = self.descriptor.extent_frames;
        if start_frame >= extent || start_frame + u64::from(frames) > extent {
            return Err(Error::malformed("closure range outside object extent"));
        }
        let channels = usize::from(self.descriptor.layout.count());
        let end = start_frame + u64::from(frames);

        // Pass 1: pure hypothesis (channel 0 = model output; others 0).
        let mut out = vec![0i32; frames as usize * channels];
        if channels >= 1 {
            for f in 0..frames as usize {
                let h = self.model.model_sample(start_frame + f as u64);
                out[f * channels] = h;
            }
        }
        // Pass 2: apply residual deltas (single saturation, Phase-E rule).
        for page in &self.pages {
            let ps = page.start_frame;
            let pe = ps + u64::from(page.frames);
            if ps >= end || pe <= start_frame {
                continue;
            }
            for r in self.page_records(page)? {
                if r.frame < start_frame || r.frame >= end {
                    continue;
                }
                let f = (r.frame - start_frame) as usize;
                let slot = f * channels + usize::from(r.channel);
                let h = i64::from(out[slot]);
                out[slot] = crate::universe::arithmetic::sat_i32(h + i64::from(r.delta));
            }
        }
        Ok(out)
    }

    /// Pages intersecting `[start, start + frames)` (observation halo info).
    pub fn pages_touching(&self, start_frame: u64, frames: u32) -> Vec<u64> {
        let end = start_frame + u64::from(frames);
        self.pages
            .iter()
            .filter(|p| {
                let ps = p.start_frame;
                let pe = ps + u64::from(p.frames);
                ps < end && pe > start_frame
            })
            .map(|p| p.start_frame)
            .collect()
    }

    /// Complete-cost breakdown of this representation.
    pub fn cost(&self, canonical_literal_bytes: u64) -> Result<CompleteCost> {
        let mut model_bytes = 0u64;
        let mut payload_bytes = 0u64;
        let mut integrity_bytes = 0u64;
        for page in &self.pages {
            match page.kind {
                PageKind::Rans => {
                    for b in &page.blocks {
                        match &b.payload {
                            BlockPayload::Rans { model, bytes, .. } => {
                                payload_bytes += bytes.len() as u64;
                                if let ModelRef::Inline(m) = model {
                                    model_bytes += m.canonical_bytes().len() as u64;
                                }
                            }
                            BlockPayload::Raw { bytes } => payload_bytes += bytes.len() as u64,
                        }
                        if b.integrity.is_some() {
                            integrity_bytes += 33;
                        }
                    }
                }
                PageKind::Raw => payload_bytes += page.raw.len() as u64,
            }
        }
        if self.model_mode == ModelMode::Shared {
            for m in &self.pool {
                model_bytes += m.canonical_bytes().len() as u64;
            }
        }
        let raw = self.descriptor.extent_frames * u64::from(self.descriptor.layout.count()) * 4;
        let mut c = CompleteCost {
            metadata_bytes: (15 + 1 + 1 + 1 + 1 + 1 + 4 + 1 + 46) as u64,
            hypothesis_bytes: semantic_model_bytes(&self.model) as u64,
            model_bytes,
            payload_bytes,
            index_bytes: PAGE_INDEX_RECORD_BYTES * self.pages.len() as u64,
            dependency_bytes: 0,
            integrity_bytes,
            complete_bytes: 0,
            raw_sample_bytes: raw,
            canonical_literal_bytes,
            source_wav_bytes: 0,
        };
        c.compute();
        Ok(c)
    }
}

// ---------------------------------------------------------------------------
// Shared-model + stream-coding helpers
// ---------------------------------------------------------------------------

fn byte_counts(stream: &[u8]) -> [u64; 256] {
    let mut counts = [0u64; 256];
    for &b in stream {
        counts[b as usize] += 1;
    }
    counts
}

/// Return the pool index of the model built from `stream`, inserting when
/// absent (content-addressed by canonical model bytes).
fn shared_pool_index(pool: &mut Vec<SymbolModel>, stream: &[u8]) -> Result<u32> {
    let model = SymbolModel::from_byte_counts(&byte_counts(stream))
        .ok_or_else(|| Error::internal("empty stream model"))?;
    let canonical = model.canonical_bytes();
    for (i, m) in pool.iter().enumerate() {
        if m.canonical_bytes() == canonical {
            return Ok(i as u32);
        }
    }
    if pool.len() as u32 >= crate::limits::MAX_ENTROPY_SHARED_MODELS {
        return Err(Error::limit("shared model pool ceiling exceeded"));
    }
    pool.push(model);
    Ok(pool.len() as u32 - 1)
}

fn digest_of_block(blk: &Block) -> Result<[u8; 32]> {
    let bytes = block::block_bytes(blk)?;
    // The digest covers the canonical bytes up to (not including) the
    // trailing integrity flag byte written by the no-integrity serialization.
    if bytes.is_empty() {
        return Err(Error::internal("empty block bytes"));
    }
    Ok(Sha256::digest(&bytes[..bytes.len() - 1]))
}

/// Encode one symbol stream as a RANS block (inline or shared model), with
/// optional per-block integrity.
fn encode_stream_block(
    symbolization: Symbolization,
    channel_scope: u8,
    stream: &[u8],
    pool: &mut Vec<SymbolModel>,
    model_mode: ModelMode,
    integrity: bool,
) -> Result<Block> {
    // An empty stream cannot be rANS-encoded; a RAW empty block is the
    // canonical zero-cost carrier (page decode handles it transparently).
    if stream.is_empty() {
        return Ok(Block {
            symbolization,
            channel_scope,
            payload: BlockPayload::Raw { bytes: Vec::new() },
            integrity: None,
        });
    }
    let counts = byte_counts(stream);
    let model_ref = match model_mode {
        ModelMode::Inline => ModelRef::Inline(
            SymbolModel::from_byte_counts(&counts)
                .ok_or_else(|| Error::internal("empty stream model"))?,
        ),
        ModelMode::Shared => ModelRef::Shared(shared_pool_index(pool, stream)?),
    };
    let model = match &model_ref {
        ModelRef::Inline(m) => m.clone(),
        ModelRef::Shared(idx) => pool
            .get(*idx as usize)
            .cloned()
            .ok_or_else(|| Error::internal("shared model pool index missing during encode"))?,
    };
    let encoded = block::encode_rans_stream(&model, stream, crate::entropy::rans::SCALE_BITS)?;
    let mut blk = Block {
        symbolization,
        channel_scope,
        payload: BlockPayload::Rans {
            model: model_ref,
            symbol_count: stream.len() as u64,
            bytes: encoded,
        },
        integrity: None,
    };
    if integrity {
        blk.integrity = Some(digest_of_block(&blk)?);
    }
    Ok(blk)
}

/// Decode a block list into symbol streams (RAW blocks pass through).
fn decode_blocks(blocks: &[Block], pool: &[SymbolModel]) -> Result<Vec<Vec<u8>>> {
    let mut out = Vec::with_capacity(blocks.len());
    for b in blocks {
        out.push(block::decode_block_symbols(
            b,
            pool,
            crate::entropy::rans::SCALE_BITS,
        )?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Residual page coding
// ---------------------------------------------------------------------------

/// Encode one residual page: per-channel correction masks + 4 delta lanes,
/// or RAW canonical record bytes when smaller.
#[allow(clippy::too_many_arguments)]
fn encode_residual_page(
    residual: &Residual,
    channels: usize,
    start: u64,
    frames: u32,
    pool: &mut Vec<SymbolModel>,
    model_mode: ModelMode,
    integrity: bool,
) -> Result<ResidualPage> {
    let ch = channels;
    // Partition records of this page per channel, in (channel, frame) order.
    // One mask bit per frame, packed LSB-first; byte count = ceil(frames/8).
    let mut masks: Vec<Vec<u8>> = vec![vec![0u8; (frames as usize).div_ceil(8)]; ch];
    let mut delta_zz: Vec<u32> = Vec::new();
    let mut records: Vec<ResidualRecord> = Vec::new();
    let page_end = start + u64::from(frames);
    for r in &residual.records {
        if r.frame < start || r.frame >= page_end {
            continue;
        }
        let c = usize::from(r.channel);
        if c >= ch {
            continue;
        }
        let off = (r.frame - start) as usize;
        masks[c][off >> 3] |= 1u8 << (off & 7);
        records.push(*r);
    }
    records.sort_by_key(|r| (r.channel, r.frame));
    delta_zz.reserve(records.len());
    for r in &records {
        delta_zz.push(symbol::zigzag(r.delta));
    }

    // Candidate blocks: C mask streams + 4 delta lanes (each over the whole
    // zz-delta list, channel-major byte positions).
    let mut blocks: Vec<Block> = Vec::with_capacity(ch + 4);
    let mut rans_total = 0u64;
    for mask in &masks {
        let blk = encode_stream_block(
            Symbolization::Identity,
            ch as u8,
            mask,
            pool,
            model_mode,
            integrity,
        )?;
        rans_total += block::block_bytes(&blk)?.len() as u64;
        blocks.push(blk);
    }
    for p in 0..4 {
        let lane: Vec<u8> = delta_zz.iter().map(|z| symbol::byte_of_le(*z, p)).collect();
        let blk = encode_stream_block(
            Symbolization::Identity,
            ch as u8,
            &lane,
            pool,
            model_mode,
            integrity,
        )?;
        rans_total += block::block_bytes(&blk)?.len() as u64;
        blocks.push(blk);
    }

    // RAW alternative: canonical per-page record bytes.
    let mut raw = Vec::with_capacity(4 + records.len() * 13);
    raw.extend_from_slice(&(records.len() as u32).to_le_bytes());
    for r in &records {
        raw.extend_from_slice(&r.frame.to_le_bytes());
        raw.push(r.channel);
        raw.extend_from_slice(&r.delta.to_le_bytes());
    }
    let kind = if rans_total < raw.len() as u64 {
        PageKind::Rans
    } else {
        PageKind::Raw
    };
    Ok(ResidualPage {
        start_frame: start,
        frames,
        kind,
        blocks: if kind == PageKind::Rans {
            blocks
        } else {
            Vec::new()
        },
        raw: if kind == PageKind::Raw {
            raw
        } else {
            Vec::new()
        },
    })
}

/// Reconstruct residual records from per-channel masks + delta lanes.
fn residual_from_streams(
    start: u64,
    frames: u32,
    channels: usize,
    mask_streams: &[Vec<u8>],
    delta_streams: &[Vec<u8>],
) -> Result<Vec<ResidualRecord>> {
    let mask_bytes = (frames as usize).div_ceil(8);
    if mask_streams.len() != channels {
        return Err(Error::malformed("mask stream count != channels"));
    }
    if delta_streams.len() != 4 {
        return Err(Error::malformed("delta lane count != 4"));
    }
    let mut cursor = 0usize;
    let mut out = Vec::new();
    for (c, mask) in mask_streams.iter().enumerate() {
        if mask.len() != mask_bytes {
            return Err(Error::malformed("mask stream length mismatch"));
        }
        for f in 0..frames as usize {
            let set = (mask[f >> 3] >> (f & 7)) & 1 != 0;
            if !set {
                continue;
            }
            // The four delta lanes hold byte positions of the same zz-coded
            // delta at `cursor` (channel-major record order).
            let mut b = [0u8; 4];
            #[allow(clippy::needless_range_loop)]
            for p in 0..4 {
                b[p] = *delta_streams
                    .get(p)
                    .and_then(|l| l.get(cursor))
                    .ok_or_else(|| Error::malformed("delta lane truncated"))?;
            }
            let delta = symbol::unzigzag(u32::from_le_bytes(b));
            out.push(ResidualRecord {
                frame: start + f as u64,
                channel: c as u8,
                delta,
            });
            cursor += 1;
        }
    }
    if cursor != delta_streams[0].len() {
        return Err(Error::malformed("delta lane length mismatch"));
    }
    Ok(out)
}

fn parse_raw_records(raw: &[u8]) -> Result<Vec<ResidualRecord>> {
    if raw.len() < 4 {
        return Err(Error::malformed("truncated RAW record page"));
    }
    let count = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let need = 4usize
        .checked_add(
            count
                .checked_mul(13)
                .ok_or_else(|| Error::limit("record count overflow"))?,
        )
        .ok_or_else(|| Error::limit("record bytes overflow"))?;
    if raw.len() != need {
        return Err(Error::malformed("RAW record page length mismatch"));
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = 4 + i * 13;
        let frame = u64::from_le_bytes(raw[at..at + 8].try_into().unwrap());
        let channel = raw[at + 8];
        let delta = i32::from_le_bytes(raw[at + 9..at + 13].try_into().unwrap());
        out.push(ResidualRecord {
            frame,
            channel,
            delta,
        });
    }
    Ok(out)
}

fn semantic_model_bytes(model: &ResidualModel) -> usize {
    match model {
        ResidualModel::Zero => 1,
        ResidualModel::Constant(_) => 5,
        ResidualModel::Periodic { cycle } => 1 + 8 + cycle.len() * 4,
    }
}

// ---------------------------------------------------------------------------
// Container bytes + parser (Phase-N-ready canonical records)
// ---------------------------------------------------------------------------

/// Serialize the shared model pool.
fn pool_bytes(pool: &[SymbolModel]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(pool.len() as u32).to_le_bytes());
    for m in pool {
        out.extend_from_slice(&m.canonical_bytes());
    }
    out
}

/// Parse the shared model pool (validates bounds and model invariants).
fn parse_pool(bytes: &[u8], mut pos: usize) -> Result<(Vec<SymbolModel>, usize)> {
    if bytes.len() < pos + 4 {
        return Err(Error::malformed("truncated pool count"));
    }
    let count = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    if count > crate::limits::MAX_ENTROPY_SHARED_MODELS as usize {
        return Err(Error::limit("shared model count above ceiling"));
    }
    let mut pool = Vec::with_capacity(count);
    for _ in 0..count {
        if bytes.len() < pos + 2 {
            return Err(Error::malformed("truncated model count"));
        }
        let n = u16::from_le_bytes(bytes[pos..pos + 2].try_into().unwrap()) as usize;
        if n == 0 || n > crate::limits::MAX_MODEL_ALPHABET {
            return Err(Error::malformed("model alphabet out of domain"));
        }
        let mlen = 2 + n * 6;
        if pos + mlen > bytes.len() {
            return Err(Error::malformed("truncated model bytes"));
        }
        let m = SymbolModel::parse_canonical(&bytes[pos..pos + mlen])
            .ok_or_else(|| Error::malformed("invalid pool model bytes"))?;
        pool.push(m);
        pos += mlen;
    }
    Ok((pool, pos))
}

/// Semantic header region: profile tag (19) + representation + extent + ...
/// `canonical_header_bytes` is a fixed 46-byte region.
const SEMANTIC_HEADER_LEN: usize = 46;

fn container_prefix(
    kind: u8,
    page_frames: u32,
    symbolization: u8,
    descriptor: &ObjectDescriptor,
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(FORMAT_TAG);
    out.push(FORMAT_VERSION);
    out.push(kind);
    out.push(0); // model_mode placeholder
    out.push(0); // reserved
    out.extend_from_slice(&page_frames.to_le_bytes());
    out.push(symbolization);
    out.extend_from_slice(&canonical_header_bytes(descriptor));
    out
}

/// Canonical container bytes of a represented literal.
pub fn literal_container_bytes(rl: &RepresentedLiteral) -> Result<Vec<u8>> {
    let mut prefix = container_prefix(
        KIND_LITERAL,
        rl.page_frames,
        rl.symbolization.code(),
        &rl.descriptor,
    );
    prefix[17] = rl.model_mode.code(); // model_mode slot (see container layout)
    let mut out = prefix;
    out.extend_from_slice(&pool_bytes(&rl.pool));
    out.extend_from_slice(&(rl.pages.len() as u32).to_le_bytes());
    for p in &rl.pages {
        out.extend_from_slice(&p.start_frame.to_le_bytes());
        out.extend_from_slice(&p.frames.to_le_bytes());
        out.push(p.kind.code());
        out.extend_from_slice(&[0u8; 3]); // pad
    }
    for p in &rl.pages {
        match p.kind {
            PageKind::Rans => {
                for b in &p.blocks {
                    out.extend_from_slice(&block::block_bytes(b)?);
                }
            }
            PageKind::Raw => out.extend_from_slice(&p.raw),
        }
    }
    Ok(out)
}

/// Parse a literal container into its in-memory representation.
pub fn parse_literal_container(bytes: &[u8]) -> Result<RepresentedLiteral> {
    let mut pos = 0usize;
    if bytes.len() < pos + 15 || &bytes[..15] != FORMAT_TAG {
        return Err(Error::malformed("bad container tag"));
    }
    pos = 15;
    if bytes.len() < pos + 1 || bytes[pos] != FORMAT_VERSION {
        return Err(Error::malformed("bad container version"));
    }
    pos += 1;
    if bytes.len() < pos + 1 || bytes[pos] != KIND_LITERAL {
        return Err(Error::malformed("not a literal container"));
    }
    pos += 1;
    if bytes.len() < pos + 1 {
        return Err(Error::malformed("truncated model mode"));
    }
    let model_mode =
        ModelMode::from_code(bytes[pos]).ok_or_else(|| Error::malformed("invalid model mode"))?;
    pos += 1;
    if bytes.len() < pos + 1 || bytes[pos] != 0 {
        return Err(Error::malformed("reserved byte nonzero"));
    }
    pos += 1;
    if bytes.len() < pos + 4 {
        return Err(Error::malformed("truncated page_frames"));
    }
    let page_frames = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
    pos += 4;
    if page_frames == 0 || page_frames > crate::limits::MAX_ENTROPY_PAGE_FRAMES {
        return Err(Error::limit("page_frames out of domain"));
    }
    if bytes.len() < pos + 1 {
        return Err(Error::malformed("truncated symbolization"));
    }
    let symbolization = Symbolization::from_code(bytes[pos])
        .ok_or_else(|| Error::malformed("invalid symbolization"))?;
    pos += 1;
    if bytes.len() < pos + SEMANTIC_HEADER_LEN {
        return Err(Error::malformed("truncated semantic header"));
    }
    let descriptor = parse_semantic_header(&bytes[pos..pos + SEMANTIC_HEADER_LEN])?;
    pos += SEMANTIC_HEADER_LEN;
    let (pool, after) = parse_pool(bytes, pos)?;
    pos = after;
    if bytes.len() < pos + 4 {
        return Err(Error::malformed("truncated page count"));
    }
    let page_count = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    if page_count > crate::limits::MAX_ENTROPY_PAGES_PER_OBJECT as usize {
        return Err(Error::limit("page count above ceiling"));
    }
    let mut meta = Vec::with_capacity(page_count);
    for _ in 0..page_count {
        if bytes.len() < pos + 16 {
            return Err(Error::malformed("truncated page index"));
        }
        let start_frame = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        let frames = u32::from_le_bytes(bytes[pos + 8..pos + 12].try_into().unwrap());
        let kind = PageKind::from_code(bytes[pos + 12])
            .ok_or_else(|| Error::malformed("invalid page kind"))?;
        if frames == 0 || frames > page_frames {
            return Err(Error::malformed("page frames out of domain"));
        }
        pos += 16;
        meta.push((start_frame, frames, kind));
    }
    let channels = usize::from(descriptor.layout.count());
    let streams = symbolization.stream_count();
    let mut pages = Vec::with_capacity(page_count);
    for (start_frame, frames, kind) in meta {
        match kind {
            PageKind::Rans => {
                let mut blocks = Vec::with_capacity(streams);
                for _ in 0..streams {
                    let (b, rest) = block::parse_block(&bytes[pos..])?;
                    pos = bytes.len() - rest.len();
                    blocks.push(b);
                }
                pages.push(LiteralPage {
                    start_frame,
                    frames,
                    kind,
                    blocks,
                    raw: Vec::new(),
                });
            }
            PageKind::Raw => {
                let expect = frames as usize * channels * 4;
                if bytes.len() < pos + expect {
                    return Err(Error::malformed("truncated RAW page"));
                }
                let raw = bytes[pos..pos + expect].to_vec();
                pos += expect;
                pages.push(LiteralPage {
                    start_frame,
                    frames,
                    kind,
                    blocks: Vec::new(),
                    raw,
                });
            }
        }
    }
    if pos != bytes.len() {
        return Err(Error::malformed("trailing bytes after container"));
    }
    Ok(RepresentedLiteral {
        descriptor,
        page_frames,
        symbolization,
        model_mode,
        pool,
        pages,
    })
}

/// Canonical container bytes of a represented residual.
pub fn residual_container_bytes(rr: &RepresentedResidual) -> Result<Vec<u8>> {
    let mut prefix = container_prefix(KIND_RESIDUAL, rr.page_frames, 0, &rr.descriptor);
    prefix[17] = rr.model_mode.code(); // model_mode slot (see container layout)
    let mut out = prefix;
    // Semantic hypothesis bytes.
    let mut model = Vec::new();
    match &rr.model {
        ResidualModel::Zero => model.push(0),
        ResidualModel::Constant(l) => {
            model.push(1);
            model.extend_from_slice(&l.to_le_bytes());
        }
        ResidualModel::Periodic { cycle } => {
            model.push(2);
            model.extend_from_slice(&(cycle.len() as u64).to_le_bytes());
            for s in cycle {
                model.extend_from_slice(&s.to_le_bytes());
            }
        }
    }
    out.extend_from_slice(&(model.len() as u32).to_le_bytes());
    out.extend_from_slice(&model);
    out.extend_from_slice(&pool_bytes(&rr.pool));
    out.extend_from_slice(&(rr.pages.len() as u32).to_le_bytes());
    for p in &rr.pages {
        out.extend_from_slice(&p.start_frame.to_le_bytes());
        out.extend_from_slice(&p.frames.to_le_bytes());
        out.push(p.kind.code());
        out.extend_from_slice(&[0u8; 3]);
    }
    for p in &rr.pages {
        match p.kind {
            PageKind::Rans => {
                for b in &p.blocks {
                    out.extend_from_slice(&block::block_bytes(b)?);
                }
            }
            PageKind::Raw => out.extend_from_slice(&p.raw),
        }
    }
    Ok(out)
}

/// Parse a residual container into its in-memory representation.
pub fn parse_residual_container(bytes: &[u8]) -> Result<RepresentedResidual> {
    if bytes.len() < 15 || &bytes[..15] != FORMAT_TAG {
        return Err(Error::malformed("bad container tag"));
    }
    let mut pos = 15usize;
    if bytes.len() < pos + 1 || bytes[pos] != FORMAT_VERSION {
        return Err(Error::malformed("bad container version"));
    }
    pos += 1;
    if bytes.len() < pos + 1 || bytes[pos] != KIND_RESIDUAL {
        return Err(Error::malformed("not a residual container"));
    }
    pos += 1;
    if bytes.len() < pos + 1 {
        return Err(Error::malformed("truncated model mode"));
    }
    let model_mode =
        ModelMode::from_code(bytes[pos]).ok_or_else(|| Error::malformed("invalid model mode"))?;
    pos += 1;
    if bytes.len() < pos + 1 || bytes[pos] != 0 {
        return Err(Error::malformed("reserved byte nonzero or truncated"));
    }
    pos += 1;
    if bytes.len() < pos + 4 {
        return Err(Error::malformed("truncated page_frames"));
    }
    let page_frames = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
    pos += 4;
    if page_frames == 0 || page_frames > crate::limits::MAX_ENTROPY_PAGE_FRAMES {
        return Err(Error::limit("page_frames out of domain"));
    }
    if bytes.len() < pos + 1 {
        return Err(Error::malformed("truncated symbolization slot"));
    }
    pos += 1; // symbolization slot (0 for residual)
    if bytes.len() < pos + SEMANTIC_HEADER_LEN {
        return Err(Error::malformed("truncated semantic header"));
    }
    let descriptor = parse_semantic_header(&bytes[pos..pos + SEMANTIC_HEADER_LEN])?;
    pos += SEMANTIC_HEADER_LEN;
    // Hypothesis model bytes.
    if bytes.len() < pos + 4 {
        return Err(Error::malformed("truncated hypothesis length"));
    }
    let hlen = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    if bytes.len() < pos + hlen {
        return Err(Error::malformed("truncated hypothesis bytes"));
    }
    let model = parse_semantic_model(&bytes[pos..pos + hlen])?;
    pos += hlen;
    let (pool, after) = parse_pool(bytes, pos)?;
    pos = after;
    if bytes.len() < pos + 4 {
        return Err(Error::malformed("truncated page count"));
    }
    let page_count = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    if page_count > crate::limits::MAX_ENTROPY_PAGES_PER_OBJECT as usize {
        return Err(Error::limit("page count above ceiling"));
    }
    let mut meta = Vec::with_capacity(page_count);
    for _ in 0..page_count {
        if bytes.len() < pos + 16 {
            return Err(Error::malformed("truncated page index"));
        }
        let start_frame = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        let frames = u32::from_le_bytes(bytes[pos + 8..pos + 12].try_into().unwrap());
        let kind = PageKind::from_code(bytes[pos + 12])
            .ok_or_else(|| Error::malformed("invalid page kind"))?;
        if frames == 0 || frames > page_frames {
            return Err(Error::malformed("page frames out of domain"));
        }
        pos += 16;
        meta.push((start_frame, frames, kind));
    }
    let channels = usize::from(descriptor.layout.count());
    let streams = channels + 4;
    let mut pages = Vec::with_capacity(page_count);
    for (start_frame, frames, kind) in meta {
        match kind {
            PageKind::Rans => {
                let mut blocks = Vec::with_capacity(streams);
                for _ in 0..streams {
                    let (b, rest) = block::parse_block(&bytes[pos..])?;
                    pos = bytes.len() - rest.len();
                    blocks.push(b);
                }
                pages.push(ResidualPage {
                    start_frame,
                    frames,
                    kind,
                    blocks,
                    raw: Vec::new(),
                });
            }
            PageKind::Raw => {
                // RAW record pages: count prefix (4) + 13/record.
                if bytes.len() < pos + 4 {
                    return Err(Error::malformed("truncated RAW record count"));
                }
                let count = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
                let len = 4 + count * 13;
                if bytes.len() < pos + len {
                    return Err(Error::malformed("truncated RAW records"));
                }
                let raw = bytes[pos..pos + len].to_vec();
                pos += len;
                pages.push(ResidualPage {
                    start_frame,
                    frames,
                    kind,
                    blocks: Vec::new(),
                    raw,
                });
            }
        }
    }
    if pos != bytes.len() {
        return Err(Error::malformed("trailing bytes after container"));
    }
    Ok(RepresentedResidual {
        descriptor,
        model,
        page_frames,
        model_mode,
        pool,
        pages,
    })
}

/// Parse a canonical semantic header region back into a descriptor.
fn parse_semantic_header(bytes: &[u8]) -> Result<ObjectDescriptor> {
    if bytes.len() != SEMANTIC_HEADER_LEN {
        return Err(Error::malformed("semantic header length"));
    }
    // canonical_header_bytes layout: profile(19) tag(1) extent(8) layout(1)
    // loop_flag(1) loop_start(8) loop_end(8) = 46.
    let tag = bytes[19];
    let representation = crate::object::descriptor::Representation::from_tag(tag)
        .ok_or_else(|| Error::malformed("unknown representation tag"))?;
    let extent = u64::from_le_bytes(bytes[20..28].try_into().unwrap());
    let layout = Layout::checked(bytes[28]).ok_or_else(|| Error::malformed("invalid layout"))?;
    let loop_flag = bytes[29];
    let loop_region = if loop_flag == 1 {
        let start = u64::from_le_bytes(bytes[30..38].try_into().unwrap());
        let end = u64::from_le_bytes(bytes[38..46].try_into().unwrap());
        Some(
            crate::object::LoopRegion::new(start, end)
                .ok_or_else(|| Error::malformed("loop region out of domain"))?,
        )
    } else if loop_flag == 0 {
        None
    } else {
        return Err(Error::malformed("invalid loop flag"));
    };
    ObjectDescriptor::new(representation, extent, layout, loop_region)
        .ok_or_else(|| Error::malformed("descriptor out of domain"))
}

/// Parse semantic hypothesis model bytes (see container format).
fn parse_semantic_model(bytes: &[u8]) -> Result<ResidualModel> {
    match bytes.first().copied() {
        Some(0) => Ok(ResidualModel::Zero),
        Some(1) => {
            if bytes.len() != 5 {
                return Err(Error::malformed("constant model length"));
            }
            Ok(ResidualModel::Constant(i32::from_le_bytes(
                bytes[1..5].try_into().unwrap(),
            )))
        }
        Some(2) => {
            if bytes.len() < 9 {
                return Err(Error::malformed("truncated periodic model"));
            }
            let n = u64::from_le_bytes(bytes[1..9].try_into().unwrap()) as usize;
            if bytes.len() != 9 + n * 4 {
                return Err(Error::malformed("periodic model length"));
            }
            let mut cycle = Vec::with_capacity(n);
            for i in 0..n {
                let at = 9 + i * 4;
                cycle.push(i32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()));
            }
            Ok(ResidualModel::Periodic { cycle })
        }
        _ => Err(Error::malformed("invalid hypothesis model")),
    }
}

/// Physical content identity of a represented literal (canonical container).
pub fn literal_content_id(rl: &RepresentedLiteral) -> Result<ContentId> {
    Ok(ContentId(Sha256::digest(&literal_container_bytes(rl)?)))
}

/// Physical content identity of a represented residual.
pub fn residual_content_id(rr: &RepresentedResidual) -> Result<ContentId> {
    Ok(ContentId(Sha256::digest(&residual_container_bytes(rr)?)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit_descriptor(frames: u64, layout: Layout) -> ObjectDescriptor {
        ObjectDescriptor::new(
            crate::object::descriptor::Representation::Literal,
            frames,
            layout,
            None,
        )
        .unwrap()
    }

    fn lcg(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s
        }
    }

    fn tone(frames: usize, ch: usize) -> Vec<i32> {
        // Deterministic tonal content: saw + harmonic wobble + small noise.
        // Compresses well under delta/zigzag symbolizations.
        let mut r = lcg(99);
        (0..frames * ch)
            .map(|k| {
                let f = k / ch;
                let mut v: i64 = (((f as i64) * 7) % 2047) - 1023;
                v *= 8192;
                let wobble = (((f as i64) * 31) % 127) - 63;
                v += wobble * 33;
                if k % ch == 1 {
                    v = -v;
                }
                let jitter = ((r() >> 33) as i64) % 65 - 32;
                (v + jitter) as i32
            })
            .collect()
    }

    fn noise(frames: usize, ch: usize) -> Vec<i32> {
        let mut r = lcg(1234);
        (0..frames * ch).map(|_| (r() >> 32) as i32).collect()
    }

    #[test]
    fn literal_roundtrip_partial_equals_full_slice() {
        for &(frames, ch) in &[(2048usize, 1usize), (5000, 2), (512, 2)] {
            for &sym in &[
                Symbolization::Identity,
                Symbolization::Lane4Plain,
                Symbolization::Lane4ZigZag,
                Symbolization::DeltaLane4,
            ] {
                let samples = tone(frames, ch);
                let d = lit_descriptor(
                    frames as u64,
                    if ch == 1 {
                        Layout::Mono
                    } else {
                        Layout::Stereo
                    },
                );
                let rl =
                    RepresentedLiteral::encode(d, &samples, 512, sym, ModelMode::Inline, false)
                        .expect("encode");
                let full = rl.materialize_full().unwrap();
                assert_eq!(full, samples, "full materialization exact for {sym:?}");
                for (start, len) in [
                    (0u64, 512u32),
                    (300, 1000),
                    (511, 2),
                    (4999, 1),
                    (64, 33),
                    (0, 2048),
                ] {
                    if start + u64::from(len) > frames as u64 {
                        continue;
                    }
                    let part = rl.materialize(start, len).unwrap();
                    let lo = start as usize * ch;
                    let hi = (start as usize + len as usize) * ch;
                    assert_eq!(part, full[lo..hi], "partial == full slice");
                }
            }
        }
    }

    #[test]
    fn literal_container_roundtrip_and_identity() {
        let frames = 4096usize;
        let ch = 2usize;
        let samples = tone(frames, ch);
        let d = lit_descriptor(frames as u64, Layout::Stereo);
        let rl = RepresentedLiteral::encode(
            d.clone(),
            &samples,
            512,
            Symbolization::DeltaLane4,
            ModelMode::Shared,
            true,
        )
        .expect("encode");
        let bytes = literal_container_bytes(&rl).unwrap();
        assert!(bytes.starts_with(FORMAT_TAG));
        // Deterministic serialization.
        assert_eq!(bytes, literal_container_bytes(&rl).unwrap());
        // Parse round-trip preserves samples.
        let parsed = parse_literal_container(&bytes).unwrap();
        assert_eq!(parsed.materialize_full().unwrap(), samples);
        assert_eq!(parsed.model_mode, ModelMode::Shared);
        assert!(!parsed.pool.is_empty());
        // Content identity is the hash of canonical bytes.
        let id = literal_content_id(&rl).unwrap();
        assert_eq!(ContentId(Sha256::digest(&bytes)), id);
    }

    #[test]
    fn literal_identity_is_semantically_stable() {
        let frames = 3000usize;
        let samples = tone(frames, 1);
        let d = lit_descriptor(frames as u64, Layout::Mono);
        let a = RepresentedLiteral::encode(
            d.clone(),
            &samples,
            256,
            Symbolization::Lane4Plain,
            ModelMode::Inline,
            false,
        )
        .unwrap();
        let b = RepresentedLiteral::encode(
            d.clone(),
            &samples,
            512,
            Symbolization::Lane4Plain,
            ModelMode::Inline,
            false,
        )
        .unwrap();
        assert_eq!(a.materialize_full().unwrap(), b.materialize_full().unwrap());
    }

    #[test]
    fn high_entropy_literal_falls_back_to_raw_pages() {
        let samples = noise(4096, 2);
        let d = lit_descriptor(4096, Layout::Stereo);
        let rl = RepresentedLiteral::encode(
            d,
            &samples,
            512,
            Symbolization::DeltaLane4,
            ModelMode::Inline,
            false,
        )
        .expect("encode");
        let raw_pages = rl.pages.iter().filter(|p| p.kind == PageKind::Raw).count();
        assert_eq!(
            raw_pages,
            rl.pages.len(),
            "all pages fall back to RAW (H.2.31 negative control)"
        );
        assert_eq!(rl.materialize_full().unwrap(), samples);
    }

    #[test]
    fn residual_roundtrip_exact_and_partial() {
        let frames = 4096usize;
        let cycle: Vec<i32> = (0..64).map(|i| i << 20).collect();
        let model = ResidualModel::Periodic { cycle };
        let mut intrinsic = vec![0i32; frames];
        #[allow(clippy::needless_range_loop)]
        for f in 0..frames {
            intrinsic[f] = model.model_sample(f as u64);
        }
        let mut rng = lcg(555);
        for _ in 0..200 {
            let f = (rng() % frames as u64) as usize;
            let delta = (((rng() >> 33) as i64) % 20000 - 10000) as i32;
            intrinsic[f] =
                crate::universe::arithmetic::sat_i32(i64::from(intrinsic[f]) + i64::from(delta));
        }
        let d = ObjectDescriptor::new(
            crate::object::descriptor::Representation::PredictorResidual,
            frames as u64,
            Layout::Mono,
            None,
        )
        .unwrap();
        let records = Residual::closing_residual(&intrinsic, 1, &model).unwrap();
        let residual = Residual::new(&d, model, records).unwrap();
        for mode in [ModelMode::Inline, ModelMode::Shared] {
            let rr = RepresentedResidual::encode(d.clone(), &residual, 512, mode, false).unwrap();
            let back = rr.reconstruct_full().unwrap();
            assert_eq!(back.model, residual.model);
            assert_eq!(
                back.records, residual.records,
                "byte-identical records ({mode:?})"
            );
            let closed = rr.materialize_closure(0, frames as u32).unwrap();
            assert_eq!(closed, intrinsic);
            let part = rr.materialize_closure(1000, 700).unwrap();
            assert_eq!(part, intrinsic[1000..1700]);
            // Container round-trip.
            let bytes = residual_container_bytes(&rr).unwrap();
            let parsed = parse_residual_container(&bytes).unwrap();
            assert_eq!(parsed.reconstruct_full().unwrap().records, residual.records);
        }
    }

    #[test]
    fn residual_container_hostile_parses_fail_typed() {
        // A valid container, then truncations at every boundary must fail
        // typed (never panic).
        let frames = 2048usize;
        let model = ResidualModel::Constant(10);
        let d = ObjectDescriptor::new(
            crate::object::descriptor::Representation::PredictorResidual,
            frames as u64,
            Layout::Mono,
            None,
        )
        .unwrap();
        let records = vec![ResidualRecord {
            frame: 7,
            channel: 0,
            delta: 3,
        }];
        let residual = Residual::new(&d, model, records).unwrap();
        let rr = RepresentedResidual::encode(d, &residual, 512, ModelMode::Inline, true).unwrap();
        let bytes = residual_container_bytes(&rr).unwrap();
        for cut in 0..bytes.len() {
            let _ = parse_residual_container(&bytes[..cut]);
        }
        let lit = literal_container_bytes(&RepresentedLiteral {
            descriptor: lit_descriptor(512, Layout::Mono),
            page_frames: 512,
            symbolization: Symbolization::Lane4Plain,
            model_mode: ModelMode::Inline,
            pool: Vec::new(),
            pages: Vec::new(),
        })
        .unwrap();
        assert!(parse_literal_container(&lit).is_ok());
    }

    #[test]
    fn residual_page_records_respect_bounds() {
        assert_eq!(parse_raw_records(&[0u8; 4]).unwrap().len(), 0);
        let bad = vec![5u8, 0, 0, 0, 1];
        assert!(parse_raw_records(&bad).is_err());
    }

    #[test]
    fn cost_ledger_counts_models_and_payloads() {
        let frames = 2048usize;
        let samples = tone(frames, 1);
        let d = lit_descriptor(frames as u64, Layout::Mono);
        let canonical = 46 + 8 + frames * 4;
        // Delta/zigzag symbolization of tonal content must beat RAW bytes on
        // complete cost under both model policies.
        let mut inline_cost = None;
        for mode in [ModelMode::Inline, ModelMode::Shared] {
            let rl = RepresentedLiteral::encode(
                d.clone(),
                &samples,
                512,
                Symbolization::DeltaLane4,
                mode,
                false,
            )
            .unwrap();
            let c = rl.cost(canonical as u64).unwrap();
            assert!(c.complete_bytes > 0);
            assert_eq!(c.raw_sample_bytes, (frames * 4) as u64);
            assert!(
                c.complete_bytes < c.raw_sample_bytes,
                "tonal content must compress ({mode:?})"
            );
            assert!(c.model_bytes > 0, "model bytes are always counted");
            if mode == ModelMode::Inline {
                inline_cost = Some(c.model_bytes);
            } else {
                // Shared models across pages cost no more than inline models.
                assert!(
                    c.model_bytes <= inline_cost.unwrap(),
                    "shared pool must not exceed per-page inline models"
                );
            }
        }
    }

    #[test]
    fn pages_touching_is_bounded() {
        let samples = tone(4096, 1);
        let d = lit_descriptor(4096, Layout::Mono);
        let rl = RepresentedLiteral::encode(
            d,
            &samples,
            512,
            Symbolization::DeltaLane4,
            ModelMode::Inline,
            false,
        )
        .unwrap();
        // A 256-frame window spans at most two 512-frame pages.
        for start in [0u64, 500, 511, 512, 513, 4000] {
            assert!(rl.pages_touching(start, 256).len() <= 2);
        }
        // Exact expectations at aligned/misaligned starts.
        assert_eq!(rl.pages_touching(0, 256).len(), 1);
        assert_eq!(rl.pages_touching(500, 256).len(), 2); // crosses 512
        assert_eq!(rl.pages_touching(512, 256).len(), 1);
    }
}
