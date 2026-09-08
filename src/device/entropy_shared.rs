//! Shared flat entropy decode (Phase H.2) — `no_std`-clean device surface.
//!
//! The GPU entropy decoder must reproduce scalar decoded symbols exactly
//! (H.2.17). This module is the *one* decode implementation: host parity
//! tests, the CPU page-parallel path, and the CUDA kernels call the same
//! functions over flat plain-data records, so exactness is by construction.
//!
//! Decode model: one **page** per worker. A page is either
//!
//! * a literal sample page (`MODE_LITERAL`): RAW pages copy their LE sample
//!   bytes straight to the output; RANS pages decode their per-symbolization
//!   streams into scratch and reconstruct interleaved samples;
//! * a residual closure page (`MODE_RESIDUAL`): RAW pages carry per-page
//!   canonical records; RANS pages carry channel masks + 4 zigzag delta
//!   lanes; both reconstruct the exact closure `sat_i32(H + R)` for the page
//!   (frames x channels interleaved).
//!
//! Every offset is bounds-checked against the actual slice lengths (hostile
//! inputs yield `false`, never out-of-bounds access, never panic).

use crate::entropy::rans::{FwdReader, SCALE_BITS, STATE_L, dec_advance, dec_init, dec_slot};
use crate::entropy::transform;
use crate::limits::{MAX_CHANNELS, MAX_ENTROPY_PAGE_FRAMES};

pub const PAGE_RANS: u8 = 0;
pub const PAGE_RAW: u8 = 1;
pub const STREAM_RANS: u8 = 0;
pub const STREAM_RAW: u8 = 1;
pub const MODE_LITERAL: u8 = 0;
pub const MODE_RESIDUAL: u8 = 1;
pub const HYP_ZERO: u8 = 0;
pub const HYP_CONSTANT: u8 = 1;
pub const HYP_PERIODIC: u8 = 2;

/// One coded stream inside a page (24 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct FlatStream {
    /// Model pool index (RANS streams only).
    pub model: u32,
    /// Number of coded symbols (decoded byte length).
    pub symbol_count: u32,
    /// `STREAM_RANS` or `STREAM_RAW`.
    pub kind: u8,
    pub _pad: [u8; 3],
    /// Byte offset into the payload arena (encoded bytes or raw symbols).
    pub payload_off: u32,
    /// Encoded/raw byte length in the payload arena.
    pub payload_len: u32,
    /// Byte offset into the per-page scratch region (decoded symbols).
    pub scratch_off: u32,
}

/// One decode page record (48 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct FlatPage {
    pub frames: u32,
    pub channels: u8,
    /// Symbolization code: 1 identity, 2 lane4_plain, 3 lane4_zigzag,
    /// 4 delta_lane4 (see `entropy::transform`).
    pub sym: u8,
    pub kind: u8,
    pub mode: u8,
    pub _pad: [u8; 3],
    /// Index into the streams arena (first stream of this page).
    pub stream_off: u32,
    pub stream_count: u32,
    /// Element offset into the output sample arena (interleaved i32).
    pub out_off: u32,
    /// Byte offset/length of the RAW payload in the payload arena.
    pub payload_off: u32,
    pub payload_len: u32,
    /// Residual hypothesis kind (`HYP_*`); literal pages: `HYP_ZERO`.
    pub hyp_kind: u8,
    pub _pad2: [u8; 3],
    /// Constant hypothesis level (`HYP_CONSTANT`).
    pub hyp_level: i32,
    /// Cycle arena offset + length (`HYP_PERIODIC`).
    pub cycle_off: u32,
    pub cycle_len: u32,
    /// Hypothesis phase at the page start (`page.start % cycle_len`).
    pub hyp_phase: u32,
    /// Scratch byte offset for this page's streams.
    pub scratch_off: u32,
}

/// Concatenated model tables; `ranges[2i]` = offset of model `i`,
/// `ranges[2i+1]` = its alphabet length.
#[derive(Debug, Clone, Copy)]
pub struct FlatModelSet<'a> {
    pub values: &'a [u16],
    pub starts: &'a [u32],
    pub freqs: &'a [u32],
    pub ranges: &'a [u32],
}

/// Model lookup by slot: binary search over cumulative starts.
#[inline]
pub fn model_slot(models: &FlatModelSet<'_>, model: u32, slot: u32) -> Option<(u32, u32, u8)> {
    let base = usize::try_from(model).ok()?.checked_mul(2)?;
    let off = *models.ranges.get(base)? as usize;
    let len = *models.ranges.get(base + 1)? as usize;
    let starts = models.starts.get(off..off + len)?;
    let idx = starts.partition_point(|&s| s <= slot).checked_sub(1)?;
    let start = *starts.get(idx)?;
    let freq = *models.freqs.get(off + idx)?;
    if slot >= start + freq {
        return None;
    }
    let value = *models.values.get(off + idx)?;
    if value > 255 {
        return None; // symbol alphabet ceiling for byte streams
    }
    Some((start, freq, value as u8))
}

/// Decode one stream's symbols into its scratch region.
pub fn decode_stream(
    st: &FlatStream,
    payload: &[u8],
    models: &FlatModelSet<'_>,
    scratch: &mut [u8],
) -> bool {
    let count = st.symbol_count as usize;
    let Some(dst) = scratch.get_mut(st.scratch_off as usize..(st.scratch_off as usize) + count)
    else {
        return false;
    };
    let src_end = (st.payload_off as usize).saturating_add(st.payload_len as usize);
    let Some(src) = payload.get(st.payload_off as usize..src_end) else {
        return false;
    };
    match st.kind {
        STREAM_RAW => {
            if src.len() != count {
                return false;
            }
            dst.copy_from_slice(src);
            true
        }
        STREAM_RANS => {
            let Some(base0) = usize::try_from(st.model).ok() else {
                return false;
            };
            let Some(base) = base0.checked_mul(2) else {
                return false;
            };
            if models.ranges.get(base + 1).map_or(0, |&l| l as usize) == 0 {
                return false;
            }
            let mut reader = FwdReader::new(src);
            let Some(mut state) = dec_init(&mut reader) else {
                return false;
            };
            for i in (0..count).rev() {
                let slot = dec_slot(&state, SCALE_BITS);
                let Some((start, freq, value)) = model_slot(models, st.model, slot) else {
                    return false;
                };
                if !dec_advance(&mut state, &mut reader, start, freq, SCALE_BITS) {
                    return false;
                }
                dst[i] = value;
            }
            // Canonical terminal condition (RANS.md decode step 3): after the
            // final symbol the state must be exactly RANS_STATE_L and the byte
            // cursor exactly at the end of this stream's encoded payload.
            // Trailing garbage, altered states, and truncations are rejected
            // identically on host and device.
            if state.0 != STATE_L || reader.bytes_consumed() != src.len() {
                return false;
            }
            true
        }
        _ => false,
    }
}

/// Hypothesis sample (code domain) for a residual page at page-local frame.
fn hypothesis_sample(page: &FlatPage, cycle: &[i32], f: usize) -> Option<i32> {
    match page.hyp_kind {
        HYP_ZERO => Some(0),
        HYP_CONSTANT => Some(page.hyp_level),
        HYP_PERIODIC => {
            if page.cycle_len == 0 {
                return None;
            }
            let idx = page.cycle_off as usize
                + ((u64::from(page.hyp_phase) + f as u64) % u64::from(page.cycle_len)) as usize;
            cycle.get(idx).copied()
        }
        _ => None,
    }
}

/// Decode one page into the output sample arena.
pub fn decode_page(
    page: &FlatPage,
    streams: &[FlatStream],
    payload: &[u8],
    models: &FlatModelSet<'_>,
    cycle: &[i32],
    out: &mut [i32],
    scratch: &mut [u8],
) -> bool {
    if page.frames == 0
        || page.frames > MAX_ENTROPY_PAGE_FRAMES
        || page.channels == 0
        || u32::from(page.channels) > MAX_CHANNELS
    {
        return false;
    }
    let ch = usize::from(page.channels);
    let slots = page.frames as usize * ch;
    let out_begin = page.out_off as usize;
    let Some(out_end) = out_begin.checked_add(slots) else {
        return false;
    };
    let Some(out_slice) = out.get_mut(out_begin..out_end) else {
        return false;
    };
    match page.mode {
        MODE_LITERAL => match page.kind {
            PAGE_RAW => {
                // RAW literal payload = LE sample bytes.
                let want = slots * 4;
                let payload_begin = page.payload_off as usize;
                let src = match payload.get(payload_begin..payload_begin + want) {
                    Some(s) => s,
                    None => return false,
                };
                for (i, s) in out_slice.iter_mut().enumerate() {
                    *s = i32::from_le_bytes([
                        src[i * 4],
                        src[i * 4 + 1],
                        src[i * 4 + 2],
                        src[i * 4 + 3],
                    ]);
                }
                true
            }
            PAGE_RANS => {
                let sbegin = page.stream_off as usize;
                let scount = page.stream_count as usize;
                let expect = match page.sym {
                    1 => 1,
                    2..=4 => 4,
                    _ => return false,
                };
                if scount != expect {
                    return false;
                }
                for i in 0..scount {
                    let st = match streams.get(sbegin + i) {
                        Some(s) => s,
                        None => return false,
                    };
                    if !decode_stream(st, payload, models, scratch) {
                        return false;
                    }
                }
                // Reborrow the decoded scratch regions (immutable) and route
                // through the shared slice transform.
                let mut lanes = [&[][..]; 4];
                for (i, lane) in lanes.iter_mut().enumerate().take(scount) {
                    let st = match streams.get(sbegin + i) {
                        Some(s) => s,
                        None => return false,
                    };
                    let len = st.symbol_count as usize;
                    let Some(bytes) =
                        scratch.get(st.scratch_off as usize..(st.scratch_off as usize) + len)
                    else {
                        return false;
                    };
                    *lane = bytes;
                }
                // Identity stream length is in bytes (4 per sample); lane
                // streams are one byte per sample.
                let per_lane = if page.sym == 1 { slots * 4 } else { slots };
                if lanes.iter().take(scount).any(|l| l.len() != per_lane) {
                    return false;
                }
                transform::samples_from_lanes(page.sym, lanes, ch, out_slice)
            }
            _ => false,
        },
        MODE_RESIDUAL => {
            decode_residual_closure_page(page, streams, payload, models, cycle, out_slice, scratch)
        }
        _ => false,
    }
}

/// Residual closure reconstruction for one page (see module docs).
fn decode_residual_closure_page(
    page: &FlatPage,
    streams: &[FlatStream],
    payload: &[u8],
    models: &FlatModelSet<'_>,
    cycle: &[i32],
    out: &mut [i32],
    scratch: &mut [u8],
) -> bool {
    let ch = usize::from(page.channels);
    let frames = page.frames as usize;
    // Pass 1: pure hypothesis (channel 0 only; other channels are 0).
    for f in 0..frames {
        let Some(h) = hypothesis_sample(page, cycle, f) else {
            return false;
        };
        out[f * ch] = h;
    }
    match page.kind {
        PAGE_RAW => {
            // Canonical page records: count u32 LE + 13 B/record with
            // page-local frames.
            let begin = page.payload_off as usize;
            let Some(end) = begin.checked_add(page.payload_len as usize) else {
                return false;
            };
            let Some(records) = payload.get(begin..end) else {
                return false;
            };
            if records.len() < 4 {
                return false;
            }
            let count =
                u32::from_le_bytes([records[0], records[1], records[2], records[3]]) as usize;
            if records.len() != 4 + count * 13 {
                return false;
            }
            for i in 0..count {
                let at = 4 + i * 13;
                let frame = u64::from_le_bytes(records[at..at + 8].try_into().unwrap_or([0u8; 8]));
                let chan = records[at + 8];
                let delta =
                    i32::from_le_bytes(records[at + 9..at + 13].try_into().unwrap_or([0u8; 4]));
                let f = usize::try_from(frame).unwrap_or(usize::MAX);
                if f >= frames || usize::from(chan) >= ch {
                    return false;
                }
                let slot = f * ch + usize::from(chan);
                let h = i64::from(out[slot]);
                out[slot] = crate::universe::arithmetic::sat_i32(h + i64::from(delta));
            }
            true
        }
        PAGE_RANS => {
            let sbegin = page.stream_off as usize;
            let scount = page.stream_count as usize;
            if scount != ch + 4 {
                return false;
            }
            // Decode all streams (channel masks then 4 delta lanes).
            for i in 0..scount {
                let st = match streams.get(sbegin + i) {
                    Some(s) => s,
                    None => return false,
                };
                if !decode_stream(st, payload, models, scratch) {
                    return false;
                }
            }
            let mask_bytes = frames.div_ceil(8);
            let mut cursor = 0usize;
            for c in 0..ch {
                let mst = match streams.get(sbegin + c) {
                    Some(s) => s,
                    None => return false,
                };
                let mask = match scratch.get(
                    mst.scratch_off as usize
                        ..(mst.scratch_off as usize) + mst.symbol_count as usize,
                ) {
                    Some(m) => m,
                    None => return false,
                };
                if mask.len() != mask_bytes {
                    return false;
                }
                for f in 0..frames {
                    if (mask[f >> 3] >> (f & 7)) & 1 == 0 {
                        continue;
                    }
                    let mut b = [0u8; 4];
                    // p indexes the four byte-position lanes at `cursor`.
                    #[allow(clippy::needless_range_loop)]
                    for p in 0..4 {
                        let lane = match streams.get(sbegin + ch + p) {
                            Some(l) => l,
                            None => return false,
                        };
                        let bytes = match scratch.get(
                            lane.scratch_off as usize
                                ..(lane.scratch_off as usize) + lane.symbol_count as usize,
                        ) {
                            Some(x) => x,
                            None => return false,
                        };
                        match bytes.get(cursor) {
                            Some(&v) => b[p] = v,
                            None => return false,
                        }
                    }
                    let delta = transform::unzigzag(u32::from_le_bytes(b));
                    let slot = f * ch + c;
                    let h = i64::from(out[slot]);
                    out[slot] = crate::universe::arithmetic::sat_i32(h + i64::from(delta));
                    cursor += 1;
                }
            }
            // Cursor must equal the delta lane length (hostile mismatch).
            let Some(d0) = streams.get(sbegin + ch) else {
                return false;
            };
            cursor == d0.symbol_count as usize
        }
        _ => false,
    }
}

/// Convenience: decode all pages of a flat job sequentially (host parity and
/// CPU reference path; device kernels drive `decode_page` per thread).
///
/// `page_range` is a half-open index range over `pages`. Each page writes to
/// `out` at its `out_off`. `scratch` must be at least as large as the largest
/// per-page scratch need (the caller's per-window scratch buffer).
#[allow(clippy::too_many_arguments)]
pub fn decode_pages(
    pages: &[FlatPage],
    streams: &[FlatStream],
    payload: &[u8],
    models: &FlatModelSet<'_>,
    cycle: &[i32],
    out: &mut [i32],
    scratch: &mut [u8],
    page_range: core::ops::Range<usize>,
) -> bool {
    for i in page_range {
        let Some(page) = pages.get(i) else {
            return false;
        };
        if !decode_page(page, streams, payload, models, cycle, out, scratch) {
            return false;
        }
    }
    true
}

/// Decode-job descriptor: arena lengths and geometry the kernel needs to
/// rebuild slices from raw pointers (mirror of the host flat job).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct EntropyJobDesc {
    pub page_count: u32,
    /// Output arena length in i32 elements.
    pub arena_samples: u32,
    /// Per-page scratch stride in bytes.
    pub scratch_stride: u32,
    pub payload_len: u32,
    pub stream_count: u32,
    pub mvalue_count: u32,
    pub mstart_count: u32,
    pub mfreq_count: u32,
    pub mrange_count: u32,
    pub cycle_count: u32,
    /// Pages are decoded one per thread; status[i] = 1 on success.
    pub statuses_len: u32,
    pub _pad: [u8; 8],
}

impl EntropyJobDesc {
    /// A zeroed descriptor (host fills it from a flat job).
    pub const fn zeroed() -> EntropyJobDesc {
        EntropyJobDesc {
            page_count: 0,
            arena_samples: 0,
            scratch_stride: 0,
            payload_len: 0,
            stream_count: 0,
            mvalue_count: 0,
            mstart_count: 0,
            mfreq_count: 0,
            mrange_count: 0,
            cycle_count: 0,
            statuses_len: 0,
            _pad: [0; 8],
        }
    }

    /// Decode page `page_id` against the raw arenas described by this
    /// descriptor. Mirrors the device kernel exactly; `false` on any
    /// malformed condition.
    ///
    /// # Safety
    /// Every pointer must reference an allocation of at least the length
    /// declared in this descriptor (bytes for byte arenas, elements for the
    /// i32/u16 arenas); `scratch` must have `scratch_stride` bytes available
    /// at `page_id * scratch_stride`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn decode_page_raw(
        &self,
        page_id: u32,
        pages: *const FlatPage,
        streams: *const FlatStream,
        payload: *const u8,
        mvalues: *const u16,
        mstarts: *const u32,
        mfreqs: *const u32,
        mranges: *const u32,
        cycle: *const i32,
        out: *mut i32,
        scratch: *mut u8,
    ) -> bool {
        // SAFETY of the caller contract: all pointers reference allocations
        // of at least the lengths declared in this descriptor. Every slice
        // below is built against exactly those declared lengths; `decode_page`
        // additionally bounds-checks all offsets before touching them.
        unsafe {
            let n_pages = self.page_count as usize;
            if (page_id as usize) >= n_pages {
                return false;
            }
            let pages_slice = core::slice::from_raw_parts(pages, n_pages);
            let streams_slice = core::slice::from_raw_parts(streams, self.stream_count as usize);
            let payload_slice = core::slice::from_raw_parts(payload, self.payload_len as usize);
            let models = FlatModelSet {
                values: core::slice::from_raw_parts(mvalues, self.mvalue_count as usize),
                starts: core::slice::from_raw_parts(mstarts, self.mstart_count as usize),
                freqs: core::slice::from_raw_parts(mfreqs, self.mfreq_count as usize),
                ranges: core::slice::from_raw_parts(mranges, self.mrange_count as usize),
            };
            let cycle_slice = core::slice::from_raw_parts(cycle, self.cycle_count as usize);
            let out_slice = core::slice::from_raw_parts_mut(out, self.arena_samples as usize);
            let stride = self.scratch_stride as usize;
            let scratch_slice = core::slice::from_raw_parts_mut(
                scratch.add((page_id as usize).saturating_mul(stride)),
                stride,
            );
            decode_page(
                &pages_slice[page_id as usize],
                streams_slice,
                payload_slice,
                &models,
                cycle_slice,
                out_slice,
                scratch_slice,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_sizes_are_stable() {
        // Flat records are upload-identical across host and device builds.
        assert_eq!(core::mem::size_of::<FlatPage>(), 56);
        assert_eq!(core::mem::size_of::<FlatStream>(), 24);
        assert_eq!(core::mem::size_of::<EntropyJobDesc>(), 52);
    }

    #[test]
    fn model_slot_rejects_bad_models() {
        let m = FlatModelSet {
            values: &[3, 7],
            starts: &[0, 8192],
            freqs: &[8192, 8192],
            ranges: &[0, 2],
        };
        assert_eq!(model_slot(&m, 0, 100), Some((0, 8192, 3)));
        assert_eq!(model_slot(&m, 0, 9000), Some((8192, 8192, 7)));
        // Model index out of range / value beyond byte alphabet.
        assert!(model_slot(&m, 1, 0).is_none());
        let wide = FlatModelSet {
            values: &[300],
            starts: &[0],
            freqs: &[16384],
            ranges: &[0, 1],
        };
        assert!(model_slot(&wide, 0, 0).is_none());
    }
}
