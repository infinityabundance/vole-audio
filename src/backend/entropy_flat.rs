//! Host-side flat entropy job builder (Phase H.2) — backend-neutral.
//!
//! Converts the in-memory entropy representations (`RepresentedLiteral`,
//! `RepresentedResidual`) into the flat plain-data jobs the shared decoder
//! (`device::entropy_shared`) and the CUDA kernels consume, and drives the
//! host parity path (`decode_*_host`) that must equal the representation
//! decode bit for bit.
//!
//! A job covers a **page range** (usually one observation window plus its
//! interpolation halo). Output geometry: the job's `range_start_frame`
//! anchors `out`; page `i` writes its samples at
//! `(page.start_frame - range_start) * channels` elements, so a window
//! aligned to the first page start has all offsets nonnegative.

use crate::device::entropy_shared::{
    FlatModelSet, FlatPage, FlatStream, HYP_CONSTANT, HYP_PERIODIC, HYP_ZERO, MODE_LITERAL,
    MODE_RESIDUAL, PAGE_RANS, PAGE_RAW, STREAM_RANS, STREAM_RAW,
};
use crate::entropy::block::BlockPayload;
use crate::entropy::model::SymbolModel;
use crate::entropy::represent::{ModelMode, RepresentedLiteral, RepresentedResidual};
use crate::entropy::symbol::Symbolization;
use crate::error::{Error, Result};
use crate::object::residual::ResidualModel;

/// One flat entropy decode job.
#[derive(Debug, Clone, Default)]
pub struct FlatEntropyJob {
    pub pages: Vec<FlatPage>,
    pub streams: Vec<FlatStream>,
    pub payload: Vec<u8>,
    // --- concatenated model tables (see FlatModelSet) ---
    pub model_values: Vec<u16>,
    pub model_starts: Vec<u32>,
    pub model_freqs: Vec<u32>,
    pub model_ranges: Vec<u32>,
    // --- residual hypothesis cycle arena ---
    pub cycle: Vec<i32>,
    /// First frame covered by the job (anchor of `out`).
    pub range_start_frame: u64,
    /// Output arena length in samples (frames x channels).
    pub arena_samples: usize,
    /// Largest per-page scratch need (bytes).
    pub max_page_scratch: usize,
    /// Element stride (channels).
    pub channels: u8,
    /// Decoded-sample arena length in frames (pages' span).
    pub arena_frames: u32,
}

impl FlatEntropyJob {
    pub fn model_set(&self) -> FlatModelSet<'_> {
        FlatModelSet {
            values: &self.model_values,
            starts: &self.model_starts,
            freqs: &self.model_freqs,
            ranges: &self.model_ranges,
        }
    }

    /// Scratch stride per page when parallel decoding (device).
    pub fn scratch_stride(&self) -> usize {
        self.max_page_scratch.next_power_of_two().max(64)
    }

    /// Host sequential decode of pages `[begin..end)`; writes into `out`
    /// (length `arena_samples`). Scratch must be `>= max_page_scratch`.
    pub fn decode_pages_host(
        &self,
        out: &mut [i32],
        scratch: &mut [u8],
        range: std::ops::Range<usize>,
    ) -> bool {
        if out.len() < self.arena_samples {
            return false;
        }
        crate::device::entropy_shared::decode_pages(
            &self.pages,
            &self.streams,
            &self.payload,
            &self.model_set(),
            &self.cycle,
            out,
            scratch,
            range,
        )
    }
}

/// Insert a model into the flat tables; returns its pool index.
fn insert_model(
    job: &mut FlatEntropyJob,
    model: &SymbolModel,
    dedupe: &mut std::collections::HashMap<Vec<u8>, u32>,
) -> u32 {
    let key = model.canonical_bytes();
    if let Some(&idx) = dedupe.get(&key) {
        return idx;
    }
    let off = job.model_values.len() as u32;
    for &v in &model.symbols {
        job.model_values.push(v);
    }
    for &s in &model.start {
        job.model_starts.push(s);
    }
    for &f in &model.freq {
        job.model_freqs.push(f);
    }
    let idx = (job.model_ranges.len() / 2) as u32;
    job.model_ranges.push(off);
    job.model_ranges.push(model.len() as u32);
    dedupe.insert(key, idx);
    idx
}

fn page_sym_code(sym: Symbolization) -> u8 {
    sym.code()
}

/// Flatten a literal representation range `[start_frame, start_frame +
/// frames)` onto a page-aligned arena (pages intersecting the range). The
/// arena's first frame is the first intersecting page's start frame.
pub fn flatten_literal_range(
    rl: &RepresentedLiteral,
    start_frame: u64,
    frames: u32,
) -> Result<FlatEntropyJob> {
    if frames == 0 {
        return Err(Error::malformed("empty range"));
    }
    let channels = rl.descriptor.layout.count();
    let ch = usize::from(channels);
    let req_end = start_frame + u64::from(frames);
    let mut job = FlatEntropyJob {
        channels,
        ..Default::default()
    };
    let mut dedupe: std::collections::HashMap<Vec<u8>, u32> = std::collections::HashMap::new();
    // Shared-mode pool is flattened first (its model order is canonical).
    if rl.model_mode == ModelMode::Shared {
        for m in &rl.pool {
            insert_model(&mut job, m, &mut dedupe);
        }
    }
    let mut range_start = u64::MAX;
    for page in &rl.pages {
        let pe = page.start_frame + u64::from(page.frames);
        if pe <= start_frame || page.start_frame >= req_end {
            continue;
        }
        range_start = range_start.min(page.start_frame);
    }
    if range_start == u64::MAX {
        return Err(Error::malformed("range outside object"));
    }
    job.range_start_frame = range_start;

    let mut max_page_scratch = 0usize;
    let mut arena_frames_end = 0u64;
    for page in &rl.pages {
        let pe = page.start_frame + u64::from(page.frames);
        if pe <= start_frame || page.start_frame >= req_end {
            continue;
        }
        arena_frames_end = arena_frames_end.max(pe);
        let mut fp = FlatPage {
            frames: page.frames,
            channels,
            sym: page_sym_code(rl.symbolization),
            kind: if page.kind == crate::entropy::represent::PageKind::Rans {
                PAGE_RANS
            } else {
                PAGE_RAW
            },
            mode: MODE_LITERAL,
            _pad: [0; 3],
            stream_off: job.streams.len() as u32,
            stream_count: page.blocks.len() as u32,
            out_off: ((page.start_frame - range_start) as usize * ch) as u32,
            payload_off: job.payload.len() as u32,
            payload_len: 0,
            hyp_kind: HYP_ZERO,
            _pad2: [0; 3],
            hyp_level: 0,
            cycle_off: 0,
            cycle_len: 0,
            hyp_phase: 0,
            scratch_off: 0,
        };
        match page.kind {
            crate::entropy::represent::PageKind::Rans => {
                let mut scratch = 0usize;
                for b in &page.blocks {
                    let (kind, model_idx, symbol_count, payload_bytes) = match &b.payload {
                        BlockPayload::Rans {
                            model,
                            symbol_count,
                            bytes,
                        } => {
                            let idx = match model {
                                crate::entropy::block::ModelRef::Inline(m) => {
                                    insert_model(&mut job, m, &mut dedupe)
                                }
                                crate::entropy::block::ModelRef::Shared(i) => *i,
                            };
                            (STREAM_RANS, idx, *symbol_count, bytes.clone())
                        }
                        BlockPayload::Raw { bytes } => {
                            (STREAM_RAW, 0, bytes.len() as u64, bytes.clone())
                        }
                    };
                    let fs = FlatStream {
                        model: model_idx,
                        symbol_count: symbol_count as u32,
                        kind,
                        _pad: [0; 3],
                        payload_off: job.payload.len() as u32,
                        payload_len: payload_bytes.len() as u32,
                        scratch_off: scratch as u32,
                    };
                    job.payload.extend_from_slice(&payload_bytes);
                    job.streams.push(fs);
                    scratch += symbol_count as usize;
                }
                max_page_scratch = max_page_scratch.max(scratch);
                // RANS pages keep payload_len 0 (payload referenced by
                // streams).
                fp.payload_len = 0;
            }
            crate::entropy::represent::PageKind::Raw => {
                fp.stream_count = 0;
                fp.payload_len = page.raw.len() as u32;
                job.payload.extend_from_slice(&page.raw);
            }
        }
        job.pages.push(fp);
    }
    job.max_page_scratch = max_page_scratch;
    job.arena_frames = (arena_frames_end - range_start) as u32;
    job.arena_samples = job.arena_frames as usize * ch;
    Ok(job)
}

/// Flatten a residual representation range to closure samples (see
/// `flatten_literal_range` for the geometry).
pub fn flatten_residual_range(
    rr: &RepresentedResidual,
    start_frame: u64,
    frames: u32,
) -> Result<FlatEntropyJob> {
    if frames == 0 {
        return Err(Error::malformed("empty range"));
    }
    let channels = rr.descriptor.layout.count();
    let ch = usize::from(channels);
    let req_end = start_frame + u64::from(frames);
    let mut job = FlatEntropyJob {
        channels,
        ..Default::default()
    };
    let mut dedupe: std::collections::HashMap<Vec<u8>, u32> = std::collections::HashMap::new();
    if rr.model_mode == ModelMode::Shared {
        for m in &rr.pool {
            insert_model(&mut job, m, &mut dedupe);
        }
    }
    // Residual hypothesis flattening.
    let (hyp_kind, hyp_level, cycle_off, cycle_len) = match &rr.model {
        ResidualModel::Zero => (HYP_ZERO, 0i32, 0u32, 0u32),
        ResidualModel::Constant(l) => (HYP_CONSTANT, *l, 0, 0),
        ResidualModel::Periodic { cycle } => {
            let off = job.cycle.len() as u32;
            job.cycle.extend_from_slice(cycle);
            (HYP_PERIODIC, 0, off, cycle.len() as u32)
        }
    };
    let mut range_start = u64::MAX;
    for page in &rr.pages {
        let pe = page.start_frame + u64::from(page.frames);
        if pe <= start_frame || page.start_frame >= req_end {
            continue;
        }
        range_start = range_start.min(page.start_frame);
    }
    if range_start == u64::MAX {
        return Err(Error::malformed("range outside object"));
    }
    job.range_start_frame = range_start;

    let mut max_page_scratch = 0usize;
    let mut arena_frames_end = 0u64;
    for page in &rr.pages {
        let pe = page.start_frame + u64::from(page.frames);
        if pe <= start_frame || page.start_frame >= req_end {
            continue;
        }
        arena_frames_end = arena_frames_end.max(pe);
        let mut fp = FlatPage {
            frames: page.frames,
            channels,
            sym: 1, // residual streams are byte streams
            kind: if page.kind == crate::entropy::represent::PageKind::Rans {
                PAGE_RANS
            } else {
                PAGE_RAW
            },
            mode: MODE_RESIDUAL,
            _pad: [0; 3],
            stream_off: job.streams.len() as u32,
            stream_count: 0,
            out_off: ((page.start_frame - range_start) as usize * ch) as u32,
            payload_off: job.payload.len() as u32,
            payload_len: 0,
            hyp_kind,
            _pad2: [0; 3],
            hyp_level,
            cycle_off,
            cycle_len,
            hyp_phase: if hyp_kind == HYP_PERIODIC {
                (page.start_frame % u64::from(cycle_len.max(1))) as u32
            } else {
                0
            },
            scratch_off: 0,
        };
        match page.kind {
            crate::entropy::represent::PageKind::Rans => {
                let mut scratch = 0usize;
                for b in &page.blocks {
                    let (kind, model_idx, symbol_count, payload_bytes) = match &b.payload {
                        BlockPayload::Rans {
                            model,
                            symbol_count,
                            bytes,
                        } => {
                            let idx = match model {
                                crate::entropy::block::ModelRef::Inline(m) => {
                                    insert_model(&mut job, m, &mut dedupe)
                                }
                                crate::entropy::block::ModelRef::Shared(i) => *i,
                            };
                            (STREAM_RANS, idx, *symbol_count, bytes.clone())
                        }
                        BlockPayload::Raw { bytes } => {
                            (STREAM_RAW, 0, bytes.len() as u64, bytes.clone())
                        }
                    };
                    let fs = FlatStream {
                        model: model_idx,
                        symbol_count: symbol_count as u32,
                        kind,
                        _pad: [0; 3],
                        payload_off: job.payload.len() as u32,
                        payload_len: payload_bytes.len() as u32,
                        scratch_off: scratch as u32,
                    };
                    job.payload.extend_from_slice(&payload_bytes);
                    job.streams.push(fs);
                    scratch += symbol_count as usize;
                }
                max_page_scratch = max_page_scratch.max(scratch);
                fp.payload_len = 0;
            }
            crate::entropy::represent::PageKind::Raw => {
                // RAW residual records are canonical with *absolute* frames;
                // the per-page device decoder expects page-local frames, so
                // rewrite the record payload here (device-internal form).
                let mut rec = Vec::new();
                let count = if page.raw.len() < 4 {
                    return Err(Error::malformed("RAW record page too short"));
                } else {
                    u32::from_le_bytes([page.raw[0], page.raw[1], page.raw[2], page.raw[3]])
                        as usize
                };
                rec.extend_from_slice(&(count as u32).to_le_bytes());
                for i in 0..count {
                    let at = 4 + i * 13;
                    let Some(slot) = page.raw.get(at..at + 13) else {
                        return Err(Error::malformed("RAW record page truncated"));
                    };
                    let abs = u64::from_le_bytes(slot[0..8].try_into().unwrap());
                    let local = abs
                        .checked_sub(page.start_frame)
                        .ok_or_else(|| Error::malformed("RAW record before page start"))?;
                    rec.extend_from_slice(&local.to_le_bytes());
                    rec.push(slot[8]);
                    rec.extend_from_slice(&slot[9..13]);
                }
                fp.payload_len = rec.len() as u32;
                job.payload.extend_from_slice(&rec);
            }
        }
        fp.stream_count = match page.kind {
            crate::entropy::represent::PageKind::Rans => page.blocks.len() as u32,
            crate::entropy::represent::PageKind::Raw => 0,
        };
        job.pages.push(fp);
    }
    job.max_page_scratch = max_page_scratch;
    job.arena_frames = (arena_frames_end - range_start) as u32;
    job.arena_samples = job.arena_frames as usize * ch;
    Ok(job)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::represent::ModelMode;
    use crate::entropy::represent::RepresentedLiteral;
    use crate::entropy::symbol::Symbolization;
    use crate::object::descriptor::{ObjectDescriptor, Representation};
    use crate::universe::layout::Layout;

    fn tone(frames: usize, ch: usize) -> Vec<i32> {
        let mut s = 0x1234_5678u64;
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
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                v + ((s >> 40) as i64 % 65) - 32
            })
            .collect::<Vec<i64>>()
            .into_iter()
            .map(|x| x.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32)
            .collect()
    }

    fn noise(frames: usize, ch: usize) -> Vec<i32> {
        let mut s = 0xdead_beefu64;
        (0..frames * ch)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s as u32) as i32
            })
            .collect()
    }

    fn check_literal(frames: usize, ch: usize, samples: Vec<i32>, sym: Symbolization) {
        let descriptor = ObjectDescriptor::new(
            Representation::Literal,
            frames as u64,
            if ch == 1 {
                Layout::Mono
            } else {
                Layout::Stereo
            },
            None,
        )
        .unwrap();
        let rl =
            RepresentedLiteral::encode(descriptor, &samples, 512, sym, ModelMode::Inline, false)
                .unwrap();
        // Whole-object job: flat decode must equal the representation decode.
        let job = flatten_literal_range(&rl, 0, frames as u32).unwrap();
        let mut out = vec![0i32; job.arena_samples];
        let mut scratch = vec![0u8; job.max_page_scratch.max(1)];
        assert!(
            job.decode_pages_host(&mut out, &mut scratch, 0..job.pages.len()),
            "host flat decode ok"
        );
        // Range-aligned arena: out == full samples (arena starts at frame 0).
        assert_eq!(out, samples, "flat decode == representation decode");
        // Windows: decode a later window and check against the slice.
        if frames > 2000 {
            let start = 1024u64;
            let job2 = flatten_literal_range(&rl, start, 600).unwrap();
            let mut out2 = vec![0i32; job2.arena_samples];
            assert!(
                job2.decode_pages_host(&mut out2, &mut scratch, 0..job2.pages.len()),
                "window decode ok"
            );
            // arena starts at the first intersecting page (>= start).
            let off = (job2.range_start_frame as usize - start as usize) * ch;
            let want_lo = (start as usize) * ch;
            let want_hi = (start as usize + 600) * ch;
            let got_lo = off;
            let got_hi = off + 600 * ch;
            assert_eq!(&out2[got_lo..got_hi], &samples[want_lo..want_hi]);
        }
    }

    #[test]
    fn literal_flat_parity_all_symbolizations() {
        for &sym in &[
            Symbolization::Identity,
            Symbolization::Lane4Plain,
            Symbolization::Lane4ZigZag,
            Symbolization::DeltaLane4,
        ] {
            check_literal(2048, 1, tone(2048, 1), sym);
            check_literal(5000, 2, tone(5000, 2), sym);
        }
    }

    #[test]
    fn literal_flat_parity_with_raw_fallback() {
        // Noise forces RAW pages; RAW page decode must equal too.
        check_literal(4096, 2, noise(4096, 2), Symbolization::DeltaLane4);
    }

    #[test]
    fn residual_flat_parity() {
        use crate::entropy::represent::RepresentedResidual;
        use crate::object::residual::{Residual, ResidualModel};
        let frames = 4096usize;
        let cycle: Vec<i32> = (0..64).map(|i| i << 20).collect();
        let model = ResidualModel::Periodic { cycle };
        let mut intrinsic = vec![0i32; frames];
        for (f, slot) in intrinsic.iter_mut().enumerate() {
            *slot = model.model_sample(f as u64);
        }
        let mut rng = 0xabc123u64;
        for _ in 0..300 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f = (rng % frames as u64) as usize;
            let delta = (((rng >> 33) as i64) % 20000) as i32 - 10000;
            intrinsic[f] =
                crate::universe::arithmetic::sat_i32(i64::from(intrinsic[f]) + i64::from(delta));
        }
        let descriptor = ObjectDescriptor::new(
            Representation::PredictorResidual,
            frames as u64,
            Layout::Mono,
            None,
        )
        .unwrap();
        let records = Residual::closing_residual(&intrinsic, 1, &model).unwrap();
        let residual = Residual::new(&descriptor, model, records).unwrap();
        let rr = RepresentedResidual::encode(descriptor, &residual, 512, ModelMode::Inline, false)
            .unwrap();
        let job = flatten_residual_range(&rr, 0, frames as u32).unwrap();
        let mut out = vec![0i32; job.arena_samples];
        let mut scratch = vec![0u8; job.max_page_scratch.max(1)];
        assert!(job.decode_pages_host(&mut out, &mut scratch, 0..job.pages.len()));
        assert_eq!(
            out, intrinsic,
            "flat residual closure == representation closure"
        );
        // Unused import guard for the compiler's dead-code lints in tests.
        let _ = HYP_ZERO;
    }

    /// Deterministic mutational fuzz of the flat decoder (H.2.47): valid
    /// jobs are seeded, then their payload bytes and plain-data records are
    /// mutated (flips / truncations / extensions / splicing) across a seeded
    /// space. The decoder must never panic, never trap, and return only
    /// `false` or bounded output — hostile input fails typed, exactly as the
    /// hostile-input court requires. Runs under `catch_unwind` so any panic
    /// fails the test with the seed recorded.
    #[test]
    fn decoder_mutational_fuzz_never_panics() {
        let mut seed = 0x_0bad_c0de_f00d_d00du64;
        let mut rng = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed
        };
        let jobs = [
            flatten_literal_range(
                &RepresentedLiteral::encode(
                    ObjectDescriptor::new(Representation::Literal, 4096, Layout::Stereo, None)
                        .unwrap(),
                    &tone(4096, 2),
                    512,
                    Symbolization::DeltaLane4,
                    ModelMode::Inline,
                    false,
                )
                .unwrap(),
                0,
                4096,
            )
            .unwrap(),
            flatten_literal_range(
                &RepresentedLiteral::encode(
                    ObjectDescriptor::new(Representation::Literal, 2048, Layout::Stereo, None)
                        .unwrap(),
                    &noise(2048, 2),
                    512,
                    Symbolization::DeltaLane4,
                    ModelMode::Inline,
                    false,
                )
                .unwrap(),
                0,
                2048,
            )
            .unwrap(),
            {
                // Procedural residual with sparse corrections (periodic
                // hypothesis), mono — exercises the residual decoder paths.
                let frames = 4096usize;
                let cycle: Vec<i32> = (0..64).map(|i| ((i as i64 - 32) * 128) as i32).collect();
                let model = crate::object::residual::ResidualModel::Periodic { cycle };
                let mut intrinsic = vec![0i32; frames];
                let mut rs = 0x1234_5678_9abc_def0u64;
                for (f, slot) in intrinsic.iter_mut().enumerate() {
                    *slot = model.model_sample(f as u64);
                }
                for _ in 0..256 {
                    rs = rs
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let f = (rs % frames as u64) as usize;
                    let delta = (((rs >> 33) as i64) % 2048) - 1024;
                    intrinsic[f] =
                        crate::universe::arithmetic::sat_i32(i64::from(intrinsic[f]) + delta);
                }
                let d = ObjectDescriptor::new(
                    Representation::PredictorResidual,
                    frames as u64,
                    Layout::Mono,
                    None,
                )
                .unwrap();
                let records =
                    crate::object::residual::Residual::closing_residual(&intrinsic, 1, &model)
                        .unwrap();
                let residual = crate::object::residual::Residual::new(&d, model, records).unwrap();
                let rr = crate::entropy::represent::RepresentedResidual::encode(
                    d,
                    &residual,
                    512,
                    ModelMode::Inline,
                    false,
                )
                .unwrap();
                flatten_residual_range(&rr, 0, frames as u32).unwrap()
            },
        ];
        const MUTATIONS: usize = 6000;
        for round in 0..MUTATIONS {
            let job = &jobs[round % jobs.len()];
            let mut mutant = job.clone();
            let r = rng();
            let op = (r >> 32) % 7;
            match op {
                // Byte flips inside the payload.
                0 | 1 => {
                    if !mutant.payload.is_empty() {
                        let n = (r as usize) % mutant.payload.len().min(64) + 1;
                        for _ in 0..n {
                            let at = (rng() as usize) % mutant.payload.len();
                            mutant.payload[at] ^= 1 << ((rng() >> 40) % 8);
                        }
                    }
                }
                // Truncations at assorted boundaries.
                2 => {
                    let cut = (r as usize) % (mutant.payload.len() + 1);
                    mutant.payload.truncate(cut);
                }
                // Extensions with garbage.
                3 => {
                    let extra = (r as usize) % 128;
                    for _ in 0..extra {
                        mutant.payload.push((rng() >> 24) as u8);
                    }
                }
                // Record-structure bit flips (pages/streams records).
                4 => {
                    if mutant.pages.is_empty() {
                        continue;
                    }
                    let arena: &mut [u8] = unsafe {
                        core::slice::from_raw_parts_mut(
                            mutant.pages.as_mut_ptr() as *mut u8,
                            core::mem::size_of_val(&mut mutant.pages[..] as &mut [FlatPage]),
                        )
                    };
                    for _ in 0..8 {
                        let at = (rng() as usize) % arena.len();
                        arena[at] ^= 1 << ((rng() >> 40) % 8);
                    }
                }
                // Record-structure bit flips over streams.
                5 => {
                    if mutant.streams.is_empty() {
                        continue;
                    }
                    let arena: &mut [u8] = unsafe {
                        core::slice::from_raw_parts_mut(
                            mutant.streams.as_mut_ptr() as *mut u8,
                            core::mem::size_of_val(&mut mutant.streams[..] as &mut [FlatStream]),
                        )
                    };
                    for _ in 0..8 {
                        let at = (rng() as usize) % arena.len();
                        arena[at] ^= 1 << ((rng() >> 40) % 8);
                    }
                }
                // Hypothesis-cycle mutations (residual periodic pages).
                _ => {
                    if !mutant.cycle.is_empty() {
                        let at = (rng() as usize) % mutant.cycle.len();
                        mutant.cycle[at] = ((rng() as i64) % (1 << 40)) as i32;
                    }
                }
            }
            // Cap declared geometry so oversized claims stay bounded (the
            // decoder also enforces its own ceilings; this keeps the fuzz in
            // the interesting region without trivially failing the first
            // check).
            for p in &mut mutant.pages {
                p.frames %= 4096;
                p.channels %= 8;
            }
            let mut out = vec![0i32; mutant.arena_samples.max(1)];
            let mut scratch = vec![0u8; mutant.max_page_scratch.max(64)];
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                mutant.decode_pages_host(&mut out, &mut scratch, 0..mutant.pages.len())
            }));
            assert!(
                result.is_ok(),
                "decoder panicked on mutation round {round} (op {op})"
            );
        }
    }

    /// Shared `decode_stream` adversarial checks (RANS.md decode step 3):
    /// the function the host flat path and the CUDA kernel both call must
    /// reject trailing bytes (extended payload_len), truncations, and a
    /// below-range initial state on real flat jobs — never silently accept
    /// them as canonical.
    #[test]
    fn shared_decode_stream_rejects_noncanonical_streams() {
        use crate::device::entropy_shared::{STREAM_RANS, decode_stream};
        let samples = tone(2048, 2);
        let descriptor =
            ObjectDescriptor::new(Representation::Literal, 2048, Layout::Stereo, None).unwrap();
        let rl = RepresentedLiteral::encode(
            descriptor,
            &samples,
            512,
            Symbolization::DeltaLane4,
            ModelMode::Inline,
            false,
        )
        .unwrap();
        let job = flatten_literal_range(&rl, 0, 2048).unwrap();
        // Locate a RANS stream inside a RANS page.
        let mut found = None;
        'pages: for p in &job.pages {
            let sbegin = p.stream_off as usize;
            for si in 0..p.stream_count as usize {
                let st = &job.streams[sbegin + si];
                if st.kind == STREAM_RANS && st.symbol_count > 0 {
                    found = Some((*p, si));
                    break 'pages;
                }
            }
        }
        let Some((page, si)) = found else {
            panic!("no RANS stream in the literal job");
        };
        let mut scratch = vec![0u8; (page.scratch_off as usize + 4096).max(64)];
        let mut st_ok =
            |job: &FlatEntropyJob, st: &crate::device::entropy_shared::FlatStream| -> bool {
                decode_stream(st, &job.payload, &job.model_set(), &mut scratch)
            };
        let st = job.streams[page.stream_off as usize + si];
        // Canonical stream decodes.
        assert!(st_ok(&job, &st), "canonical RANS stream must decode");
        let sidx = page.stream_off as usize + si;
        // Trailing bytes: extend payload_len so the stream bleeds into the
        // next stream's bytes / past the payload arena — rejected.
        for extra in 1..=4u32 {
            let mut j = job.clone();
            j.streams[sidx].payload_len = st.payload_len.saturating_add(extra);
            assert!(
                !st_ok(&j, &j.streams[sidx]),
                "trailing {extra} bytes must be rejected"
            );
        }
        // Truncation of the encoded payload — rejected.
        for cut in 1..=4u32 {
            if st.payload_len > cut {
                let mut j = job.clone();
                j.streams[sidx].payload_len = st.payload_len - cut;
                assert!(
                    !st_ok(&j, &j.streams[sidx]),
                    "truncation by {cut} must be rejected"
                );
            }
        }
        // Below-range initial state — rejected at dec_init.
        let mut j = job.clone();
        let low = (crate::entropy::rans::STATE_L - 1).to_le_bytes();
        j.payload[st.payload_off as usize..st.payload_off as usize + 4].copy_from_slice(&low);
        assert!(
            !st_ok(&j, &j.streams[sidx]),
            "below-range initial state must be rejected"
        );
    }
}
