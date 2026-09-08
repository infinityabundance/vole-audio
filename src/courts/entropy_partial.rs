//! `court entropy-partial` — partial entropy materialization (H.2.13).
//!
//! Over the frozen corpus: for fixed and randomized window patterns
//! (including reverse-order and cross-page windows), the partial
//! materialization of a represented literal must equal the same slice of the
//! full scalar observation. The court also reports the decode halo (pages
//! touched per window), symbols decoded, and transient working memory (the
//! bounded per-page scratch of the largest page touched).

use crate::entropy::corpus;
use crate::entropy::represent::ModelMode;
use crate::entropy::represent::{PageKind, RepresentedLiteral};
use crate::entropy::symbol::Symbolization;
use crate::status::Verdict;
use std::path::Path;

const PAGE_FRAMES: u32 = 512;
const WINDOW_FRAMES: u32 = 512;

/// Run the court; writes an immutable receipt under `receipts/entropy-partial/`.
pub fn run(receipts_root: &Path) -> crate::error::Result<Verdict> {
    let fail = |why: &str| -> crate::error::Result<Verdict> {
        let mut b = crate::evidence::receipt::ReceiptBuilder::new("entropy-partial");
        b.result(Verdict::FailedCorrectness)
            .result_detail(format!("entropy-partial failed: {why}"));
        let (_, path) = b.finish_write(receipts_root)?;
        eprintln!("court entropy-partial: FAILED_CORRECTNESS ({why})");
        eprintln!("  receipt: {}", path.display());
        Ok(Verdict::FailedCorrectness)
    };

    let mut seed = 0x_70_61_72_74u64; // "part"
    let mut rng = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };

    let fixtures = corpus::all();
    let mut cells: Vec<serde_json::Value> = Vec::new();
    let mut windows_checked = 0usize;

    for fx in &fixtures {
        let channels = usize::from(fx.channels);
        let layout = if fx.channels == 1 {
            crate::universe::layout::Layout::Mono
        } else {
            crate::universe::layout::Layout::Stereo
        };
        let descriptor = crate::object::descriptor::ObjectDescriptor::new(
            crate::object::descriptor::Representation::Literal,
            fx.frames() as u64,
            layout,
            None,
        )
        .unwrap();
        for &sym in &[
            Symbolization::Lane4Plain,
            Symbolization::Lane4ZigZag,
            Symbolization::DeltaLane4,
        ] {
            let rl = match RepresentedLiteral::encode(
                descriptor.clone(),
                &fx.samples,
                PAGE_FRAMES,
                sym,
                ModelMode::Inline,
                false,
            ) {
                Ok(r) => r,
                Err(e) => return fail(&format!("{} encode: {e}", fx.name)),
            };
            let full = match rl.materialize_full() {
                Ok(f) => f,
                Err(e) => return fail(&format!("{} full decode: {e}", fx.name)),
            };
            if full != fx.samples {
                return fail(&format!("{} full decode mismatch", fx.name));
            }
            // Window battery: aligned, misaligned, single-frame, reverse
            // order (descending starts), cross-page.
            let mut windows: Vec<(u64, u32)> = vec![
                (0, WINDOW_FRAMES),
                (PAGE_FRAMES as u64 / 2, WINDOW_FRAMES / 2),
                (PAGE_FRAMES as u64 - 1, 2),
                (fx.frames() as u64 - 1, 1),
            ];
            for _ in 0..24 {
                let start = rng() % fx.frames() as u64;
                let len = 1 + (rng() % 800) as u32;
                if start + u64::from(len) <= fx.frames() as u64 {
                    windows.push((start, len));
                }
            }
            // Sort by descending start (reverse traversal pattern).
            windows.sort_by_key(|w| core::cmp::Reverse(w.0));

            let mut coded_symbols = 0u64;
            let mut max_page_scratch = 0u64;
            let mut total_pages_touched = 0usize;
            for &(start, len) in &windows {
                if start + u64::from(len) > fx.frames() as u64 {
                    continue;
                }
                let part = match rl.materialize(start, len) {
                    Ok(p) => p,
                    Err(e) => return fail(&format!("{} partial: {e}", fx.name)),
                };
                let lo = start as usize * channels;
                let hi = (start as usize + len as usize) * channels;
                if part != full[lo..hi] {
                    return fail(&format!(
                        "{} {sym:?}: partial != full slice at {start}+{len}",
                        fx.name
                    ));
                }
                windows_checked += 1;
                let touching = rl.pages_touching(start, len);
                total_pages_touched += touching.len();
                for page in &rl.pages {
                    let ps = page.start_frame;
                    let pe = ps + u64::from(page.frames);
                    if ps >= start + u64::from(len) || pe <= start {
                        continue;
                    }
                    // Entropy symbols actually decoded: coded symbol bytes of
                    // RANS streams (RAW pages are byte copies, not coded).
                    if page.kind == PageKind::Rans {
                        coded_symbols += page
                            .blocks
                            .iter()
                            .map(|b| b.payload.symbol_count())
                            .sum::<u64>();
                    }
                    // Transient working memory: bounded per-page scratch of
                    // the largest touched page (sample bytes).
                    let scratch = page.frames as u64 * fx.channels as u64 * 4;
                    max_page_scratch = max_page_scratch.max(scratch);
                }
            }
            cells.push(serde_json::json!({
                "fixture": fx.name,
                "kind": fx.kind,
                "channels": fx.channels,
                "frames": fx.frames(),
                "symbolization": sym.name(),
                "page_frames": PAGE_FRAMES,
                "windows_checked": windows.len(),
                "pages_touched_total": total_pages_touched,
                "mean_pages_touched": total_pages_touched as f64 / windows.len() as f64,
                "entropy_symbols_decoded_total": coded_symbols,
                "max_page_scratch_bytes": max_page_scratch,
                "partial_equals_full": true,
            }));
        }
    }
    if windows_checked == 0 {
        return fail("no windows checked");
    }

    let mut builder = crate::evidence::receipt::ReceiptBuilder::new("entropy-partial");
    builder
        .result(Verdict::Supported)
        .result_detail(format!(
            "partial == full slice over {windows_checked} windows across all fixtures (incl. reverse order)"
        ))
        .params(crate::evidence::receipt::CourtParams {
            universe: Some("vole.audio.u1".into()),
            profile: Some("u1/v1 + vole.entropy.p1/p1/v1".into()),
            backend: Some("scalar".into()),
            content_kind: Some("entropy-corpus-v1 partial".into()),
            ..Default::default()
        })
        .extra("cells", serde_json::Value::Array(cells));
    let (_, path) = builder.finish_write(receipts_root)?;
    println!("court entropy-partial: SUPPORTED");
    println!("  windows checked: {windows_checked}");
    println!("  receipt: {}", path.display());
    Ok(Verdict::Supported)
}
