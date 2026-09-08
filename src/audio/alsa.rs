//! ALSA `hw:` direct-mmap endpoint support (Phase H, D1 court) — Linux only.
//!
//! Scope discipline (paper §"ALSA — first physical endpoint"): only `hw:`
//! hardware PCM devices are accepted (a plug/convert PCM would answer a
//! different question); access is `SND_PCM_ACCESS_MMAP_INTERLEAVED`; the
//! endpoint format for the first D1 court is `S32_LE` (the canonical i32
//! sample-code domain maps byte-for-byte; no quantizing pack step). The
//! compatibility copy calls (`snd_pcm_mmap_writei`/`writen`) are never used —
//! the actual mmap areas are written.
//!
//! The module is split:
//! * `alsa_ffi` — audited dlopen FFI;
//! * this file — discovery, configuration, mmap-region discipline, evidence
//!   snapshots, failure classification. All `unsafe` sits behind small
//!   audited wrappers; semantic code never touches raw pointers.

#![cfg(target_os = "linux")]

use crate::audio::alsa_ffi::{
    self, ChannelArea, SND_PCM_ACCESS_MMAP_INTERLEAVED, SND_PCM_FORMAT_S32_LE, SND_PCM_NONBLOCK,
    SND_PCM_STATE_PREPARED, SND_PCM_STATE_RUNNING, SND_PCM_STATE_XRUN, SND_PCM_STREAM_PLAYBACK,
    SndPcm, Uframes,
};
use crate::error::{Error, Kind, Result};
use crate::status::Verdict;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Bytes per sample of the frozen D1 endpoint format (S32_LE = 32 bits).
pub const S32_BYTES_PER_SAMPLE: u64 = 4;
/// Bit width of S32_LE.
pub const S32_BITS_PER_SAMPLE: u64 = 32;

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// One playback-capable ALSA PCM discovered on this host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EndpointInfo {
    /// `hw:<card>,<device>` PCM name.
    pub pcm_name: String,
    pub card: u32,
    pub device: u32,
    /// Card name token from /proc/asound/cards, e.g. "O12".
    pub card_name: Option<String>,
    /// Card description from /proc/asound/cards, e.g. "USB-Audio - Onyx Artist 1-2".
    pub card_description: Option<String>,
    /// PCM label from /proc/asound/pcm, e.g. "USB Audio".
    pub pcm_label: Option<String>,
    /// Kernel driver of the card device (best effort via sysfs), e.g.
    /// "snd-usb-audio" / "snd_hda_intel".
    pub driver: Option<String>,
}

/// Parse one `/proc/asound/pcm` entry line: "CC-DD: name : desc : playback 1
/// [: capture 1]". Returns (card, device, has_playback) when the shape
/// matches; never panics on hostile input.
pub fn parse_proc_pcm_line(line: &str) -> Option<(u32, u32, bool)> {
    let line = line.trim();
    let head = line.split(':').next()?;
    let (c, d) = head.split_once('-')?;
    let card: u32 = c.trim().parse().ok()?;
    let device: u32 = d.trim().parse().ok()?;
    let rest = &line[head.len()..];
    Some((card, device, rest.contains("playback")))
}

/// Parse one `/proc/asound/cards` line:
/// " 0 [O12            ]: USB-Audio - Onyx Artist 1-2" -> (0, "O12", desc).
pub fn parse_proc_cards_line(line: &str) -> Option<(u32, String, String)> {
    let line = line.trim();
    let (num, rest) = line.split_once('[')?;
    let card: u32 = num.trim().parse().ok()?;
    let (name, desc) = rest.split_once(']')?;
    let name = name.trim().to_string();
    let desc = desc.trim_start_matches(':').trim().to_string();
    Some((card, name, desc))
}

/// Best-effort kernel driver name of a sound card (sysfs).
pub fn card_driver(card: u32) -> Option<String> {
    let dev = PathBuf::from(format!("/sys/class/sound/card{card}/device"));
    let driver = dev.join("driver");
    std::fs::read_link(&driver)
        .ok()
        .and_then(|p| p.file_name().map(|f| f.to_string_lossy().into_owned()))
}

/// Enumerate playback-capable ALSA PCMs from /proc/asound.
pub fn list_playback_endpoints() -> Vec<EndpointInfo> {
    let mut cards: Vec<(u32, String, String)> = Vec::new();
    if let Ok(text) = std::fs::read_to_string("/proc/asound/cards") {
        for line in text.lines() {
            if let Some(t) = parse_proc_cards_line(line) {
                cards.push(t);
            }
        }
    }
    let card_meta = |c: u32| -> (Option<String>, Option<String>) {
        cards
            .iter()
            .find(|(n, _, _)| *n == c)
            .map(|(_, name, desc)| (Some(name.clone()), Some(desc.clone())))
            .unwrap_or((None, None))
    };
    let mut out = Vec::new();
    if let Ok(text) = std::fs::read_to_string("/proc/asound/pcm") {
        for line in text.lines() {
            let Some((card, device, playback)) = parse_proc_pcm_line(line) else {
                continue;
            };
            if !playback {
                continue;
            }
            let (card_name, card_description) = card_meta(card);
            let pcm_label = line.trim().split(':').nth(1).map(|s| s.trim().to_string());
            out.push(EndpointInfo {
                pcm_name: format!("hw:{card},{device}"),
                card,
                device,
                card_name,
                card_description,
                pcm_label,
                driver: card_driver(card),
            });
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Failure classification
// ---------------------------------------------------------------------------

/// A structured endpoint failure: stage + rc + message. Courts record it
/// verbatim and classify it with `classify_failure`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlsaFailure {
    /// Where the failure happened: "open", "hw_access", "hw_format",
    /// "hw_apply", "sw_params", "prepare", "run".
    pub stage: String,
    /// Exact libasound rc (negative errno) when available.
    pub rc: Option<i32>,
    pub message: String,
}

impl AlsaFailure {
    pub fn new(stage: impl Into<String>, rc: Option<i32>, message: impl Into<String>) -> Self {
        AlsaFailure {
            stage: stage.into(),
            rc,
            message: message.into(),
        }
    }

    /// True for xrun-class conditions: an explicit `-EPIPE` rc or a short
    /// `snd_pcm_mmap_commit` (a nonnegative transfer count that is less than
    /// the requested frames — ALSA's documented xrun-class event, produced by
    /// `mmap_commit` with `rc: None`). Sessions count these as discontinuities.
    pub fn is_xrun_class(&self) -> bool {
        self.rc == Some(-libc::EPIPE) || self.message.starts_with("short mmap_commit")
    }
}

impl std::fmt::Display for AlsaFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.rc {
            Some(rc) => write!(f, "{}: {} (rc {rc})", self.stage, self.message),
            None => write!(f, "{}: {}", self.stage, self.message),
        }
    }
}

/// Classify an endpoint failure into the verdict vocabulary. Pure function
/// (unit-tested): the stage tells us whether the endpoint or the API surface
/// refused, and the rc gives the reason.
pub fn classify_failure(f: &AlsaFailure) -> (Verdict, String) {
    let rc = f.rc;
    match (f.stage.as_str(), rc) {
        ("open", Some(rc)) if rc == -libc::EBUSY => (
            Verdict::Inconclusive,
            "device busy (in use by another application)".into(),
        ),
        ("open", Some(rc)) if rc == -libc::ENOENT => {
            (Verdict::NotApplicable, format!("no such PCM device ({f})"))
        }
        ("hw_access", _) => (
            Verdict::UnsupportedByHardware,
            "endpoint driver refuses SND_PCM_ACCESS_MMAP_INTERLEAVED (no hw mmap)".into(),
        ),
        ("hw_format", _) => (
            Verdict::UnsupportedByHardware,
            "endpoint does not offer S32_LE at the requested rate/channels (D1 first \
             court requires the exact i32 code domain; other formats are documented \
             future packing work)"
                .into(),
        ),
        ("hw_apply", Some(rc)) if rc == -libc::EBUSY => (
            Verdict::Inconclusive,
            "hw params busy (device in use)".into(),
        ),
        ("hw_apply", _) => (
            Verdict::UnsupportedByHardware,
            format!("hardware rejected the requested configuration ({f})"),
        ),
        ("rate", _) => (
            Verdict::UnsupportedByHardware,
            format!("endpoint would not grant the exact requested rate ({f})"),
        ),
        ("mmap_geometry", _) => (
            Verdict::UnsupportedByHardware,
            format!("unexpected interleaved channel-area geometry ({f})"),
        ),
        ("run", Some(rc)) if rc == -libc::EPIPE => (
            Verdict::FailedDeadline,
            "xrun (underrun): endpoint starved; explicit discontinuity recorded".into(),
        ),
        ("run", Some(rc)) if rc == -libc::ESTRPIPE => (
            Verdict::Inconclusive,
            "stream suspended (ESTRPIPE); resume policy recorded".into(),
        ),
        ("run", _) => (Verdict::FailedDeadline, format!("run-loop failure ({f})")),
        _ => (
            Verdict::Inconclusive,
            format!("unclassified endpoint failure ({f})"),
        ),
    }
}

// ---------------------------------------------------------------------------
// Endpoint configuration + handle
// ---------------------------------------------------------------------------

/// Frozen D1 endpoint configuration request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EndpointRequest {
    /// `hw:<card>,<device>` (anything else is refused — no plug conversion).
    pub pcm_name: String,
    /// Requested sample rate.
    pub rate_hz: u32,
    /// Requested channel count (the flat world's output channels must match).
    pub channels: u32,
    /// Requested period in frames.
    pub period_frames: u32,
    /// Requested buffer in frames.
    pub buffer_frames: u32,
}

impl EndpointRequest {
    pub fn hw_name_ok(&self) -> bool {
        self.pcm_name.starts_with("hw:")
    }
    /// Bytes per frame for S32_LE at `channels`.
    pub fn frame_bytes(&self) -> u64 {
        u64::from(self.channels) * S32_BYTES_PER_SAMPLE
    }
}

/// What the endpoint actually gave us (evidence snapshot).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EndpointSnapshot {
    pub pcm_name: String,
    pub driver: Option<String>,
    pub access: String,
    pub format: String,
    pub rate_hz: u32,
    pub channels: u32,
    pub period_frames: u64,
    pub buffer_frames: u64,
    /// buffer_frames * frame_bytes (the registered region size).
    pub buffer_bytes: u64,
    /// mmap area base (host address, hex).
    pub area_base: Option<String>,
    /// area.first / area.step for channel 0 (bits) — interleaved layout
    /// evidence.
    pub area_first_bits: Option<u32>,
    pub area_step_bits: Option<u32>,
    /// Per-channel (first, step) layout in bits, e.g. "ch0 (0,64); ch1 (32,64)".
    pub area_layout: Option<String>,
    /// True when the full per-channel interleaved geometry was validated at
    /// open (addr equal, first == ch*32, step == channels*32).
    pub area_layout_validated: bool,
    pub state: String,
}

/// Expected interleaved geometry of channel `c` of a `channels`-channel
/// S32_LE layout: `(first, step)` in bits — first = c*32, step = channels*32
/// (samples of one channel are `step` bits apart, all channels share one
/// contiguous `addr`). Pure function so the expectation is unit-tested.
pub fn expected_interleaved_first_step(c: u32, channels: u32) -> (u32, u32) {
    (
        c * S32_BITS_PER_SAMPLE as u32,
        channels * S32_BITS_PER_SAMPLE as u32,
    )
}

/// An open, configured, prepared ALSA `hw:` PCM (RAII: closes on drop).
/// Owns its libasound handle so the dlopen handle outlives the PCM.
pub struct AlsaPcm {
    _lib: alsa_ffi::AlsaLib,
    handle: SndPcm,
    pub request: EndpointRequest,
    /// Actual negotiated values (== request for the frozen court; the exact
    /// rate is enforced at open).
    pub rate_hz: u32,
    pub period_frames: u64,
    pub buffer_frames: u64,
    /// mmap area base for the whole interleaved buffer (host address).
    pub area_base: usize,
    pub area_first_bits: u32,
    pub area_step_bits: u32,
    /// Per-channel (first, step) in bits, validated at open.
    pub area_layout: Vec<(u32, u32)>,
}

impl AlsaPcm {
    /// Open + configure + prepare `request`. Refuses non-`hw:` names.
    pub fn open(request: &EndpointRequest) -> std::result::Result<AlsaPcm, AlsaFailure> {
        if !request.hw_name_ok() {
            return Err(AlsaFailure::new(
                "open",
                None,
                format!(
                    "refusing non-hw PCM name '{}' (plug/conversion would not be a direct \
                     hardware endpoint)",
                    request.pcm_name
                ),
            ));
        }
        let lib = alsa_ffi::AlsaLib::open()
            .map_err(|e| AlsaFailure::new("open", None, format!("libasound load: {e}")))?;
        let fns = lib.fns;
        let name = match std::ffi::CString::new(request.pcm_name.clone()) {
            Ok(n) => n,
            Err(_) => {
                return Err(AlsaFailure::new("open", None, "pcm name contains NUL"));
            }
        };
        // Non-blocking probe open; switched to blocking before the run loop.
        let mut handle: SndPcm = 0;
        // SAFETY: out-param; name lives for the call; nonblock mode.
        let rc = unsafe {
            (fns.snd_pcm_open.expect("bound"))(
                &mut handle,
                name.as_ptr(),
                SND_PCM_STREAM_PLAYBACK,
                SND_PCM_NONBLOCK,
            )
        };
        if rc != 0 {
            let f = AlsaFailure::new("open", Some(rc), alsa_ffi::snd_strerror_str(rc));
            return Err(f);
        }
        // Configure hardware parameters.
        let cfg = match configure_hw(&fns, handle, request) {
            Ok(c) => c,
            Err(f) => {
                // SAFETY: close the handle opened above.
                unsafe { (fns.snd_pcm_close.expect("bound"))(handle) };
                return Err(f);
            }
        };
        // mmap area base: prepare the stream first (mmap areas and the
        // avail state are only meaningful from PREPARED onward), then ask for
        // one frame without committing (no state change).
        // SAFETY: valid handle.
        let rc = unsafe { (fns.snd_pcm_prepare.expect("bound"))(handle) };
        if rc != 0 {
            let f = AlsaFailure::new("prepare", Some(rc), alsa_ffi::snd_strerror_str(rc));
            // SAFETY: close.
            unsafe { (fns.snd_pcm_close.expect("bound"))(handle) };
            return Err(f);
        }
        let area = match mmap_area_geometry(&fns, handle, request.channels) {
            Ok(a) => a,
            Err(f) => {
                // SAFETY: close.
                unsafe { (fns.snd_pcm_close.expect("bound"))(handle) };
                return Err(f);
            }
        };
        Ok(AlsaPcm {
            _lib: lib,
            handle,
            request: request.clone(),
            rate_hz: cfg.rate_hz,
            period_frames: cfg.period_frames,
            buffer_frames: cfg.buffer_frames,
            area_base: area.0,
            area_first_bits: area.1,
            area_step_bits: area.2,
            area_layout: area.3,
        })
    }

    /// Switch the handle to blocking mode (for the paced run loop).
    pub fn set_blocking(&self) -> Result<()> {
        // SAFETY: valid handle.
        let rc = unsafe { (self._lib.fns.snd_pcm_nonblock.expect("bound"))(self.handle, 0) };
        if rc != 0 {
            return Err(alsa_error("snd_pcm_nonblock(0)", rc));
        }
        Ok(())
    }

    pub fn name(&self) -> String {
        // SAFETY: driver-owned string; valid while the pcm is open.
        let p = unsafe { (self._lib.fns.snd_pcm_name.expect("bound"))(self.handle) };
        if p.is_null() {
            self.request.pcm_name.clone()
        } else {
            // SAFETY: NUL-terminated driver string.
            unsafe { std::ffi::CStr::from_ptr(p) }
                .to_string_lossy()
                .into_owned()
        }
    }

    pub fn prepare(&self) -> std::result::Result<(), AlsaFailure> {
        // SAFETY: valid handle.
        let rc = unsafe { (self._lib.fns.snd_pcm_prepare.expect("bound"))(self.handle) };
        if rc != 0 {
            return Err(AlsaFailure::new(
                "prepare",
                Some(rc),
                alsa_ffi::snd_strerror_str(rc),
            ));
        }
        Ok(())
    }

    /// Start the stream (explicit; the run loop calls this after the first
    /// committed chunk — the sw start-threshold does not reliably auto-start
    /// on every driver).
    pub fn start(&self) -> std::result::Result<(), AlsaFailure> {
        // SAFETY: valid handle.
        let rc = unsafe { (self._lib.fns.snd_pcm_start.expect("bound"))(self.handle) };
        if rc != 0 {
            return Err(AlsaFailure::new(
                "start",
                Some(rc),
                alsa_ffi::snd_strerror_str(rc),
            ));
        }
        Ok(())
    }

    pub fn state(&self) -> i32 {
        // SAFETY: valid handle.
        unsafe { (self._lib.fns.snd_pcm_state.expect("bound"))(self.handle) }
    }

    pub fn state_label(&self) -> &'static str {
        state_label(self.state())
    }

    /// Wait up to `timeout_ms` for the stream to become writable (avail >=
    /// avail_min). Returns Ok(true) ready, Ok(false) timeout.
    pub fn wait(&self, timeout_ms: i32) -> std::result::Result<bool, AlsaFailure> {
        // SAFETY: valid handle.
        let rc = unsafe { (self._lib.fns.snd_pcm_wait.expect("bound"))(self.handle, timeout_ms) };
        match rc {
            1 => Ok(true),
            0 => Ok(false),
            other => Err(AlsaFailure::new(
                "run",
                Some(other),
                alsa_ffi::snd_strerror_str(other),
            )),
        }
    }

    /// Current avail in frames; negative errno on error.
    pub fn avail(&self) -> std::result::Result<u64, AlsaFailure> {
        // SAFETY: valid handle.
        let rc = unsafe { (self._lib.fns.snd_pcm_avail_update.expect("bound"))(self.handle) };
        if rc < 0 {
            return Err(AlsaFailure::new(
                "run",
                Some(rc as i32),
                alsa_ffi::snd_strerror_str(rc as i32),
            ));
        }
        Ok(rc as u64)
    }

    /// Begin an mmap write region of at most `want` frames. Returns
    /// (offset_frames, contiguous_frames). Does not commit.
    pub fn mmap_begin(&self, want: u64) -> std::result::Result<(u64, u64), AlsaFailure> {
        let mut area_ptr: *const ChannelArea = std::ptr::null();
        let mut offset: Uframes = 0;
        // mmap_begin reads *frames as the *requested* size and writes back
        // the contiguous frames granted (<= request).
        let mut frames: Uframes = want as Uframes;
        let state_before = self.state();
        // SAFETY: out-params; area_ptr points at driver-owned area array
        // (channel 0 is the interleaved base); no commit yet.
        let rc = unsafe {
            (self._lib.fns.snd_pcm_mmap_begin.expect("bound"))(
                self.handle,
                &mut area_ptr,
                &mut offset,
                &mut frames,
            )
        };
        if std::env::var("VOLE_ALSA_DEBUG")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            // SAFETY: valid handle; debug-only diagnostic read.
            let avail =
                unsafe { (self._lib.fns.snd_pcm_avail_update.expect("bound"))(self.handle) };
            eprintln!(
                "[alsa-debug] mmap_begin(want {want}): rc {rc} off {offset} frames {frames} \
                 state {state_before}->{} avail_after {avail}",
                self.state_label()
            );
        }
        if rc != 0 {
            return Err(AlsaFailure::new(
                "run",
                Some(rc),
                alsa_ffi::snd_strerror_str(rc),
            ));
        }
        if area_ptr.is_null() || frames == 0 || frames > want {
            return Err(AlsaFailure::new(
                "run",
                Some(rc),
                format!("mmap_begin returned invalid region (frames {frames}, want {want})"),
            ));
        }
        // Re-prove the channel-area geometry on EVERY begin: each chunk must
        // still refer to the SAME registered interleaved ring (the returned
        // base must equal the base registered at open, all channels share it,
        // first == ch*32, step == channels*32) rather than relying on driver
        // stability across the session. If ALSA ever returned a different
        // mapping, D1 must not continue writing through the old registration.
        // SAFETY: the area array has one entry per channel (interleaved).
        let areas = unsafe { std::slice::from_raw_parts(area_ptr, self.request.channels as usize) };
        let base = areas[0].addr as usize;
        if base != self.area_base {
            return Err(AlsaFailure::new(
                "mmap_geometry",
                None,
                format!(
                    "mmap_begin returned base 0x{base:x} != registered base 0x{:x}: the mapping \
                     changed; refusing to write through the stale registration",
                    self.area_base
                ),
            ));
        }
        if self.area_layout.len() != self.request.channels as usize
            || self.area_layout.len() != areas.len()
        {
            return Err(AlsaFailure::new(
                "mmap_geometry",
                None,
                "area array length changed since open".to_string(),
            ));
        }
        for (c, area) in areas.iter().enumerate() {
            let (expect_first, expect_step) =
                expected_interleaved_first_step(c as u32, self.request.channels);
            let (open_first, open_step) = self.area_layout[c];
            if area.addr as usize != base
                || area.first != expect_first
                || area.step != expect_step
                || area.first != open_first
                || area.step != open_step
            {
                return Err(AlsaFailure::new(
                    "mmap_geometry",
                    None,
                    format!(
                        "channel {c} geometry changed since open (addr 0x{:x} first {} step {}; \
                         expected first {expect_first} step {expect_step})",
                        area.addr as usize, area.first, area.step
                    ),
                ));
            }
        }
        Ok((offset as u64, frames as u64))
    }

    /// Commit `frames` written frames at `offset`. ALSA returns the exact
    /// number of frames actually transferred; a nonnegative result that is
    /// not `frames` is a short commit (an xrun-class event) and is refused
    /// here — the court's claim is "every chunk committed", so the exact
    /// transferred count is part of the evidence.
    pub fn mmap_commit(&self, offset: u64, frames: u64) -> std::result::Result<(), AlsaFailure> {
        // SAFETY: valid handle; frames must be exactly what was written.
        let rc = unsafe {
            (self._lib.fns.snd_pcm_mmap_commit.expect("bound"))(
                self.handle,
                offset as Uframes,
                frames as Uframes,
            )
        };
        if rc < 0 {
            return Err(AlsaFailure::new(
                "run",
                Some(rc as i32),
                alsa_ffi::snd_strerror_str(rc as i32),
            ));
        }
        let transferred = rc as u64;
        if transferred != frames {
            return Err(AlsaFailure::new(
                "run",
                None,
                format!(
                    "short mmap_commit: requested {frames} frames, driver transferred {transferred} \
                     (xrun-class event; explicit discontinuity recorded)"
                ),
            ));
        }
        Ok(())
    }

    /// Recover from an xrun/suspend error (`recover(rc)`); Ok(true) when the
    /// stream was recovered to PREPARED/RUNNING.
    pub fn recover(&self, rc: i32) -> std::result::Result<bool, AlsaFailure> {
        // SAFETY: valid handle; non-silent so we know what happened.
        let r = unsafe { (self._lib.fns.snd_pcm_recover.expect("bound"))(self.handle, rc, 0) };
        if r != 0 {
            return Err(AlsaFailure::new(
                "run",
                Some(r),
                alsa_ffi::snd_strerror_str(r),
            ));
        }
        Ok(matches!(
            self.state(),
            SND_PCM_STATE_PREPARED | SND_PCM_STATE_RUNNING
        ))
    }

    /// Drain (block until queued frames played). Returns the state after.
    pub fn drain(&self) -> std::result::Result<i32, AlsaFailure> {
        // SAFETY: valid handle.
        let rc = unsafe { (self._lib.fns.snd_pcm_drain.expect("bound"))(self.handle) };
        if rc != 0 {
            return Err(AlsaFailure::new(
                "run",
                Some(rc),
                alsa_ffi::snd_strerror_str(rc),
            ));
        }
        Ok(self.state())
    }

    pub fn snapshot(&self) -> EndpointSnapshot {
        let layout = self
            .area_layout
            .iter()
            .enumerate()
            .map(|(c, (first, step))| format!("ch{c} first={first} step={step}"))
            .collect::<Vec<_>>()
            .join("; ");
        EndpointSnapshot {
            pcm_name: self.name(),
            driver: None, // filled by the court from discovery
            access: "MMAP_INTERLEAVED".into(),
            format: "S32_LE".into(),
            rate_hz: self.rate_hz,
            channels: self.request.channels,
            period_frames: self.period_frames,
            buffer_frames: self.buffer_frames,
            buffer_bytes: self.buffer_frames * self.request.frame_bytes(),
            area_base: Some(format!("0x{:x}", self.area_base)),
            area_first_bits: Some(self.area_first_bits),
            area_step_bits: Some(self.area_step_bits),
            area_layout: Some(layout),
            area_layout_validated: true,
            state: self.state_label().into(),
        }
    }
}

impl Drop for AlsaPcm {
    fn drop(&mut self) {
        if self.handle != 0 {
            // SAFETY: close the pcm (best effort at teardown; drop order is
            // declaration order, so the lib handle outlives this).
            unsafe { (self._lib.fns.snd_pcm_close.expect("bound"))(self.handle) };
        }
    }
}

fn state_label(state: i32) -> &'static str {
    match state {
        alsa_ffi::SND_PCM_STATE_OPEN => "OPEN",
        alsa_ffi::SND_PCM_STATE_SETUP => "SETUP",
        SND_PCM_STATE_PREPARED => "PREPARED",
        SND_PCM_STATE_RUNNING => "RUNNING",
        SND_PCM_STATE_XRUN => "XRUN",
        alsa_ffi::SND_PCM_STATE_DRAINING => "DRAINING",
        alsa_ffi::SND_PCM_STATE_PAUSED => "PAUSED",
        alsa_ffi::SND_PCM_STATE_SUSPENDED => "SUSPENDED",
        alsa_ffi::SND_PCM_STATE_DISCONNECTED => "DISCONNECTED",
        _ => "UNKNOWN",
    }
}

fn alsa_error(what: &str, rc: i32) -> Error {
    Error::new(
        Kind::External,
        format!("alsa {what}: {} (rc {rc})", alsa_ffi::snd_strerror_str(rc)),
    )
}

/// Negotiated hardware configuration.
struct Negotiated {
    rate_hz: u32,
    period_frames: u64,
    buffer_frames: u64,
}

/// Configure the frozen D1 shape: MMAP_INTERLEAVED + S32_LE + channels +
/// rate near + period/buffer near.
fn configure_hw(
    fns: &alsa_ffi::AlsaFns,
    handle: SndPcm,
    req: &EndpointRequest,
) -> std::result::Result<Negotiated, AlsaFailure> {
    let mut params: alsa_ffi::HwParams = 0;
    // SAFETY: out-param.
    let rc = unsafe { (fns.snd_pcm_hw_params_malloc.expect("bound"))(&mut params) };
    if rc != 0 {
        return Err(AlsaFailure::new(
            "hw_apply",
            Some(rc),
            alsa_ffi::snd_strerror_str(rc),
        ));
    }
    let finish = |params: alsa_ffi::HwParams| {
        // SAFETY: free the params object.
        unsafe { (fns.snd_pcm_hw_params_free.expect("bound"))(params) };
    };
    // SAFETY: init params to full range.
    let rc = unsafe { (fns.snd_pcm_hw_params_any.expect("bound"))(handle, params) };
    if rc != 0 {
        let f = AlsaFailure::new("hw_apply", Some(rc), alsa_ffi::snd_strerror_str(rc));
        finish(params);
        return Err(f);
    }
    // SAFETY: set access (exact; this is where non-mmap drivers refuse).
    let rc = unsafe {
        (fns.snd_pcm_hw_params_set_access.expect("bound"))(
            handle,
            params,
            SND_PCM_ACCESS_MMAP_INTERLEAVED,
        )
    };
    if rc != 0 {
        let f = AlsaFailure::new("hw_access", Some(rc), alsa_ffi::snd_strerror_str(rc));
        finish(params);
        return Err(f);
    }
    // SAFETY: set format (exact S32_LE).
    let rc = unsafe {
        (fns.snd_pcm_hw_params_set_format.expect("bound"))(handle, params, SND_PCM_FORMAT_S32_LE)
    };
    if rc != 0 {
        let f = AlsaFailure::new("hw_format", Some(rc), alsa_ffi::snd_strerror_str(rc));
        finish(params);
        return Err(f);
    }
    // SAFETY: set channels.
    let rc = unsafe {
        (fns.snd_pcm_hw_params_set_channels.expect("bound"))(handle, params, req.channels)
    };
    if rc != 0 {
        let f = AlsaFailure::new("hw_format", Some(rc), alsa_ffi::snd_strerror_str(rc));
        finish(params);
        return Err(f);
    }
    // SAFETY: rate near (out param updated to the nearest supported rate).
    // The *exact* requested rate is mandatory: playing the frozen media
    // timeline at a nearby rate would be a silent resample, so a driver that
    // cannot grant it exactly is refused here (never silently accepted).
    let mut rate = req.rate_hz;
    let mut dir = 0;
    let rc = unsafe {
        (fns.snd_pcm_hw_params_set_rate_near.expect("bound"))(handle, params, &mut rate, &mut dir)
    };
    if rc != 0 {
        let f = AlsaFailure::new("hw_apply", Some(rc), alsa_ffi::snd_strerror_str(rc));
        finish(params);
        return Err(f);
    }
    // SAFETY: period size near.
    let mut period = req.period_frames as Uframes;
    let rc = unsafe {
        (fns.snd_pcm_hw_params_set_period_size_near.expect("bound"))(
            handle,
            params,
            &mut period,
            &mut dir,
        )
    };
    if rc != 0 {
        let f = AlsaFailure::new("hw_apply", Some(rc), alsa_ffi::snd_strerror_str(rc));
        finish(params);
        return Err(f);
    }
    // SAFETY: buffer size near.
    let mut buffer = req.buffer_frames as Uframes;
    let rc = unsafe {
        (fns.snd_pcm_hw_params_set_buffer_size_near.expect("bound"))(handle, params, &mut buffer)
    };
    if rc != 0 {
        let f = AlsaFailure::new("hw_apply", Some(rc), alsa_ffi::snd_strerror_str(rc));
        finish(params);
        return Err(f);
    }
    // SAFETY: commit params.
    let rc = unsafe { (fns.snd_pcm_hw_params.expect("bound"))(handle, params) };
    if rc != 0 {
        let f = AlsaFailure::new("hw_apply", Some(rc), alsa_ffi::snd_strerror_str(rc));
        finish(params);
        return Err(f);
    }
    // Read back the negotiated values (never assume the request was granted
    // verbatim).
    let mut got_rate: u32 = 0;
    // SAFETY: out-param.
    let _ = unsafe {
        (fns.snd_pcm_hw_params_get_rate.expect("bound"))(
            params,
            &mut got_rate,
            std::ptr::null_mut(),
        )
    };
    let mut got_period: alsa_ffi::Uframes = 0;
    // SAFETY: out-param.
    let _ = unsafe {
        (fns.snd_pcm_hw_params_get_period_size.expect("bound"))(
            params,
            &mut got_period,
            std::ptr::null_mut(),
        )
    };
    let mut got_buffer: alsa_ffi::Uframes = 0;
    // SAFETY: out-param.
    let _ =
        unsafe { (fns.snd_pcm_hw_params_get_buffer_size.expect("bound"))(params, &mut got_buffer) };
    if got_rate != req.rate_hz {
        let f = AlsaFailure::new(
            "rate",
            None,
            format!(
                "negotiated rate {got_rate} != requested {}; refusing a silent resample",
                req.rate_hz
            ),
        );
        finish(params);
        return Err(f);
    }
    finish(params);
    // Switch params to sw params: avail_min = period, start_threshold =
    // period (the stream auto-starts when one full period is committed, so
    // the first commit triggers the DMA — no silent pre-roll).
    let mut sw: alsa_ffi::SwParams = 0;
    // SAFETY: out-param.
    let rc = unsafe { (fns.snd_pcm_sw_params_malloc.expect("bound"))(&mut sw) };
    if rc != 0 {
        return Err(AlsaFailure::new(
            "sw_params",
            Some(rc),
            alsa_ffi::snd_strerror_str(rc),
        ));
    }
    // SAFETY: current sw params as the base.
    let rc = unsafe { (fns.snd_pcm_sw_params_current.expect("bound"))(handle, sw) };
    if rc != 0 {
        // SAFETY: free sw.
        unsafe { (fns.snd_pcm_sw_params_free.expect("bound"))(sw) };
        return Err(AlsaFailure::new(
            "sw_params",
            Some(rc),
            alsa_ffi::snd_strerror_str(rc),
        ));
    }
    // SAFETY: avail_min = period.
    let rc = unsafe { (fns.snd_pcm_sw_params_set_avail_min.expect("bound"))(handle, sw, period) };
    if rc == 0 {
        // SAFETY: start_threshold = period.
        let rc2 = unsafe {
            (fns.snd_pcm_sw_params_set_start_threshold.expect("bound"))(handle, sw, period)
        };
        let _ = rc2;
    }
    // SAFETY: commit sw params.
    let rc = unsafe { (fns.snd_pcm_sw_params.expect("bound"))(handle, sw) };
    // SAFETY: free sw.
    unsafe { (fns.snd_pcm_sw_params_free.expect("bound"))(sw) };
    if rc != 0 {
        return Err(AlsaFailure::new(
            "sw_params",
            Some(rc),
            alsa_ffi::snd_strerror_str(rc),
        ));
    }
    Ok(Negotiated {
        rate_hz: got_rate,
        period_frames: got_period as u64,
        buffer_frames: got_buffer as u64,
    })
}

/// (base, ch0 first bits, ch0 step bits, per-channel (first, step) layout).
type AreaGeometry = (usize, u32, u32, Vec<(u32, u32)>);

/// Ask mmap_begin for one frame (without committing) to learn the interleaved
/// area base and validate the **entire** per-channel geometry: every channel
/// shares one contiguous `addr`, `first == ch * 32` bits and
/// `step == channels * 32` bits (two packed S32_LE samples are 64 bits
/// apart). The simplified pointer arithmetic used later
/// (`base + offset * frame_bytes`) is only sound once this is proved.
fn mmap_area_geometry(
    fns: &alsa_ffi::AlsaFns,
    handle: SndPcm,
    channels: u32,
) -> std::result::Result<AreaGeometry, AlsaFailure> {
    let mut area_ptr: *const ChannelArea = std::ptr::null();
    let mut offset: Uframes = 0;
    let mut frames: Uframes = 1; // request one frame (input value; see mmap_begin)
    // SAFETY: out-params; no commit (state unchanged).
    let rc = unsafe {
        (fns.snd_pcm_mmap_begin.expect("bound"))(handle, &mut area_ptr, &mut offset, &mut frames)
    };
    if rc != 0 {
        return Err(AlsaFailure::new(
            "mmap_begin",
            Some(rc),
            alsa_ffi::snd_strerror_str(rc),
        ));
    }
    if area_ptr.is_null() || frames == 0 {
        return Err(AlsaFailure::new("mmap_begin", None, "null/empty area"));
    }
    // SAFETY: the area array has one entry per channel (interleaved).
    let areas = unsafe { std::slice::from_raw_parts(area_ptr, channels as usize) };
    let base = areas[0].addr as usize;
    let mut layout = Vec::with_capacity(channels as usize);
    for (c, area) in areas.iter().enumerate() {
        let (expect_first, expect_step) = expected_interleaved_first_step(c as u32, channels);
        if area.addr as usize != base || area.first != expect_first || area.step != expect_step {
            return Err(AlsaFailure::new(
                "mmap_geometry",
                None,
                format!(
                    "channel {c}: got addr 0x{:x} first {} step {}; expected addr 0x{base:x} \
                     first {expect_first} step {expect_step}",
                    area.addr as usize, area.first, area.step
                ),
            ));
        }
        layout.push((area.first, area.step));
    }
    Ok((base, layout[0].0, layout[0].1, layout))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proc_pcm_line_parsing() {
        assert_eq!(
            parse_proc_pcm_line(" 0-00: USB Audio : USB Audio : playback 1 : capture 1"),
            Some((0, 0, true))
        );
        assert_eq!(
            parse_proc_pcm_line("01-03: HDMI 0 : HDMI 0 : playback 1"),
            Some((1, 3, true))
        );
        assert_eq!(
            parse_proc_pcm_line("03-00: USB Audio : USB Audio : capture 1"),
            Some((3, 0, false))
        );
        assert_eq!(parse_proc_pcm_line("garbage"), None);
        assert_eq!(parse_proc_pcm_line(""), None);
    }

    #[test]
    fn proc_cards_line_parsing() {
        assert_eq!(
            parse_proc_cards_line(" 0 [O12            ]: USB-Audio - Onyx Artist 1-2"),
            Some((0, "O12".into(), "USB-Audio - Onyx Artist 1-2".into()))
        );
        assert_eq!(parse_proc_cards_line("junk"), None);
    }

    #[test]
    fn hw_name_refusal_and_frame_bytes() {
        let req = EndpointRequest {
            pcm_name: "default".into(),
            rate_hz: 48_000,
            channels: 2,
            period_frames: 512,
            buffer_frames: 1024,
        };
        assert!(!req.hw_name_ok());
        assert_eq!(req.frame_bytes(), 8);
        let req2 = EndpointRequest {
            pcm_name: "hw:0,0".into(),
            ..req
        };
        assert!(req2.hw_name_ok());
    }

    #[test]
    fn failure_classification_is_honest() {
        // Busy device is never a fabricated hardware verdict.
        assert_eq!(
            classify_failure(&AlsaFailure::new("open", Some(-libc::EBUSY), "busy")).0,
            Verdict::Inconclusive
        );
        // mmap refusal is a hardware-class negative.
        assert_eq!(
            classify_failure(&AlsaFailure::new("hw_access", Some(-22), "no mmap")).0,
            Verdict::UnsupportedByHardware
        );
        assert_eq!(
            classify_failure(&AlsaFailure::new("hw_format", Some(-22), "no S32")).0,
            Verdict::UnsupportedByHardware
        );
        // xrun maps to a deadline/negative class with explicit discontinuity.
        assert_eq!(
            classify_failure(&AlsaFailure::new("run", Some(-libc::EPIPE), "xrun")).0,
            Verdict::FailedDeadline
        );
        // rate refusal / unexpected geometry are hardware-class negatives.
        assert_eq!(
            classify_failure(&AlsaFailure::new("rate", None, "negotiated 44100")).0,
            Verdict::UnsupportedByHardware
        );
        assert_eq!(
            classify_failure(&AlsaFailure::new("mmap_geometry", None, "ch1 step 32")).0,
            Verdict::UnsupportedByHardware
        );
        // Unknown stages stay inconclusive.
        assert_eq!(
            classify_failure(&AlsaFailure::new("mystery", None, "?")).0,
            Verdict::Inconclusive
        );
    }

    #[test]
    fn expected_interleaved_geometry_is_packed_s32() {
        // Two channels of S32_LE: ch0 first=0 step=64, ch1 first=32 step=64.
        assert_eq!(expected_interleaved_first_step(0, 2), (0, 64));
        assert_eq!(expected_interleaved_first_step(1, 2), (32, 64));
        assert_eq!(expected_interleaved_first_step(0, 1), (0, 32));
        // Stereo frame = 64 bits; step scales with channel count.
        assert_eq!(expected_interleaved_first_step(3, 8), (96, 256));
    }

    #[test]
    fn xrun_class_detection_covers_short_commits() {
        // -EPIPE underrun.
        assert!(AlsaFailure::new("run", Some(-libc::EPIPE), "xrun").is_xrun_class());
        // Short (nonnegative-but-less) mmap_commit: rc None, marked message.
        assert!(AlsaFailure::new(
            "run",
            None,
            "short mmap_commit: requested 512 frames, driver transferred 128 frames (xrun-class \
             event; explicit discontinuity recorded)"
        )
        .is_xrun_class());
        // Ordinary failures are not xrun-class.
        assert!(!AlsaFailure::new("open", Some(-libc::EBUSY), "busy").is_xrun_class());
        assert!(!AlsaFailure::new("run", None, "mmap_begin failed").is_xrun_class());
    }
}
