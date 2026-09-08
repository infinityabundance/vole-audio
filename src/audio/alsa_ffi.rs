//! Minimal audited ALSA (libasound) FFI for the D1 endpoint court — Linux
//! only, dynamically loaded (`libasound.so.2`) so a machine without the
//! library (or non-Linux hosts) never breaks the build or the other courts.
//!
//! Only the exact surface Phase H uses is bound. Every numeric constant is
//! frozen from the ALSA headers on this system (`/usr/include/alsa/pcm.h`);
//! the values are part of the stable libasound ABI and are pinned by the
//! `frozen_constants` test. Function signatures are transcribed one-to-one
//! from `pcm.h` (audited-FFI discipline — see the comment on each `type`).
//!
//! The dlopen handle outlives every function pointer (mirrors
//! `backend::cuda::ffi`).

#![cfg(target_os = "linux")]

use crate::error::{Error, Kind};
use libc::{c_char, c_int, c_long, c_uint, c_ulong, c_void};
use std::ffi::CString;

// ---------------------------------------------------------------------------
// Types (sizes from the libasound ABI; x86-64 shown)
// ---------------------------------------------------------------------------

/// `snd_pcm_t*` — opaque (kept as a pointer-sized handle).
pub type SndPcm = usize;
/// `snd_pcm_uframes_t` = unsigned long (8 bytes on x86-64).
pub type Uframes = c_ulong;
/// `snd_pcm_sframes_t` = long.
pub type Sframes = c_long;
/// Opaque `snd_pcm_hw_params_t*` / `snd_pcm_sw_params_t*`.
pub type HwParams = usize;
pub type SwParams = usize;

/// `snd_pcm_channel_area_t` (pcm.h): base address plus `first` (offset to
/// first sample in bits) and `step` (sample distance in bits).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ChannelArea {
    pub addr: *mut c_void,
    pub first: c_uint,
    pub step: c_uint,
}

// ---------------------------------------------------------------------------
// Frozen constants — transcribed from /usr/include/alsa/pcm.h and pinned by
// the `frozen_constants` test (values are stable libasound ABI).
// ---------------------------------------------------------------------------

/// snd_pcm_stream_t: playback = 0 (pcm.h line ~118).
pub const SND_PCM_STREAM_PLAYBACK: c_int = 0;

/// snd_pcm_access_t (pcm.h lines 115-130).
pub const SND_PCM_ACCESS_MMAP_INTERLEAVED: c_int = 0;
pub const SND_PCM_ACCESS_MMAP_NONINTERLEAVED: c_int = 1;
pub const SND_PCM_ACCESS_RW_INTERLEAVED: c_int = 3;

/// snd_pcm_format_t (pcm.h enum, S8=0 …).
pub const SND_PCM_FORMAT_S16_LE: c_int = 2;
pub const SND_PCM_FORMAT_S32_LE: c_int = 10;

/// snd_pcm_state_t (pcm.h lines 305-322).
pub const SND_PCM_STATE_OPEN: c_int = 0;
pub const SND_PCM_STATE_SETUP: c_int = 1;
pub const SND_PCM_STATE_PREPARED: c_int = 2;
pub const SND_PCM_STATE_RUNNING: c_int = 3;
pub const SND_PCM_STATE_XRUN: c_int = 4;
pub const SND_PCM_STATE_DRAINING: c_int = 5;
pub const SND_PCM_STATE_PAUSED: c_int = 6;
pub const SND_PCM_STATE_SUSPENDED: c_int = 7;
pub const SND_PCM_STATE_DISCONNECTED: c_int = 8;

/// snd_pcm_open mode flags (pcm.h line 405).
pub const SND_PCM_NONBLOCK: c_int = 0x0000_0001;

// Linux errno values returned as negative `snd_pcm_*` error codes.
/// EPIPE — xrun.
pub const EPIPE: c_int = 32;
/// ESTRPIPE — stream suspended (e.g. resume).
pub const ESTRPIPE: c_int = 86;
/// EBADFD — wrong file descriptor/state.
pub const EBADFD: c_int = 77;

macro_rules! fns {
    ($( $name:ident : $fty:ty ),* $(,)?) => {
        #[allow(non_snake_case)]
        #[derive(Clone, Copy)]
        pub struct AlsaFns { $( pub $name: Option<$fty> ),* }

        impl AlsaFns {
            /// # SAFETY
            /// `lib` must stay loaded (valid dlopen handle) for as long as the
            /// returned function pointers are used.
            pub unsafe fn load(lib: *mut c_void) -> Self {
                unsafe {
                    AlsaFns {
                        $( $name: resolve(lib, stringify!($name)), )*
                    }
                }
            }
        }
    };
}

/// dlsym one symbol.
///
/// # SAFETY
/// The returned pointer is only valid while `lib` stays loaded.
unsafe fn resolve<T>(lib: *mut c_void, name: &str) -> Option<T> {
    let cname = CString::new(name).expect("symbol name has no NUL");
    unsafe {
        let p = libc::dlsym(lib, cname.as_ptr());
        if p.is_null() {
            None
        } else {
            Some(std::mem::transmute_copy(&p))
        }
    }
}

// Signatures transcribed from /usr/include/alsa/pcm.h (see line numbers).
type FnPcmOpen = unsafe extern "C" fn(*mut SndPcm, *const c_char, c_int, c_int) -> c_int; // 528
type FnPcmClose = unsafe extern "C" fn(SndPcm) -> c_int; // 537
type FnPcmName = unsafe extern "C" fn(SndPcm) -> *const c_char; // 538
type FnPcmNonblock = unsafe extern "C" fn(SndPcm, c_int) -> c_int; // 544
type FnHwParamsMalloc = unsafe extern "C" fn(*mut HwParams) -> c_int; // 791
type FnHwParamsFree = unsafe extern "C" fn(HwParams); // 792
type FnHwParamsAny = unsafe extern "C" fn(SndPcm, HwParams) -> c_int; // 735
type FnHwParamsSetAccess = unsafe extern "C" fn(SndPcm, HwParams, c_int) -> c_int; // 799
type FnHwParamsGetFormat = unsafe extern "C" fn(HwParams, *mut c_int) -> c_int; // 805
type FnHwParamsSetFormat = unsafe extern "C" fn(SndPcm, HwParams, c_int) -> c_int; // 807
type FnHwParamsGetChannels = unsafe extern "C" fn(HwParams, *mut c_uint) -> c_int; // 821
type FnHwParamsSetChannels = unsafe extern "C" fn(SndPcm, HwParams, c_uint) -> c_int; // 825
type FnHwParamsGetRate = unsafe extern "C" fn(HwParams, *mut c_uint, *mut c_int) -> c_int; // 833
type FnHwParamsSetRateNear =
    unsafe extern "C" fn(SndPcm, HwParams, *mut c_uint, *mut c_int) -> c_int; // 841
type FnHwParamsGetPeriodSize = unsafe extern "C" fn(HwParams, *mut Uframes, *mut c_int) -> c_int; // 865
type FnHwParamsSetPeriodSizeNear =
    unsafe extern "C" fn(SndPcm, HwParams, *mut Uframes, *mut c_int) -> c_int; // 873
type FnHwParamsGetBufferSize = unsafe extern "C" fn(HwParams, *mut Uframes) -> c_int; // 903
type FnHwParamsSetBufferSizeNear = unsafe extern "C" fn(SndPcm, HwParams, *mut Uframes) -> c_int; // 911
type FnHwParams = unsafe extern "C" fn(SndPcm, HwParams) -> c_int; // 551
type FnSwParamsMalloc = unsafe extern "C" fn(*mut SwParams) -> c_int; // 934
type FnSwParamsFree = unsafe extern "C" fn(SwParams); // 935
type FnSwParamsCurrent = unsafe extern "C" fn(SndPcm, SwParams) -> c_int; // 553
type FnSwParamsSetAvailMin = unsafe extern "C" fn(SndPcm, SwParams, Uframes) -> c_int; // 945
type FnSwParamsSetStartThreshold = unsafe extern "C" fn(SndPcm, SwParams, Uframes) -> c_int; // 949
type FnSwParams = unsafe extern "C" fn(SndPcm, SwParams) -> c_int; // 554
type FnPrepare = unsafe extern "C" fn(SndPcm) -> c_int; // 555
type FnStart = unsafe extern "C" fn(SndPcm) -> c_int; // 558
type FnDrop = unsafe extern "C" fn(SndPcm) -> c_int; // 559
type FnDrain = unsafe extern "C" fn(SndPcm) -> c_int; // 560
type FnState = unsafe extern "C" fn(SndPcm) -> c_int; // 562
type FnAvailUpdate = unsafe extern "C" fn(SndPcm) -> Sframes; // 568
type FnWait = unsafe extern "C" fn(SndPcm, c_int) -> c_int; // 578
type FnMmapBegin =
    unsafe extern "C" fn(SndPcm, *mut *const ChannelArea, *mut Uframes, *mut Uframes) -> c_int; // 1142
type FnMmapCommit = unsafe extern "C" fn(SndPcm, Uframes, Uframes) -> Sframes; // 1146
type FnRecover = unsafe extern "C" fn(SndPcm, c_int, c_int) -> c_int; // 680
type FnStrError = unsafe extern "C" fn(c_int) -> *const c_char; // error.h:50

fns! {
    snd_pcm_open: FnPcmOpen,
    snd_pcm_close: FnPcmClose,
    snd_pcm_name: FnPcmName,
    snd_pcm_nonblock: FnPcmNonblock,
    snd_pcm_hw_params_malloc: FnHwParamsMalloc,
    snd_pcm_hw_params_free: FnHwParamsFree,
    snd_pcm_hw_params_any: FnHwParamsAny,
    snd_pcm_hw_params_set_access: FnHwParamsSetAccess,
    snd_pcm_hw_params_get_format: FnHwParamsGetFormat,
    snd_pcm_hw_params_set_format: FnHwParamsSetFormat,
    snd_pcm_hw_params_get_channels: FnHwParamsGetChannels,
    snd_pcm_hw_params_set_channels: FnHwParamsSetChannels,
    snd_pcm_hw_params_get_rate: FnHwParamsGetRate,
    snd_pcm_hw_params_set_rate_near: FnHwParamsSetRateNear,
    snd_pcm_hw_params_get_period_size: FnHwParamsGetPeriodSize,
    snd_pcm_hw_params_set_period_size_near: FnHwParamsSetPeriodSizeNear,
    snd_pcm_hw_params_get_buffer_size: FnHwParamsGetBufferSize,
    snd_pcm_hw_params_set_buffer_size_near: FnHwParamsSetBufferSizeNear,
    snd_pcm_hw_params: FnHwParams,
    snd_pcm_sw_params_malloc: FnSwParamsMalloc,
    snd_pcm_sw_params_free: FnSwParamsFree,
    snd_pcm_sw_params_current: FnSwParamsCurrent,
    snd_pcm_sw_params_set_avail_min: FnSwParamsSetAvailMin,
    snd_pcm_sw_params_set_start_threshold: FnSwParamsSetStartThreshold,
    snd_pcm_sw_params: FnSwParams,
    snd_pcm_prepare: FnPrepare,
    snd_pcm_start: FnStart,
    snd_pcm_drop: FnDrop,
    snd_pcm_drain: FnDrain,
    snd_pcm_state: FnState,
    snd_pcm_avail_update: FnAvailUpdate,
    snd_pcm_wait: FnWait,
    snd_pcm_mmap_begin: FnMmapBegin,
    snd_pcm_mmap_commit: FnMmapCommit,
    snd_pcm_recover: FnRecover,
    snd_strerror: FnStrError,
}

/// Loaded libasound handle plus its resolved symbols.
pub struct AlsaLib {
    _lib: *mut c_void,
    pub fns: AlsaFns,
}

impl AlsaLib {
    /// Open `libasound.so.2` and resolve the bound API surface.
    pub fn open() -> crate::error::Result<AlsaLib> {
        // SAFETY: dlopen RTLD_NOW|RTLD_LOCAL; the handle is kept for the
        // lifetime of the AlsaLib (all fn pointers die with it).
        let name = CString::new("libasound.so.2").expect("no NUL");
        let lib = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if lib.is_null() {
            return Err(Error::new(
                Kind::Unavailable,
                "libasound.so.2 not loadable on this host",
            ));
        }
        let fns = unsafe { AlsaFns::load(lib) };
        // Every bound symbol must exist; a partial surface is an error, not a
        // silent degradation.
        let missing = fns.missing();
        if !missing.is_empty() {
            return Err(Error::new(
                Kind::Unsupported,
                format!("libasound missing required symbols: {}", missing.join(", ")),
            ));
        }
        install_strerror(fns.snd_strerror.expect("bound"));
        Ok(AlsaLib { _lib: lib, fns })
    }
}

impl AlsaFns {
    fn missing(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        macro_rules! probe {
            ($($name:ident),* $(,)?) => {
                $( if self.$name.is_none() { out.push(stringify!($name)); } )*
            };
        }
        probe! {
            snd_pcm_open, snd_pcm_close, snd_pcm_name, snd_pcm_nonblock,
            snd_pcm_hw_params_malloc, snd_pcm_hw_params_free, snd_pcm_hw_params_any,
            snd_pcm_hw_params_set_access, snd_pcm_hw_params_get_format,
            snd_pcm_hw_params_set_format, snd_pcm_hw_params_get_channels,
            snd_pcm_hw_params_set_channels, snd_pcm_hw_params_get_rate,
            snd_pcm_hw_params_set_rate_near, snd_pcm_hw_params_get_period_size,
            snd_pcm_hw_params_set_period_size_near, snd_pcm_hw_params_get_buffer_size,
            snd_pcm_hw_params_set_buffer_size_near, snd_pcm_hw_params,
            snd_pcm_sw_params_malloc, snd_pcm_sw_params_free, snd_pcm_sw_params_current,
            snd_pcm_sw_params_set_avail_min, snd_pcm_sw_params_set_start_threshold,
            snd_pcm_sw_params, snd_pcm_prepare, snd_pcm_start, snd_pcm_drop, snd_pcm_drain,
            snd_pcm_state, snd_pcm_avail_update, snd_pcm_wait, snd_pcm_mmap_begin,
            snd_pcm_mmap_commit, snd_pcm_recover, snd_strerror,
        }
        out
    }
}

impl std::fmt::Debug for AlsaLib {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AlsaLib(libasound.so.2)")
    }
}

impl Drop for AlsaLib {
    fn drop(&mut self) {
        if !self._lib.is_null() {
            // SAFETY: close after all uses (AlsaLib owns the only references).
            unsafe { libc::dlclose(self._lib) };
        }
    }
}

/// Format an ALSA error (`rc` may be negative errno) with the alsa message.
pub fn alsa_error(what: &str, rc: c_int) -> Error {
    let msg = snd_strerror_str(rc);
    Error::new(Kind::External, format!("alsa {what}: {msg} (rc {rc})"))
}

/// `snd_strerror` text for an errno (positive or negative); never panics.
pub fn snd_strerror_str(rc: c_int) -> String {
    let f = STRERROR.get().copied().unwrap_or(__strerror_unresolved);
    // SAFETY: f is the resolved driver symbol or the null-returning stub.
    let p = unsafe { f(rc) };
    if !p.is_null() {
        // SAFETY: NUL-terminated static string owned by libasound.
        let c = unsafe { std::ffi::CStr::from_ptr(p) };
        return c.to_string_lossy().into_owned();
    }
    format!("errno {rc}")
}

static STRERROR: std::sync::OnceLock<unsafe extern "C" fn(c_int) -> *const c_char> =
    std::sync::OnceLock::new();

unsafe extern "C" fn __strerror_unresolved(_rc: c_int) -> *const c_char {
    std::ptr::null()
}

/// Install the resolved `snd_strerror` for error formatting (called by
/// `AlsaLib::open`; idempotent).
pub fn install_strerror(f: unsafe extern "C" fn(c_int) -> *const c_char) {
    let _ = STRERROR.set(f);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_constants_match_libasound_abi() {
        // Values transcribed from /usr/include/alsa/pcm.h (stable ABI).
        assert_eq!(SND_PCM_STREAM_PLAYBACK, 0);
        assert_eq!(SND_PCM_ACCESS_MMAP_INTERLEAVED, 0);
        assert_eq!(SND_PCM_ACCESS_MMAP_NONINTERLEAVED, 1);
        assert_eq!(SND_PCM_ACCESS_RW_INTERLEAVED, 3);
        assert_eq!(SND_PCM_FORMAT_S16_LE, 2);
        assert_eq!(SND_PCM_FORMAT_S32_LE, 10);
        assert_eq!(SND_PCM_STATE_OPEN, 0);
        assert_eq!(SND_PCM_STATE_SETUP, 1);
        assert_eq!(SND_PCM_STATE_PREPARED, 2);
        assert_eq!(SND_PCM_STATE_RUNNING, 3);
        assert_eq!(SND_PCM_STATE_XRUN, 4);
        assert_eq!(SND_PCM_STATE_DRAINING, 5);
        assert_eq!(SND_PCM_STATE_SUSPENDED, 7);
        assert_eq!(SND_PCM_STATE_DISCONNECTED, 8);
        assert_eq!(SND_PCM_NONBLOCK, 1);
        assert_eq!(EPIPE, 32);
        assert_eq!(ESTRPIPE, 86);
        assert_eq!(EBADFD, 77);
    }

    #[test]
    fn channel_area_layout_is_bit_fields() {
        // addr at offset 0, first at 8, step at 12 (LP64 x86-64).
        assert_eq!(std::mem::offset_of!(ChannelArea, addr), 0);
        assert_eq!(std::mem::offset_of!(ChannelArea, first), 8);
        assert_eq!(std::mem::offset_of!(ChannelArea, step), 12);
        assert_eq!(std::mem::size_of::<ChannelArea>(), 16);
    }
}
