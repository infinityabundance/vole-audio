//! CUDA D1 host-registration surface (Phase H) — `cuMemHostRegister` against
//! an *existing* endpoint mapping (the ALSA mmap region), never a substitute
//! pinned buffer.
//!
//! Discipline (paper §"D1 experiment"): the exact region the endpoint maps is
//! what gets registered. If registration fails, the exact driver result is
//! recorded and classified; the court never "fixes" the failure by allocating
//! a new CUDA pinned buffer and pretending it is the endpoint region.
//!
//! Everything here is evidence-first: an attempt records the exact base,
//! length, page alignment, flags, driver result code and string, pointer
//! attributes observed after success, and the unregister path. Pure helpers
//! (page rounding, rc classification) are unit-tested; the driver calls are
//! thin audited wrappers over `ffi::Fns`.

use crate::backend::cuda::driver::CudaContext;
use crate::backend::cuda::ffi::{
    CU_MEMHOSTREGISTER_DEVICEMAP, CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
    CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_PERMITTED,
    CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_OUT_OF_MEMORY, CUdeviceptr, Fns,
    POINTER_ATTRIBUTE_DEVICE_POINTER, POINTER_ATTRIBUTE_MAPPED, POINTER_ATTRIBUTE_MEMORY_TYPE,
    cuda_error,
};
use crate::error::Error;
use crate::status::Verdict;
use serde::{Deserialize, Serialize};
use std::ffi::{c_int, c_void};
use std::sync::Arc;

/// Registration flag set used by the D1 attempt: DEVICEMAP (map into the
/// CUDA address space so the device can write the region). PORTABLE is
/// deliberately NOT set: the registration is process-local by design, and
/// portability across contexts is not part of the claim.
pub const D1_REGISTER_FLAGS: u32 = CU_MEMHOSTREGISTER_DEVICEMAP;

/// Host page size used for alignment evidence (queried once).
pub fn host_page_size() -> usize {
    // SAFETY: sysconf(_SC_PAGESIZE) is side-effect free; 4096 is a sane
    // fallback if it ever returned <= 0.
    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if v > 0 { v as usize } else { 4096 }
}

/// Round `addr` down to a page boundary.
pub const fn align_down(addr: usize, page: usize) -> usize {
    addr & !(page - 1)
}

/// Round `addr + len` up to a page boundary (checked; `usize::MAX` safe).
pub fn align_up(addr: usize, len: usize, page: usize) -> Option<usize> {
    let end = addr.checked_add(len)?;
    let rem = end % page;
    Some(if rem == 0 { end } else { end + (page - rem) })
}

/// Registration range evidence: the exact range handed to
/// `cuMemHostRegister` plus its page relationship. The range never extends
/// beyond the caller's declared legal mapping (`base..base+len`): we only
/// report alignment facts; the caller decides what is legal to register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterRange {
    /// Exact mapping base (host address).
    pub base: usize,
    /// Exact mapping length in bytes (as mapped by the endpoint).
    pub len: usize,
    /// Host page size used for the alignment analysis.
    pub page: usize,
    /// base % page == 0.
    pub base_page_aligned: bool,
    /// (len % page == 0) — the endpoint ring often is; recorded, not assumed.
    pub len_page_aligned: bool,
}

impl RegisterRange {
    pub fn analyze(base: usize, len: usize, page: usize) -> RegisterRange {
        RegisterRange {
            base,
            len,
            page,
            base_page_aligned: base.is_multiple_of(page),
            len_page_aligned: len.is_multiple_of(page),
        }
    }

    /// Human "0x{base:x}+{len}" used in receipts.
    pub fn label(&self) -> String {
        format!("0x{:x}+{}", self.base, self.len)
    }
}

/// One `cuPointerGetAttribute` query result: exact attribute, rc, driver
/// string, and decoded value. A failed query is evidence, never a silent
/// `None`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PointerQuery {
    /// Which address was queried: "device-pointer" or "host-pointer".
    pub target: String,
    /// Attribute name, e.g. "MEMORY_TYPE".
    pub attribute: String,
    /// Exact driver rc (0 = success).
    pub rc: i32,
    /// Driver error string when rc != 0 (else "").
    pub message: String,
    /// Decoded value when the query succeeded.
    pub value: Option<String>,
}

/// Pointer-attribute evidence for a registered mapping. Every attribute is
/// queried against both the device pointer (`cuMemHostGetDevicePointer`) and
/// the original host pointer (unified addressing); every query records its
/// own rc + driver string + value.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PointerEvidence {
    pub queries: Vec<PointerQuery>,
}

impl PointerEvidence {
    /// Query attributes of a registered host range: `host_ptr` is the exact
    /// pointer registered, `dev_ptr` the pointer returned by
    /// `cuMemHostGetDevicePointer`. Best effort per query: each row records
    /// rc + driver string + value.
    pub fn query(ctx: &Arc<CudaContext>, host_ptr: usize, dev_ptr: CUdeviceptr) -> PointerEvidence {
        // Best effort: make the owning context current so the attribute
        // queries target it. A failure surfaces as per-query rc rows, never a
        // panic.
        let _guard = ctx.enter().ok();
        let fns = &ctx.fns;
        let mut ev = PointerEvidence::default();
        let attribute_name = |a: c_int| -> &'static str {
            match a {
                POINTER_ATTRIBUTE_MEMORY_TYPE => "MEMORY_TYPE",
                POINTER_ATTRIBUTE_DEVICE_POINTER => "DEVICE_POINTER",
                POINTER_ATTRIBUTE_MAPPED => "MAPPED",
                _ => "?",
            }
        };
        // Each attribute is queried against both addresses (device pointer
        // from the registration, and the host pointer itself in UVA).
        for (target, ptr) in [
            ("device-pointer", dev_ptr),
            ("host-pointer", host_ptr as CUdeviceptr),
        ] {
            for attribute in [
                POINTER_ATTRIBUTE_MEMORY_TYPE,
                POINTER_ATTRIBUTE_DEVICE_POINTER,
                POINTER_ATTRIBUTE_MAPPED,
            ] {
                // int- or pointer-sized wire value (8 bytes covers both).
                let mut data = [0u8; 8];
                // SAFETY: the driver writes at most 8 bytes (int or device
                // pointer); `data` outlives the call.
                let rc = unsafe {
                    (fns.cuPointerGetAttribute.expect("bound"))(
                        data.as_mut_ptr() as *mut c_void,
                        attribute,
                        ptr,
                    )
                };
                let mut row = PointerQuery {
                    target: target.to_string(),
                    attribute: attribute_name(attribute).to_string(),
                    rc,
                    message: String::new(),
                    value: None,
                };
                if rc == 0 {
                    row.value = Some(decode_attr(attribute, &data));
                } else {
                    row.message = crate::backend::cuda::ffi::error_string(fns, rc);
                }
                ev.queries.push(row);
            }
        }
        ev
    }
}

/// Decode a successful attribute payload.
fn decode_attr(attribute: c_int, data: &[u8; 8]) -> String {
    match attribute {
        POINTER_ATTRIBUTE_MEMORY_TYPE => {
            use crate::backend::cuda::ffi::{
                CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST, CU_MEMORYTYPE_UNIFIED,
            };
            let v = i32::from_le_bytes(data[..4].try_into().expect("4 bytes"));
            if v == CU_MEMORYTYPE_HOST {
                "HOST".into()
            } else if v == CU_MEMORYTYPE_DEVICE {
                "DEVICE".into()
            } else if v == CU_MEMORYTYPE_UNIFIED {
                "UNIFIED".into()
            } else {
                format!("UNKNOWN({v})")
            }
        }
        POINTER_ATTRIBUTE_DEVICE_POINTER => {
            format!("0x{:x}", u64::from_le_bytes(*data))
        }
        _ => {
            let v = i32::from_le_bytes(data[..4].try_into().expect("4 bytes"));
            format!("{v}")
        }
    }
}

impl PointerEvidence {
    /// Convenience: first successful MEMORY_TYPE query value, if any.
    pub fn memory_type(&self) -> Option<&str> {
        self.queries.iter().find_map(|q| {
            if q.attribute == "MEMORY_TYPE" && q.rc == 0 {
                q.value.as_deref()
            } else {
                None
            }
        })
    }
}

/// A registered host-memory range (RAII): unregistered on drop. Retains the
/// context, so it cannot be unregistered against a destroyed context.
pub struct HostRegistration {
    pub ctx: Arc<CudaContext>,
    /// Exact host pointer passed to cuMemHostRegister.
    pub host_ptr: *mut c_void,
    /// Exact byte length passed to cuMemHostRegister.
    pub bytes: usize,
    /// Flags passed (D1_REGISTER_FLAGS).
    pub flags: u32,
    /// Device-visible pointer for `host_ptr` (cuMemHostGetDevicePointer),
    /// valid when `device_ptr.is_some()`.
    pub device_ptr: Option<CUdeviceptr>,
}

/// Result of one registration attempt: success carries the RAII handle;
/// failure carries the exact driver rc + message (never silently swallowed).
#[allow(clippy::large_enum_variant)] // the success payload is the RAII handle itself
pub enum RegistrationAttempt {
    Registered(HostRegistration),
    /// The driver API surface is missing a required symbol.
    MissingSymbol(String),
    Failed {
        rc: c_int,
        message: String,
    },
}

/// Attempt to register exactly `[base, base+len)` of an existing mapping.
///
/// # SAFETY
/// `base..base+len` must be a live, mapped, writable host range for the whole
/// call and for the lifetime of the returned `HostRegistration`; the caller
/// (the D1 court) holds the ALSA mapping open for that window.
pub unsafe fn attempt_register(
    ctx: &Arc<CudaContext>,
    base: usize,
    len: usize,
) -> RegistrationAttempt {
    // The registration targets the calling thread's *current* context, so make
    // the owning context current for the duration (rc 201 if the driver
    // refuses, which `classify` reports as a context-binding defect).
    let _guard = match ctx.enter() {
        Ok(g) => g,
        Err(e) => {
            return RegistrationAttempt::Failed {
                rc: CUDA_ERROR_INVALID_CONTEXT,
                message: format!("cuCtxPushCurrent: {e}"),
            };
        }
    };
    let fns = &ctx.fns;
    if !fns.d1_surface_complete() {
        let missing = [
            ("cuMemHostRegister", fns.cuMemHostRegister.is_some()),
            ("cuMemHostUnregister", fns.cuMemHostUnregister.is_some()),
            (
                "cuMemHostGetDevicePointer",
                fns.cuMemHostGetDevicePointer.is_some(),
            ),
            ("cuPointerGetAttribute", fns.cuPointerGetAttribute.is_some()),
        ]
        .iter()
        .filter(|(_, present)| !present)
        .map(|(n, _)| *n)
        .collect::<Vec<_>>()
        .join(", ");
        return RegistrationAttempt::MissingSymbol(missing);
    }
    let host_ptr = base as *mut c_void;
    // SAFETY: caller guarantees the range is live and writable for the
    // registration (and unregister below).
    let rc = unsafe { (fns.cuMemHostRegister.expect("bound"))(host_ptr, len, D1_REGISTER_FLAGS) };
    if rc != 0 {
        return RegistrationAttempt::Failed {
            rc,
            message: crate::backend::cuda::ffi::error_string(fns, rc),
        };
    }
    // cuMemHostGetDevicePointer(flags = 0) — the only legal value.
    let mut dptr: CUdeviceptr = 0;
    // SAFETY: out-param; the pointer was just registered.
    let rc2 = unsafe { (fns.cuMemHostGetDevicePointer.expect("bound"))(&mut dptr, host_ptr, 0) };
    if rc2 != 0 {
        // Registration succeeded but the device pointer is unobtainable: the
        // region is not usable from the device. Unregister (best effort) and
        // report the exact failure.
        // SAFETY: unregister the pointer registered above.
        unsafe { (fns.cuMemHostUnregister.expect("bound"))(host_ptr) };
        return RegistrationAttempt::Failed {
            rc: rc2,
            message: crate::backend::cuda::ffi::error_string(fns, rc2),
        };
    }
    RegistrationAttempt::Registered(HostRegistration {
        ctx: ctx.clone(),
        host_ptr,
        bytes: len,
        flags: D1_REGISTER_FLAGS,
        device_ptr: Some(dptr),
    })
}

impl HostRegistration {
    /// Classify a registration failure rc into the evidence vocabulary.
    /// `symbol_present`/`host_register_supported` come from the probe; the
    /// classifier never invents support that was not probed.
    pub fn classify(
        rc: c_int,
        symbol_present: bool,
        host_register_supported: bool,
    ) -> (Verdict, &'static str) {
        use crate::status::Verdict as V;
        if !symbol_present {
            return (
                V::UnsupportedByApi,
                "cuMemHostRegister symbol not in this driver",
            );
        }
        if !host_register_supported {
            return (
                V::UnsupportedByHardware,
                "device attribute HOST_REGISTER_SUPPORTED is false",
            );
        }
        match rc {
            CUDA_ERROR_NOT_SUPPORTED => (
                V::UnsupportedByApi,
                "rc 801 CUDA_ERROR_NOT_SUPPORTED: this memory class is not registerable",
            ),
            CUDA_ERROR_INVALID_VALUE => (
                V::UnsupportedByApi,
                "rc 1 CUDA_ERROR_INVALID_VALUE: range rejected (not user-pageable \
                 host memory and/or alignment); recorded exactly, no substitute buffer",
            ),
            CUDA_ERROR_NOT_PERMITTED => (
                V::UnsupportedByApi,
                "rc 800 CUDA_ERROR_NOT_PERMITTED: policy/security refusal",
            ),
            CUDA_ERROR_OUT_OF_MEMORY => (
                V::Inconclusive,
                "rc 2 CUDA_ERROR_OUT_OF_MEMORY: environment resource state",
            ),
            CUDA_ERROR_INVALID_CONTEXT => (
                V::Inconclusive,
                "rc 201 CUDA_ERROR_INVALID_CONTEXT: context binding defect",
            ),
            CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => (
                V::Inconclusive,
                "rc 712 already-registered: double registration (court defect)",
            ),
            _ => (
                V::Inconclusive,
                "unclassified rc; exact rc + driver string recorded in the receipt",
            ),
        }
    }
}

impl Drop for HostRegistration {
    fn drop(&mut self) {
        if !self.host_ptr.is_null() {
            // SAFETY: unregister exactly what was registered; the caller
            // guarantees the mapping outlives this drop. Enter the owning
            // context so the unregister targets it.
            let _ = self.ctx.enter();
            unsafe { (self.ctx.fns.cuMemHostUnregister.expect("bound"))(self.host_ptr) };
        }
    }
}

/// Build a driver `Error` for an unexpected registration-path failure.
pub fn reg_error(fns: &Fns, what: &str, rc: c_int) -> Error {
    cuda_error(fns, what, rc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_rounding_is_exact() {
        let page = 4096usize;
        assert_eq!(align_down(0x1234_5000, page), 0x1234_5000);
        assert_eq!(align_down(0x1234_5123, page), 0x1234_5000);
        assert_eq!(align_up(0x1234_5000, 4096, page), Some(0x1234_5000 + 4096));
        assert_eq!(align_up(0x1234_5000, 1, page), Some(0x1234_6000));
        assert_eq!(align_up(0x1234_5000, 0, page), Some(0x1234_5000));
        assert_eq!(align_up(usize::MAX - 3, 8, page), None, "overflow is None");
    }

    #[test]
    fn range_analysis_reports_alignment() {
        let page = 4096usize;
        let a = RegisterRange::analyze(0x1000, 8192, page);
        assert!(a.base_page_aligned && a.len_page_aligned);
        let b = RegisterRange::analyze(0x1004, 8192, page);
        assert!(!b.base_page_aligned);
        assert_eq!(b.label(), "0x1004+8192");
    }

    #[test]
    fn registration_classifier_is_total_and_honest() {
        // symbol absent beats everything
        assert_eq!(
            HostRegistration::classify(0, false, true).0,
            Verdict::UnsupportedByApi
        );
        // device says no host-register support
        assert_eq!(
            HostRegistration::classify(0, true, false).0,
            Verdict::UnsupportedByHardware
        );
        // documented negative codes map to UNSUPPORTED_BY_API with reasons
        for rc in [
            CUDA_ERROR_NOT_SUPPORTED,
            CUDA_ERROR_INVALID_VALUE,
            CUDA_ERROR_NOT_PERMITTED,
        ] {
            assert_eq!(
                HostRegistration::classify(rc, true, true).0,
                Verdict::UnsupportedByApi,
                "rc {rc}"
            );
        }
        // resource/state codes stay INCONCLUSIVE (never guessed)
        for rc in [CUDA_ERROR_OUT_OF_MEMORY, 999] {
            assert_eq!(
                HostRegistration::classify(rc, true, true).0,
                Verdict::Inconclusive,
                "rc {rc}"
            );
        }
    }

    #[test]
    fn d1_flags_are_the_documented_set() {
        // DEVICEMAP only; PORTABLE deliberately absent (process-local claim).
        assert_eq!(
            D1_REGISTER_FLAGS,
            crate::backend::cuda::ffi::CU_MEMHOSTREGISTER_DEVICEMAP
        );
    }
}
