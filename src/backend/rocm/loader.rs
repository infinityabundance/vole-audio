//! Minimal audited dlopen loader for the ROCm runtime (Phase I probe; Phase J
//! will reuse it for the real module/kernel surface).
//!
//! Same discipline as `backend::cuda::ffi`: `dlopen` with `RTLD_NOW |
//! RTLD_LOCAL`, the handle kept alive for exactly as long as the resolved
//! function pointers are used, `dlsym` + transmute of the symbol address,
//! and typed failures — no unsafe beyond the loader itself, every block
//! carries its SAFETY rationale.

use std::ffi::CString;
use std::os::raw::c_void;

/// A dlopen'd library handle (RAII: dlclose on drop). Function pointers
/// resolved from it must not outlive the handle.
pub struct Lib {
    handle: *mut c_void,
    /// Soname/path this handle was opened from (evidence).
    pub name: String,
}

// SAFETY: Lib owns the only reference to the handle; moving it between
// threads is safe because dlclose happens on drop in the thread that drops.
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

impl Lib {
    /// Open `name` with RTLD_NOW | RTLD_LOCAL.
    pub fn open(name: &str) -> Result<Lib, String> {
        // SAFETY: dlopen initializes the library and returns an owned
        // handle; CString has no interior NUL. The handle is stored and
        // dlclose'd on drop, and all resolved pointers die with it.
        let cname = CString::new(name).map_err(|_| "library name has no NUL".to_string())?;
        let handle = unsafe { libc::dlopen(cname.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            return Err(format!("dlopen {name} failed"));
        }
        Ok(Lib {
            handle,
            name: name.to_string(),
        })
    }

    /// Open the first candidate that loads. `dlopen(soname)` uses the ld.so
    /// cache (covers distro/multiarch installs registered with ldconfig);
    /// the absolute-path candidates cover custom installs the cache does
    /// not know (e.g. /opt/rocm before ldconfig, ROCM_LIB_PATH overrides).
    pub fn open_candidates<'a>(
        candidates: impl IntoIterator<Item = &'a str>,
    ) -> Result<Lib, String> {
        let mut last = String::new();
        for c in candidates {
            match Lib::open(c) {
                Ok(lib) => return Ok(lib),
                Err(e) => last = e,
            }
        }
        Err(if last.is_empty() {
            "no candidate library loaded".to_string()
        } else {
            last
        })
    }

    /// Resolve one exported symbol (plain C names; no demangling).
    ///
    /// # SAFETY
    /// The returned pointer is only valid while `self` (and therefore the
    /// handle) is alive; callers must respect that lifetime.
    pub unsafe fn symbol<T>(&self, name: &str) -> Option<T> {
        // SAFETY: dlsym returns *mut c_void; transmute to the fn pointer
        // type (mirrors backend::cuda::ffi::resolve). The handle outlives
        // the returned pointer by the caller's contract.
        let cname = CString::new(name).expect("symbol name has no NUL");
        let p = unsafe { libc::dlsym(self.handle, cname.as_ptr()) };
        if p.is_null() {
            None
        } else {
            Some(unsafe { std::mem::transmute_copy(&p) })
        }
    }
}

impl Drop for Lib {
    fn drop(&mut self) {
        // SAFETY: the handle was returned by dlopen and is still owned by
        // self; no resolved symbol outlives this drop by contract.
        unsafe {
            libc::dlclose(self.handle);
        }
    }
}

/// Load a runtime soname (via ld cache + custom-install candidate paths)
/// and resolve a required entry symbol.
/// Returns `Ok(lib_name)` when both succeed, `Err(reason)` otherwise.
pub fn probe_soname_symbol(soname: &str, symbol: &str) -> Result<String, String> {
    let mut candidates = vec![soname.to_string()];
    // ROCM_LIB_PATH override (colon-separated absolute dirs).
    if let Ok(p) = std::env::var("ROCM_LIB_PATH") {
        for dir in p.split(':').filter(|d| !d.is_empty()) {
            candidates.push(format!("{dir}/{soname}"));
        }
    }
    // /opt/rocm* installs the ld cache may not know (auto-discovered).
    if let Ok(entries) = std::fs::read_dir("/opt") {
        let mut rocm_dirs = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with("rocm"))
            .map(|e| e.path())
            .collect::<Vec<_>>();
        rocm_dirs.sort();
        for dir in rocm_dirs {
            for sub in ["lib", "lib64"] {
                let p = dir.join(sub).join(soname);
                if p.exists() {
                    candidates.push(p.to_string_lossy().into_owned());
                }
            }
        }
    }
    let names = candidates.iter().map(String::as_str).collect::<Vec<_>>();
    let lib = Lib::open_candidates(names)?;
    // SAFETY: the symbol pointer is used only to prove resolvability inside
    // this function while `lib` is alive.
    let found = unsafe { lib.symbol::<unsafe extern "C" fn()>(symbol) }.is_some();
    if found {
        Ok(lib.name.clone())
    } else {
        Err(format!(
            "{soname} loaded but missing required symbol {symbol}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_library_is_a_typed_error() {
        let r = Lib::open("libvole_audio_definitely_absent.so.99");
        assert!(r.is_err());
    }

    #[test]
    fn libc_symbols_resolve() {
        // libc is always loadable; exercises the dlsym path on this host.
        let lib = Lib::open("libc.so.6").expect("libc dlopens");
        // SAFETY: test-only; the pointer does not outlive `lib`.
        let sym = unsafe { lib.symbol::<unsafe extern "C" fn() -> *mut c_void>("malloc") };
        assert!(sym.is_some());
        let missing = unsafe { lib.symbol::<unsafe extern "C" fn()>("vole_no_such_symbol") };
        assert!(missing.is_none());
    }
}
