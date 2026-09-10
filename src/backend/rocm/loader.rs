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

/// Enumerate library files under `libdir` whose name starts with `prefix`
/// (e.g. every `libamdhip64.so*` version in a ROCm lib dir).
pub fn scan_dir(prefix: &str, libdir: &std::path::Path) -> Vec<String> {
    let mut found: Vec<String> = std::fs::read_dir(libdir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.path().is_file())
                .filter(|e| e.file_name().to_string_lossy().starts_with(prefix))
                .map(|e| e.path().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    found.sort();
    found
}

/// Every explicit ROCm library directory: `ROCM_LIB_PATH` entries (each
/// searched for the full prefix family) plus `/opt/rocm*/lib` and
/// `/opt/rocm*/lib64`.
pub fn rocm_lib_dirs() -> Vec<std::path::PathBuf> {
    let mut dirs: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(p) = std::env::var("ROCM_LIB_PATH") {
        for dir in p.split(':').filter(|d| !d.is_empty()) {
            dirs.push(std::path::PathBuf::from(dir));
        }
    }
    if let Ok(entries) = std::fs::read_dir("/opt") {
        let mut rocm_dirs = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .filter(|e| e.file_name().to_string_lossy().starts_with("rocm"))
            .map(|e| e.path())
            .collect::<Vec<_>>();
        rocm_dirs.sort();
        for dir in rocm_dirs {
            for sub in ["lib", "lib64"] {
                let libdir = dir.join(sub);
                if libdir.is_dir() {
                    dirs.push(libdir);
                }
            }
        }
    }
    dirs
}

/// Candidate library paths for a family of sonames, given explicit library
/// directories: the sonames themselves (ld.so cache), then every
/// matching-version file in each directory (scanning by the family prefix
/// discovers future versioned SONAMEs). Pure — no process-environment
/// access, so tests can supply their own directory without mutating global
/// state.
pub fn lib_candidates_in(sonames: &[&str], explicit_dirs: &[std::path::PathBuf]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for s in sonames {
        out.push((*s).to_string());
    }
    let mut seen: Vec<String> = Vec::new();
    for s in sonames {
        let prefix = if let Some((head, tail)) = s.rsplit_once(".so") {
            if !tail.is_empty() && tail.chars().all(|c| c == '.' || c.is_ascii_digit()) {
                let mut p = head.to_string();
                p.push_str(".so");
                p
            } else {
                (*s).to_string()
            }
        } else {
            (*s).to_string()
        };
        if seen.contains(&prefix) {
            continue;
        }
        seen.push(prefix.clone());
        for dir in explicit_dirs {
            for p in scan_dir(&prefix, dir) {
                if !out.contains(&p) {
                    out.push(p);
                }
            }
        }
    }
    out
}

/// Candidate library paths for a family of sonames using the production
/// explicit directories (`ROCM_LIB_PATH` entries + `/opt/rocm*/lib{,64}`).
pub fn lib_candidates(sonames: &[&str]) -> Vec<String> {
    lib_candidates_in(sonames, &rocm_lib_dirs())
}

/// Load a runtime soname (via ld cache + explicit ROCm library dirs) and
/// resolve a D0 table and a D1-additional table.
pub fn probe_soname_surfaces<'a>(
    soname: &str,
    d0: &[&'a str],
    d1: &[&'a str],
) -> Result<(String, Vec<&'a str>, Vec<&'a str>), String> {
    let candidates = lib_candidates(&[soname]);
    let names = candidates.iter().map(String::as_str).collect::<Vec<_>>();
    let lib = Lib::open_candidates(names)?;
    // SAFETY: the symbol pointers are used only to prove resolvability
    // inside this function while `lib` is alive.
    let resolve = |table: &[&'a str]| -> Vec<&'a str> {
        let mut missing = Vec::new();
        for sym in table {
            let found = unsafe { lib.symbol::<unsafe extern "C" fn()>(sym) }.is_some();
            if !found {
                missing.push(*sym);
            }
        }
        missing
    };
    let d0_missing = resolve(d0);
    let d1_missing = resolve(d1);
    Ok((lib.name.clone(), d0_missing, d1_missing))
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

#[cfg(test)]
mod extra_tests {
    use super::*;

    #[test]
    fn versioned_soname_family_is_discovered_in_explicit_dirs() {
        // ROCm 7 can ship libamdhip64.so.7 without an unversioned symlink;
        // the candidate builder must find it inside the explicit library
        // directories it is given (pure function: no process environment is
        // mutated by this test).
        let dir = std::env::temp_dir().join(format!(
            "vole-rocm-cand-{}-{}",
            std::process::id(),
            crate::evidence::timing::monotonic_raw_ns()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let so7 = dir.join("libamdhip64.so.7");
        std::fs::write(&so7, b"").unwrap();
        // Some unrelated file must not be picked up.
        std::fs::write(dir.join("libamdhip64_helper.txt"), b"").unwrap();
        let dirs = vec![dir.clone()];
        let cands = lib_candidates_in(
            &[
                "libamdhip64.so.7",
                "libamdhip64.so.6",
                "libamdhip64.so.5",
                "libamdhip64.so",
            ],
            &dirs,
        );
        // The unversioned-soname row alone also discovers the family (scan
        // by prefix), so a .so.7-only install is not mislabeled absent.
        let cands2 = lib_candidates_in(&["libamdhip64.so"], &dirs);
        let path = so7.to_string_lossy().into_owned();
        assert!(cands.contains(&path), "candidates: {cands:?}");
        assert!(cands2.contains(&path), "candidates: {cands2:?}");
        // Only files whose name starts with the family prefix are scanned.
        let helper = dir.join("libamdhip64_helper.txt");
        assert!(!cands.contains(&helper.to_string_lossy().into_owned()));
        // The unversioned soname is always the first candidate (ld cache).
        assert_eq!(cands2[0], "libamdhip64.so");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
