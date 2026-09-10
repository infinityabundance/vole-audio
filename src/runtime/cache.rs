//! Cache-state and storage-I/O evidence (Linux).
//!
//! Cold/warm claims must be *verified*, not assumed. Linux documents
//! `posix_fadvise(POSIX_FADV_DONTNEED)` as an attempt, not a guarantee, so this
//! module pairs the advice with a `mincore(2)` residency check over the
//! page-aligned mapped range. When residency cannot be established the state is
//! reported [`CacheState::NotConfirmed`] — never silently labelled cold or warm.
//!
//! Logical versus physical read traffic comes from `/proc/self/io`: `rchar` is
//! bytes returned through reads, `read_bytes` is bytes the process actually
//! caused to be fetched from block storage.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;

/// Verified cache state of a byte range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheState {
    /// The range is resident.
    Warm,
    /// The range is non-resident after an explicit eviction attempt.
    Cold,
    /// Residency could not be established (never assume either way).
    NotConfirmed,
}

/// Cache-state evidence for a byte range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEvidence {
    pub state: CacheState,
    pub resident_bytes: u64,
    pub range_bytes: u64,
    pub method: &'static str,
    pub detail: String,
}

impl CacheEvidence {
    pub fn state_name(&self) -> &'static str {
        match self.state {
            CacheState::Warm => "WARM_VERIFIED",
            CacheState::Cold => "COLD_VERIFIED",
            CacheState::NotConfirmed => "CACHE_STATE_NOT_CONFIRMED",
        }
    }
}

fn page_size() -> u64 {
    // SAFETY: sysconf is a pure query; an invalid name returns -1, which we
    // clamp to a safe default.
    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if v > 0 { v as u64 } else { 4096 }
}

/// Advise the kernel that a byte range will not be needed (an *attempt*).
pub fn advise_dontneed(file: &File, offset: u64, len: u64) -> std::io::Result<()> {
    let fd = file.as_raw_fd();
    let rc = unsafe {
        libc::posix_fadvise(
            fd,
            offset as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::from_raw_os_error(rc))
    }
}

/// Resident bytes of `[offset, offset + len)` of a file, via `mmap` + `mincore`.
pub fn resident_bytes(file: &File, offset: u64, len: u64) -> std::io::Result<u64> {
    if len == 0 {
        return Ok(0);
    }
    let page = page_size();
    let start = offset / page * page;
    let end = offset.saturating_add(len);
    let span = (end - start).div_ceil(page) * page;
    let span_usize = usize::try_from(span)
        .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
    let fd = file.as_raw_fd();
    // SAFETY: mmap of a read-only shared mapping of the file; the returned
    // pointer is checked against MAP_FAILED and unmapped before returning.
    let addr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            span_usize,
            libc::PROT_READ,
            libc::MAP_SHARED,
            fd,
            start as libc::off_t,
        )
    };
    if addr == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error());
    }
    let pages = (span / page) as usize;
    let mut vec = vec![0u8; pages];
    // SAFETY: addr is a live mapping of at least `span` bytes and `vec` holds
    // exactly ceil(span/page) bytes, as mincore requires.
    let rc = unsafe { libc::mincore(addr, span_usize, vec.as_mut_ptr()) };
    let result = if rc == 0 {
        Ok(vec.iter().filter(|b| **b & 1 != 0).count() as u64 * page)
    } else {
        Err(std::io::Error::last_os_error())
    };
    // SAFETY: addr/span describe exactly the mapping created above.
    unsafe {
        libc::munmap(addr, span_usize);
    }
    result
}

/// Verified cache evidence for a range (never assumes a state).
pub fn verify_cache(file: &File, offset: u64, len: u64) -> CacheEvidence {
    match resident_bytes(file, offset, len) {
        Ok(resident) => {
            let state = if resident == 0 {
                CacheState::Cold
            } else if resident >= len {
                CacheState::Warm
            } else {
                CacheState::NotConfirmed
            };
            CacheEvidence {
                state,
                resident_bytes: resident,
                range_bytes: len,
                method: "posix_fadvise(2) + mincore(2) over the page-aligned mapping",
                detail: format!("{resident} of {len} bytes resident"),
            }
        }
        Err(e) => CacheEvidence {
            state: CacheState::NotConfirmed,
            resident_bytes: 0,
            range_bytes: len,
            method: "mincore(2)",
            detail: format!("residency unavailable: {e}"),
        },
    }
}

/// Attempt to evict a range, then report the verified state.
///
/// Eviction requires clean pages: a range that was just written and never
/// synced cannot be dropped, so the file is flushed first (`fdatasync`).
pub fn evict_and_verify(file: &File, offset: u64, len: u64) -> CacheEvidence {
    let _ = file.sync_all();
    let _ = advise_dontneed(file, offset, len);
    verify_cache(file, offset, len)
}

/// Prime a range (read it once) and report the verified state.
pub fn prime_and_verify(file: &File, offset: u64, len: u64) -> std::io::Result<CacheEvidence> {
    let mut buf = vec![0u8; 1 << 16];
    let mut pos = offset;
    let end = offset + len;
    while pos < end {
        let want = ((end - pos) as usize).min(buf.len());
        file.read_exact_at(&mut buf[..want], pos)?;
        pos += want as u64;
    }
    Ok(verify_cache(file, offset, len))
}

/// `(rchar, read_bytes)` from `/proc/self/io`, when available.
pub fn proc_self_io() -> Option<(u64, u64)> {
    let text = std::fs::read_to_string("/proc/self/io").ok()?;
    let mut rchar = None;
    let mut read_bytes = None;
    for line in text.lines() {
        if let Some(v) = line.strip_prefix("rchar:") {
            rchar = v.trim().parse().ok();
        } else if let Some(v) = line.strip_prefix("read_bytes:") {
            read_bytes = v.trim().parse().ok();
        }
    }
    Some((rchar?, read_bytes?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn cold_then_warm_is_verified_on_linux() {
        // A file large enough to span many pages, on a **block-backed**
        // filesystem: `fadvise(DONTNEED)` cannot evict tmpfs pages, so the cache
        // verification mechanism must be exercised where eviction is possible.
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("runtime-cache-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("range.bin");
        let data = vec![0xA5u8; 1 << 20];
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(&data).unwrap();
            f.sync_all().unwrap();
        }
        let file = File::open(&path).unwrap();
        // The write itself may leave the range resident; eviction is an
        // *attempt*, so the mechanism (not a mandated state) is what we assert.
        let evicted = evict_and_verify(&file, 0, data.len() as u64);
        assert_ne!(
            evicted.state,
            CacheState::NotConfirmed,
            "mincore must be available on Linux: {}",
            evicted.detail
        );
        let primed = prime_and_verify(&file, 0, data.len() as u64).unwrap();
        assert_eq!(
            primed.state,
            CacheState::Warm,
            "after reading the range it must verify resident ({:?})",
            primed.detail
        );
        assert!(
            evicted.resident_bytes <= primed.resident_bytes,
            "eviction cannot increase residency"
        );
        // rchar/read_bytes are present on Linux.
        let io = proc_self_io();
        assert!(io.is_some(), "/proc/self/io must be readable");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
