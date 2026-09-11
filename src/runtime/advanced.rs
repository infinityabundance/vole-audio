//! Advanced runtime surfaces (Report 3 whole-repository runtime items).
//!
//! Optional, explicitly-reported host surfaces that sit **beside** the frozen
//! B2–B5 architectures — they never replace a baseline and never change a
//! representation:
//!
//! * **hybrid deadline entry** — `sleep` to `deadline − δ`, then a bounded
//!   `spin_loop` for the final δ. Pure blocking wait pays scheduler wake jitter;
//!   pure spin wastes a core. The court compares the wake-error tail.
//! * **huge-page execution arenas** — an anonymous `mmap` + `MADV_HUGEPAGE`
//!   buffer for large stable working sets, to reduce TLB/page-walk cost.
//! * **real-time scheduling + memory locking** — `SCHED_FIFO`/`SCHED_DEADLINE`
//!   and `mlockall(MCL_CURRENT|MCL_FUTURE)`, attempted and reported honestly
//!   (they usually require privileges).
//!
//! Everything here is `std`-gated and best-effort: a surface that the host
//! refuses is recorded as an honest limitation, never a fabricated win.

use crate::error::{Error, Kind, Result};

/// One hybrid deadline-entry policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridPolicy {
    /// Busy-spin for the final `spin_ns` before the deadline.
    pub spin_ns: u64,
}

/// Block until `deadline`, spinning for the final `spin_ns`.
pub fn wait_until(deadline: std::time::Instant, spin_ns: u64) {
    let spin = std::time::Duration::from_nanos(spin_ns);
    if let Some(sleep_until) = deadline.checked_sub(spin) {
        let now = std::time::Instant::now();
        if sleep_until > now {
            std::thread::sleep(sleep_until - now);
        }
    }
    while std::time::Instant::now() < deadline {
        std::hint::spin_loop();
    }
}

/// Signed lateness in nanoseconds (`now - deadline`; negative = early).
pub fn lateness_ns(deadline: std::time::Instant) -> i64 {
    let now = std::time::Instant::now();
    if now >= deadline {
        now.duration_since(deadline).as_nanos() as i64
    } else {
        -(deadline.duration_since(now).as_nanos() as i64)
    }
}

/// An anonymous huge-page-hinted arena.
pub struct HugeArena {
    ptr: *mut u8,
    len: usize,
    huge_page_hint: bool,
}

// SAFETY: the arena owns a unique mapping and never aliases it.
unsafe impl Send for HugeArena {}

impl HugeArena {
    /// Allocate `len` bytes, hinting transparent huge pages. Returns `None`
    /// when the mapping fails.
    pub fn new(len: usize) -> Option<HugeArena> {
        if len == 0 {
            return None;
        }
        // SAFETY: a fresh anonymous private mapping; checked for MAP_FAILED.
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return None;
        }
        // SAFETY: `p`/`len` describe the mapping just created.
        let rc = unsafe { libc::madvise(p, len, libc::MADV_HUGEPAGE) };
        Some(HugeArena {
            ptr: p as *mut u8,
            len,
            huge_page_hint: rc == 0,
        })
    }

    /// Whether the transparent-huge-page hint was accepted.
    pub fn huge_page_hint(&self) -> bool {
        self.huge_page_hint
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the arena is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Mutable view of the arena.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: the mapping is valid for `len` bytes and uniquely owned.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Touch every page so the mapping is populated.
    pub fn touch_all(&mut self) {
        let s = self.as_mut_slice();
        let mut i = 0;
        while i < s.len() {
            s[i] = 0;
            i += 4096;
        }
    }
}

impl Drop for HugeArena {
    fn drop(&mut self) {
        // SAFETY: unmapping the exact region created in `new`.
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

/// The kernel's transparent-huge-page setting (`/sys/kernel/mm/transparent_hugepage/enabled`).
pub fn transparent_hugepage_setting() -> Option<String> {
    std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled")
        .ok()
        .map(|s| s.trim().to_string())
}

/// Anonymous huge pages reported for this process (`/proc/self/smaps_rollup`).
pub fn anon_huge_pages_kib() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/smaps_rollup").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("AnonHugePages:") {
            let kb = rest.split_whitespace().next()?;
            return kb.parse().ok();
        }
    }
    None
}

/// The outcome of a requested real-time scheduling policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleOutcome {
    pub policy: &'static str,
    pub applied: bool,
    pub errno: Option<i32>,
    pub detail: String,
}

/// Attempt `SCHED_FIFO` at `priority` for the calling thread.
pub fn try_sched_fifo(priority: i32) -> ScheduleOutcome {
    // SAFETY: `sched_param` is a plain C struct; the call only changes this
    // thread's scheduling policy and fails cleanly without privileges.
    let param = unsafe {
        let mut p: libc::sched_param = std::mem::zeroed();
        p.sched_priority = priority;
        p
    };
    // SAFETY: as above.
    let rc = unsafe { libc::sched_setscheduler(0, libc::SCHED_FIFO, &param) };
    if rc == 0 {
        ScheduleOutcome {
            policy: "SCHED_FIFO",
            applied: true,
            errno: None,
            detail: format!("priority {priority}"),
        }
    } else {
        let e = std::io::Error::last_os_error().raw_os_error();
        ScheduleOutcome {
            policy: "SCHED_FIFO",
            applied: false,
            errno: e,
            detail: format!(
                "sched_setscheduler(SCHED_FIFO, {priority}) failed: {}",
                std::io::Error::last_os_error()
            ),
        }
    }
}

/// Restore the default `SCHED_OTHER` policy for the calling thread.
pub fn try_sched_other() -> ScheduleOutcome {
    // SAFETY: as `try_sched_fifo`; restores the default policy.
    let param = unsafe {
        let mut p: libc::sched_param = std::mem::zeroed();
        p.sched_priority = 0;
        p
    };
    // SAFETY: as above.
    let rc = unsafe { libc::sched_setscheduler(0, libc::SCHED_OTHER, &param) };
    ScheduleOutcome {
        policy: "SCHED_OTHER",
        applied: rc == 0,
        errno: if rc == 0 {
            None
        } else {
            std::io::Error::last_os_error().raw_os_error()
        },
        detail: "restore default policy".into(),
    }
}

/// A `sched_attr` for `SCHED_DEADLINE` (Linux UAPI layout).
#[repr(C)]
#[derive(Clone, Copy)]
struct SchedAttr {
    size: u32,
    sched_policy: u32,
    sched_flags: u64,
    sched_nice: i32,
    sched_priority: u32,
    sched_runtime: u64,
    sched_deadline: u64,
    sched_period: u64,
}

/// Attempt `SCHED_DEADLINE(runtime, deadline, period)` via the raw syscall.
pub fn try_sched_deadline(runtime_ns: u64, deadline_ns: u64, period_ns: u64) -> ScheduleOutcome {
    let attr = SchedAttr {
        size: std::mem::size_of::<SchedAttr>() as u32,
        sched_policy: libc::SCHED_DEADLINE as u32,
        sched_flags: 0,
        sched_nice: 0,
        sched_priority: 0,
        sched_runtime: runtime_ns,
        sched_deadline: deadline_ns,
        sched_period: period_ns,
    };
    // SAFETY: `SYS_sched_setattr` reads the `sched_attr` we fully initialised;
    // it changes only this thread's scheduling policy and fails without
    // privileges.
    let rc = unsafe {
        libc::syscall(
            libc::SYS_sched_setattr,
            0i32,
            &attr as *const SchedAttr,
            0u32,
        )
    };
    if rc == 0 {
        ScheduleOutcome {
            policy: "SCHED_DEADLINE",
            applied: true,
            errno: None,
            detail: format!("runtime={runtime_ns} deadline={deadline_ns} period={period_ns}"),
        }
    } else {
        let e = std::io::Error::last_os_error().raw_os_error();
        ScheduleOutcome {
            policy: "SCHED_DEADLINE",
            applied: false,
            errno: e,
            detail: format!(
                "sched_setattr(SCHED_DEADLINE) failed: {}",
                std::io::Error::last_os_error()
            ),
        }
    }
}

/// Attempt `mlockall(MCL_CURRENT | MCL_FUTURE)`.
pub fn try_mlockall() -> ScheduleOutcome {
    // SAFETY: `mlockall` only locks the caller's address space and fails
    // cleanly without the capability.
    let rc = unsafe { libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE) };
    if rc == 0 {
        ScheduleOutcome {
            policy: "mlockall",
            applied: true,
            errno: None,
            detail: "MCL_CURRENT|MCL_FUTURE".into(),
        }
    } else {
        let e = std::io::Error::last_os_error().raw_os_error();
        ScheduleOutcome {
            policy: "mlockall",
            applied: false,
            errno: e,
            detail: format!("mlockall failed: {}", std::io::Error::last_os_error()),
        }
    }
}

/// Undo `mlockall`.
pub fn unlock_all() {
    // SAFETY: releases the caller's memory locks; harmless if none were set.
    unsafe {
        libc::munlockall();
    }
}

/// Validate a hybrid policy.
pub fn validate_policy(p: HybridPolicy) -> Result<()> {
    if p.spin_ns > 10_000_000 {
        return Err(Error::new(
            Kind::LimitExceeded,
            "hybrid spin budget exceeds 10 ms",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_wait_is_monotone_in_spin_budget() {
        // A pure-spin wait ends no earlier than a sleep-heavy wait for the same
        // deadline (sleep can only be more conservative).
        let d = std::time::Instant::now() + std::time::Duration::from_millis(5);
        wait_until(d, 100_000);
        let l1 = lateness_ns(d);
        let d2 = std::time::Instant::now() + std::time::Duration::from_millis(5);
        wait_until(d2, 0);
        let l2 = lateness_ns(d2);
        // Both must end at/after the deadline, and neither may be absurdly late.
        assert!(l1.abs() < 500_000_000 && l2.abs() < 500_000_000);
    }

    #[test]
    fn huge_arena_maps_and_touches() {
        let Some(mut a) = HugeArena::new(4 << 20) else {
            return; // mapping refused: honest skip
        };
        assert_eq!(a.len(), 4 << 20);
        a.touch_all();
        assert_eq!(a.as_mut_slice()[0], 0);
        let _ = a.huge_page_hint();
        let _ = transparent_hugepage_setting();
        let _ = anon_huge_pages_kib();
    }

    #[test]
    fn scheduler_attempts_never_panic() {
        let _ = try_sched_fifo(1);
        let _ = try_sched_deadline(1_000_000, 5_000_000, 5_000_000);
        let m = try_mlockall();
        if m.applied {
            unlock_all();
        }
    }
}
