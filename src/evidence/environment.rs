//! Host environment capture for evidence receipts.
//!
//! Best-effort, never fatal: every field is `Option` and capture cannot fail
//! the run. Receipts must state the actual environment (kernel, scheduler,
//! governor, SIMD width available, ...) because every performance statement is
//! conditioned on it.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

/// Linux kernel + OS identification via uname(2) and /etc/os-release.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OsInfo {
    pub sysname: Option<String>,
    pub release: Option<String>,
    pub version: Option<String>,
    pub machine: Option<String>,
    pub nodename: Option<String>,
    pub os_pretty_name: Option<String>,
}

/// CPU description (model + SIMD-relevant flags).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuInfo {
    pub model: Option<String>,
    /// Sorted unique flag list from /proc/cpuinfo (first CPU).
    pub flags: Vec<String>,
    /// Feature-presence booleans that courts branch on.
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_neon: bool,
}

/// Memory description.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemInfo {
    pub total_bytes: Option<u64>,
}

/// Snapshot of the machine + toolchain a receipt was produced on.
///
/// Canonicalism note: new optional fields are appended at the *end* and
/// serialized only when present, so archived receipts (created before a field
/// existed) still parse and re-verify byte-identically.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Environment {
    pub os: OsInfo,
    pub cpu: CpuInfo,
    pub memory: MemInfo,
    pub git_commit: Option<String>,
    /// Source-tree dirty state. Receipt-output writes (`receipts/`) are
    /// excluded from this computation: writing evidence must never by itself
    /// mark the very tree it attests as dirty.
    pub git_dirty: Option<bool>,
    pub git_branch: Option<String>,
    pub rustc_version: Option<String>,
    pub crate_version: String,
    pub target_arch: String,
    pub target_os: String,
    /// scaling_governor of cpu0, if readable.
    pub cpu_governor: Option<String>,
    /// DMI BIOS version, if readable.
    pub bios_version: Option<String>,
    /// SHA-256-class content hash of the *committed* source tree
    /// (`git rev-parse HEAD^{tree}`): an immutable anchor that is meaningful
    /// even when the work tree is dirty (the commit the tree hash names is
    /// stable regardless of uncommitted changes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_tree_sha: Option<String>,
}

fn read_first_line(path: &str) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim_end().to_string())
}

fn read_value_from_lines(content: &str, key: &str) -> Option<String> {
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix(key) {
            let rest = rest.trim();
            let rest = rest.strip_prefix(':').unwrap_or(rest).trim();
            return Some(rest.to_string());
        }
    }
    None
}

impl OsInfo {
    pub fn capture() -> Self {
        // SAFETY: uname writes into the provided libc::utsname buffer; the
        // struct is valid for writes of the platform-defined fixed size.
        let mut u: libc::utsname = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::uname(&mut u) };
        let fld = |c: &[i8]| -> Option<String> {
            let bytes: Vec<u8> = c
                .iter()
                .take_while(|&&b| b != 0)
                .map(|&b| b as u8)
                .collect();
            if bytes.is_empty() {
                None
            } else {
                String::from_utf8(bytes).ok()
            }
        };
        let mut info = Self {
            sysname: fld(&u.sysname),
            release: fld(&u.release),
            version: fld(&u.version),
            machine: fld(&u.machine),
            nodename: fld(&u.nodename),
            os_pretty_name: None,
        };
        let _ = rc;
        if let Ok(osrel) = std::fs::read_to_string("/etc/os-release") {
            for line in osrel.lines() {
                if let Some(rest) = line.strip_prefix("PRETTY_NAME=") {
                    info.os_pretty_name = Some(rest.trim_matches('"').to_string());
                    break;
                }
            }
        }
        info
    }
}

impl CpuInfo {
    pub fn capture() -> Self {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
        let model = read_value_from_lines(&cpuinfo, "model name");
        let flags_line = read_value_from_lines(&cpuinfo, "flags");
        let mut flags: Vec<String> = flags_line
            .map(|f| {
                f.split_whitespace()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        flags.sort();
        flags.dedup();
        let has = |name: &str| flags.iter().any(|f| f == name);
        Self {
            model,
            has_avx2: has("avx2"),
            has_avx512: has("avx512f"),
            has_neon: has("asimd"),
            flags,
        }
    }
}

impl MemInfo {
    pub fn capture() -> Self {
        let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let total = read_value_from_lines(&meminfo, "MemTotal").and_then(|v| {
            let kb: u64 = v.split_whitespace().next()?.parse().ok()?;
            Some(kb * 1024)
        });
        Self { total_bytes: total }
    }
}

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let s = s.trim();
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

/// `git status --porcelain` limited to tracked-source dirtiness.
///
/// The `receipts/` output directory is excluded by pathspec so that writing
/// evidence (immutable, `create_new`) never by itself marks the attested tree
/// dirty. Any other uncommitted source/untracked change still does.
fn git_source_dirty() -> Option<bool> {
    let out = Command::new("git")
        .args([
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            ".",
            ":(exclude)receipts",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    Some(!s.trim().is_empty())
}

impl Environment {
    /// Capture the current machine/toolchain snapshot.
    pub fn capture() -> Self {
        let os = OsInfo::capture();
        let cpu = CpuInfo::capture();
        let memory = MemInfo::capture();
        let dirty = git_source_dirty();
        let gov = read_first_line("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
        let bios = read_first_line("/sys/class/dmi/id/bios_version");
        Self {
            git_commit: git(&["rev-parse", "HEAD"]),
            git_dirty: dirty,
            git_branch: git(&["rev-parse", "--abbrev-ref", "HEAD"]),
            rustc_version: rustc_version(),
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            target_arch: std::env::consts::ARCH.to_string(),
            target_os: std::env::consts::OS.to_string(),
            cpu_governor: gov,
            bios_version: bios,
            os,
            cpu,
            memory,
            git_tree_sha: git(&["rev-parse", "HEAD^{tree}"]),
        }
    }

    /// True if running inside a git work tree with a known commit.
    pub fn git_identity(&self) -> Option<String> {
        match (&self.git_commit, self.git_dirty) {
            (Some(c), Some(d)) => Some(format!("{}{}", c, if d { "-dirty" } else { "" })),
            _ => None,
        }
    }
}

/// Ask the pinned rustc for its version string.
pub fn rustc_version() -> Option<String> {
    let exe = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let out = Command::new(exe).arg("--version").output().ok()?;
    if out.status.success() {
        String::from_utf8(out.stdout)
            .ok()
            .map(|s| s.trim().to_string())
    } else {
        None
    }
}

/// Best-effort current working directory; None if unreadable.
pub fn current_dir() -> Option<std::path::PathBuf> {
    std::env::current_dir().ok()
}

/// True if the path exists on disk (used for artifact presence checks).
pub fn exists(p: &Path) -> bool {
    p.exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_succeeds_best_effort() {
        let env = Environment::capture();
        // Kernel release should exist on Linux.
        assert!(env.os.release.is_some());
        // CPU model should exist.
        assert!(env.cpu.model.is_some());
        // Crate version comes from Cargo (never hardcode the number here).
        assert_eq!(env.crate_version, env!("CARGO_PKG_VERSION"));
        // On x86_64 CI/dev boxes either avx2 or none is fine; the fields must
        // be consistent with the flag list.
        if env.cpu.has_avx2 {
            assert!(env.cpu.flags.iter().any(|f| f == "avx2"));
        }
    }

    #[test]
    fn git_identity_format() {
        let dirty = Environment {
            git_commit: Some("abc123".into()),
            git_dirty: Some(true),
            ..Environment::default()
        };
        assert_eq!(dirty.git_identity(), Some("abc123-dirty".into()));
        let clean = Environment {
            git_commit: Some("abc123".into()),
            git_dirty: Some(false),
            ..Environment::default()
        };
        assert_eq!(clean.git_identity(), Some("abc123".into()));
    }
}
