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
    /// ---- Compile-time identity (stamped by build.rs; compiled-from) ----
    /// Commit the host binary was built from. Together with the runtime
    /// fields above (executed-in-worktree), this binds receipts to the
    /// source tree: a seal requires compiled-from == executed-in-worktree.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_git_commit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_tree_sha: Option<String>,
    /// Build-time dirty state (receipts/ excluded, as in the runtime field).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_dirty: Option<bool>,
    /// rustc that compiled this binary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_rustc: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_target: Option<String>,
    /// Computed source-binding state (bound / unavailable / mismatch /
    /// dirty). A seal requires `bound`.
    #[serde(default)]
    pub source_binding: SourceBinding,
    /// SHA-256 of the filtered **seal subject** (see `crate::evidence::subject`):
    /// every tracked source file except the evidence/governance trees
    /// (`receipts/`, `target/`, `scripts/out/`, `docs/`, `.git/`). This is the
    /// identity a seal compares: it survives committing receipts/docs (which
    /// cannot affect execution) and changes only when code changes. `None`
    /// outside a git work tree (unbounded evidence).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seal_subject_hash: Option<String>,
}

/// Compile-time stamped constants (set by build.rs).
pub const BUILD_GIT_COMMIT: Option<&str> = option_env!("VOLE_BUILD_GIT_COMMIT");
pub const BUILD_TREE_SHA: Option<&str> = option_env!("VOLE_BUILD_TREE_SHA");
/// "true"/"false" as stamped by build.rs (parsed at capture time — const
/// str matching is not stable).
pub const BUILD_GIT_DIRTY: Option<&str> = option_env!("VOLE_BUILD_GIT_DIRTY");
pub const BUILD_RUSTC: Option<&str> = option_env!("VOLE_BUILD_RUSTC");
pub const BUILD_PROFILE: Option<&str> = option_env!("VOLE_BUILD_PROFILE");
pub const BUILD_TARGET: Option<&str> = option_env!("VOLE_BUILD_TARGET");

/// Source-binding state of a receipt: how the executing binary relates to
/// the tree the receipt attests. One meaning everywhere: a **seal** requires
/// `Bound`; a crates.io tarball run reports `Unavailable` (it can still run
/// courts and emit receipts — it just says the evidence is unbounded).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceBinding {
    /// Compiled-from == executed-in-worktree, and both dirty states are
    /// explicitly false.
    Bound,
    /// No git identity on either side (e.g. the crates.io tarball): the
    /// evidence is genuinely unbounded.
    #[default]
    Unavailable,
    /// Only one side has a git identity, or both commits are present/equal
    /// but a dirty state is unknown (could not be measured): the identity
    /// picture is incomplete and cannot be bound.
    Partial,
    /// Compiled-from commit != executed-in-worktree commit.
    Mismatch,
    /// Compiled-from and/or work tree is explicitly dirty.
    Dirty,
}

impl SourceBinding {
    pub fn label(self) -> &'static str {
        match self {
            SourceBinding::Bound => "bound",
            SourceBinding::Unavailable => "unavailable",
            SourceBinding::Partial => "partial",
            SourceBinding::Mismatch => "mismatch",
            SourceBinding::Dirty => "dirty",
        }
    }
}

impl Environment {
    /// Compute the source-binding state from the recorded identities.
    /// Bound requires both identities present and equal with both dirty
    /// states **explicitly false** (an unknown dirty state is never bound).
    pub fn source_binding(&self) -> SourceBinding {
        match (&self.build_git_commit, &self.git_commit) {
            (Some(b), Some(r)) if b == r => {
                let build_clean = self.build_dirty == Some(false);
                let run_clean = self.git_dirty == Some(false);
                if build_clean && run_clean {
                    SourceBinding::Bound
                } else if self.build_dirty == Some(true) || self.git_dirty == Some(true) {
                    SourceBinding::Dirty
                } else {
                    // Commits equal but a dirty state could not be measured.
                    SourceBinding::Partial
                }
            }
            (Some(_), Some(_)) => SourceBinding::Mismatch,
            // Exactly one side has an identity: incomplete picture.
            (Some(_), None) | (None, Some(_)) => SourceBinding::Partial,
            (None, None) => SourceBinding::Unavailable,
        }
    }

    /// True only when the evidence is genuinely bound: compiled-from ==
    /// executed-in-worktree, both explicitly clean. (Non-git builds are not
    /// "bound".)
    pub fn source_bound(&self) -> bool {
        self.source_binding() == SourceBinding::Bound
    }
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
        let mut env = Self {
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
            // Compile-time (compiled-from) identity stamped by build.rs.
            build_git_commit: BUILD_GIT_COMMIT.map(str::to_string),
            build_tree_sha: BUILD_TREE_SHA.map(str::to_string),
            build_dirty: BUILD_GIT_DIRTY.map(|s| s == "true"),
            build_rustc: BUILD_RUSTC.map(str::to_string),
            build_profile: BUILD_PROFILE.map(str::to_string),
            build_target: BUILD_TARGET.map(str::to_string),
            source_binding: SourceBinding::Unavailable,
            // Filtered source-tree identity (best effort): absent outside a
            // git work tree or when the subject cannot be computed. A seal
            // requires it present on both the receipts and the verifier.
            seal_subject_hash: crate::evidence::subject::repo_root()
                .as_deref()
                .and_then(|root| {
                    crate::evidence::subject::seal_subject_hash(root)
                        .ok()
                        .flatten()
                }),
        };
        env.source_binding = env.source_binding();
        env
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

    #[test]
    fn seal_subject_hash_is_present_in_a_git_worktree() {
        let env = Environment::capture();
        if env.git_commit.is_some() {
            // In a git work tree the subject exists and is a SHA-256 hex
            // digest; outside one (crates.io tarball) it is absent — the
            // evidence is unbounded, never fabricated.
            let s = env
                .seal_subject_hash
                .as_deref()
                .expect("subject in a git work tree");
            assert_eq!(s.len(), 64);
            assert!(s.bytes().all(|b| b.is_ascii_hexdigit()));
        } else {
            assert!(env.seal_subject_hash.is_none());
        }
    }

    #[test]
    fn source_binding_states_are_distinct() {
        use SourceBinding::*;
        let bound = Environment {
            git_commit: Some("abc123".into()),
            git_dirty: Some(false),
            build_git_commit: Some("abc123".into()),
            build_dirty: Some(false),
            ..Environment::default()
        };
        assert_eq!(bound.source_binding(), Bound);
        assert!(bound.source_bound());
        // Compiled from an older commit while running at a newer one: the
        // exact stale-binary case a seal must reject.
        let stale = Environment {
            git_commit: Some("def456".into()),
            git_dirty: Some(false),
            build_git_commit: Some("abc123".into()),
            build_dirty: Some(false),
            ..Environment::default()
        };
        assert_eq!(stale.source_binding(), Mismatch);
        assert!(!stale.source_bound());
        // Either side explicitly dirty.
        let dirty_tree = Environment {
            git_commit: Some("abc123".into()),
            git_dirty: Some(true),
            build_git_commit: Some("abc123".into()),
            build_dirty: Some(false),
            ..Environment::default()
        };
        assert_eq!(dirty_tree.source_binding(), Dirty);
        assert!(!dirty_tree.source_bound());
        // Equal commits but an unknown (unmeasured) build dirty state: never
        // bound — Partial, not Dirty, not Bound.
        let unknown_dirty = Environment {
            git_commit: Some("abc123".into()),
            git_dirty: Some(false),
            build_git_commit: Some("abc123".into()),
            build_dirty: None,
            ..Environment::default()
        };
        assert_eq!(unknown_dirty.source_binding(), Partial);
        assert!(!unknown_dirty.source_bound());
        // Non-git tarball build (both absent): genuinely unbounded.
        let tarball = Environment::default();
        assert_eq!(tarball.source_binding(), Unavailable);
        assert!(!tarball.source_bound());
        // One side only: partial identity, never bound.
        let one_side = Environment {
            git_commit: Some("abc123".into()),
            build_git_commit: None,
            ..Environment::default()
        };
        assert_eq!(one_side.source_binding(), Partial);
        assert!(!one_side.source_bound());
    }
}
