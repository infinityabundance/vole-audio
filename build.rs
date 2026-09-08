//! Compile-time source identity stamping (evidence binding).
//!
//! Receipts record the *runtime* work-tree identity (`Environment::capture`,
//! executed-in-worktree). This build script additionally stamps the
//! *compile-time* identity of the host binary itself (compiled-from), so a
//! court run can never bind a receipt to a source tree the binary was not
//! built from:
//!
//!     source tree  ->  host binary  ->  GPU artifact  ->  observation  ->  receipt
//!        (build-time git identity)      (build.rs)      (sidecar)        (cryptographic)
//!
//! The seal procedure requires compiled-from == executed-in-worktree.
//!
//! The build script reruns whenever any source file changes or the git
//! metadata moves (HEAD / branch ref / packed-refs are tracked explicitly,
//! resolved through `git rev-parse --git-path` for linked worktrees), so the
//! stamped identity always matches the tree the binary was actually built
//! from — including history-only HEAD moves that change no file. Outside a
//! git work tree (e.g. the crates.io tarball) the fields are empty and
//! receipts simply carry `None` — the evidence reports
//! `source_binding = UNAVAILABLE`, and a seal requires `BOUND`.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    // Rerun when sources change…
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");
    // …and when the git identity moves. File-change tracking alone misses
    // history-only HEAD moves (commit/checkout changes no file), which
    // would leave the compiled-from stamp stale. Track the git metadata
    // itself: .git/HEAD, the resolved branch ref, and packed-refs. Paths
    // are resolved through `git rev-parse --git-path` so linked worktrees
    // work too. (Without rerun-if-* cargo may not rerun on every build, so
    // this explicit tracking is what keeps the stamp honest.)
    let git_path = |arg: &str| {
        Command::new("git")
            .args(["rev-parse", "--git-path", arg])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .filter(|s| !s.is_empty())
    };
    for tracked in ["HEAD", "packed-refs"] {
        if let Some(p) = git_path(tracked) {
            println!("cargo:rerun-if-changed={p}");
        }
    }
    let branch = git(&["rev-parse", "--abbrev-ref", "HEAD"]);
    if let Some(b) = branch
        .filter(|b| b != "HEAD")
        .and_then(|b| git_path(&format!("refs/heads/{b}")))
    {
        println!("cargo:rerun-if-changed={b}");
    }

    // Compiled-from identity. `--no-optional-locks` and the receipts/
    // pathspec exclusion mirror Environment::capture's runtime measurement:
    // writing evidence never marks the attested tree dirty by itself.
    let commit = git(&["rev-parse", "HEAD"]);
    let tree = git(&["rev-parse", "HEAD^{tree}"]);
    let dirty = git(&[
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude)receipts",
    ])
    .map(|s| !s.is_empty());

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let rustc_version = Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());
    let profile = std::env::var("PROFILE").ok();
    let target = std::env::var("TARGET").ok();

    for (key, value) in [
        ("VOLE_BUILD_GIT_COMMIT", commit),
        ("VOLE_BUILD_TREE_SHA", tree),
        ("VOLE_BUILD_RUSTC", rustc_version),
        ("VOLE_BUILD_PROFILE", profile),
        ("VOLE_BUILD_TARGET", target),
    ] {
        if let Some(v) = value {
            println!("cargo:rustc-env={key}={v}");
        }
    }
    println!(
        "cargo:rustc-env=VOLE_BUILD_GIT_DIRTY={}",
        dirty.unwrap_or(false)
    );
}
