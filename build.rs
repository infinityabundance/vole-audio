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
//! The build script reruns whenever any source file changes (cheap), so the
//! stamped identity always matches the tree the binary was actually built
//! from. Outside a git work tree (e.g. the crates.io tarball) the fields are
//! empty and receipts simply carry `None` — the evidence is "not a git
//! work tree", which the runtime capture already reports.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    // Re-stamp whenever sources change (build identity must track the tree).
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");

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
