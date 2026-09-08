#!/bin/sh
# Build the ROCm (amdgcn-amd-amdhsa) device code object from this same package.
#
# Phase I: emits an AMDGPU code object (ELF shared object) compiled directly
# from src/lib.rs — the one package, no second crate — with `--no-default-
# features` semantics (std-gated modules cfg'd out; device code is no_std).
# The artifact is loaded by the ROCm runtime at runtime (backend::rocm /
# Phase J on ROCm hardware).
#
# AMDGCN toolchain reality (documented divergence from the CUDA path):
#
#   * rustup ships NO prebuilt rust-std for amdgcn-amd-amdhsa on this pinned
#     nightly, so core is built from source via `cargo -Z build-std=core`
#     (requires the rust-src component of the same toolchain).
#   * The amdgcn cdylib link ICEs in fat-LTO ("cannot find embedded bitcode",
#     rustc_codegen_llvm back/write.rs) whenever the build-std rlibs carry no
#     embedded bitcode — which incremental mode forces. CARGO_INCREMENTAL=0 is
#     therefore required (not a workaround for a flake; it is the documented
#     stable invocation).
#   * AMDGPU code objects are per-ISA (`-C target-cpu=gfx…`), unlike the
#     portable PTX text the CUDA path emits. The build uses the gfx named by
#     VOLE_ROCM_GFX (default gfx906) purely as compile evidence; loadability
#     on real ROCm hardware is Phase-J evidence and is never assumed.
#
# Build isolation (stale-artifact safety): every build runs in a fresh
# per-toolchain/per-gfx target directory
#   target/vole-rocm/<rustc-commit-hash>/<gfx>[/verify]
# which is removed before the build, so no stale core/compiler_builtins rlib
# from another toolchain, flag set, or gfx can ever be linked, and the
# core/compiler_builtins glob is guaranteed to resolve to exactly one rlib
# (enforced, never `head -1` of a pile).
#
# Determinism: `--verify-deterministic` builds twice in two isolated
# directories and requires SHA256(A1) == SHA256(A2), writing
# scripts/out/vole_audio.amdgcn.determinism.json — byte-determinism becomes
# measured evidence instead of prose.
#
# Artifacts (scripts/out/, gitignored):
#   vole_audio.amdgcn.elf     AMDGPU code object (entry: vole_render_d0,
#                             vole_entropy_decode, vole_upmix_mono_dup)
#   vole_audio.amdgcn.sha256  artifact SHA-256
#   vole_audio.amdgcn.elf.json  provenance metadata (rustc/LLVM, target,
#                             gfx, source hash, entries; legacy
#                             vole_audio.amdgcn.json copy for migration)
set -eu

cd "$(dirname "$0")/.."

OUT=scripts/out
mkdir -p "$OUT"

RUSTC=${RUSTC:-rustc}
GFX=${VOLE_ROCM_GFX:-gfx906}
VERIFY=0
[ "${1:-}" = "--verify-deterministic" ] && VERIFY=1

echo "vole-audio: build-rocm-device.sh"
echo "  rustc: $("$RUSTC" --version)"
echo "  target: amdgcn-amd-amdhsa (code object for gfx=${GFX}; VOLE_ROCM_GFX overrides)"

# Preflight: rust-src (required for -Z build-std=core) present?
SYSROOT=$("$RUSTC" --print sysroot)
if [ ! -d "$SYSROOT/lib/rustlib/src/rust/library/core" ]; then
    echo "  error: rust-src component not installed for $(rustc --version | cut -d' ' -f2)."
    echo "  install: rustup component add rust-src --toolchain <pinned nightly>"
    exit 3
fi

# Source-tree anchor (excludes generated receipts so the attested tree stays
# clean when receipts were just written).
TREE=$(git rev-parse HEAD^{tree} 2>/dev/null || echo "not-a-git-tree")
DIRTY=$(git status --porcelain --untracked-files=all -- . ':(exclude)receipts' 2>/dev/null | wc -l | tr -d ' ')

RUSTC_COMMIT=$("$RUSTC" -vV 2>/dev/null | grep 'commit-hash:' | cut -d' ' -f2 || echo "unknown")

# Build one code object into an isolated fresh target directory.
build_one() { # $1 = target dir suffix ("" or "/verify")
    local suffix=$1
    local tdir="target/vole-rocm/$RUSTC_COMMIT/$GFX$suffix"
    rm -rf "$tdir"
    CARGO_INCREMENTAL=0 RUSTFLAGS="-C target-cpu=$GFX" \
        cargo build -Z build-std=core --release --target amdgcn-amd-amdhsa \
        --no-default-features --lib --target-dir "$tdir" >/dev/null
    local deps="$tdir/amdgcn-amd-amdhsa/release/deps"
    # Exactly one core and one compiler_builtins rlib must exist (isolated
    # fresh dir; enforced, not assumed).
    local ncore ncb
    ncore=$(ls "$deps"/libcore-*.rlib 2>/dev/null | wc -l | tr -d ' ')
    ncb=$(ls "$deps"/libcompiler_builtins-*.rlib 2>/dev/null | wc -l | tr -d ' ')
    if [ "$ncore" -ne 1 ] || [ "$ncb" -ne 1 ]; then
        echo "  error: expected exactly one core/compiler_builtins rlib (got $ncore/$ncb) in $deps" >&2
        exit 1
    fi
    local CORE CB
    CORE=$(ls "$deps"/libcore-*.rlib)
    CB=$(ls "$deps"/libcompiler_builtins-*.rlib)
    "$RUSTC" \
        --edition 2024 \
        --crate-type cdylib \
        --target amdgcn-amd-amdhsa \
        --crate-name vole_audio \
        -C target-cpu="$GFX" \
        -C opt-level=3 \
        -C codegen-units=1 \
        -C debuginfo=0 \
        -C lto=off \
        --extern "core=$CORE" \
        --extern "compiler_builtins=$CB" \
        -L "dependency=$deps" \
        src/lib.rs \
        -o "$tdir/vole_audio.amdgcn.elf"
    sha256sum "$tdir/vole_audio.amdgcn.elf" | cut -d' ' -f1
}

SHA1=$(build_one "")
cp "target/vole-rocm/$RUSTC_COMMIT/$GFX/vole_audio.amdgcn.elf" "$OUT/vole_audio.amdgcn.elf"

# Determinism evidence: a second isolated build must produce identical bytes.
if [ "$VERIFY" = 1 ]; then
    echo "  verifying byte-determinism with a second isolated build..."
    SHA2=$(build_one "/verify")
    if [ "$SHA1" = "$SHA2" ]; then
        EQUAL=true
        echo "  deterministic: sha256($SHA1) == sha256($SHA2)"
    else
        EQUAL=false
        echo "  error: builds differ ($SHA1 != $SHA2) — artifact is NOT byte-deterministic" >&2
    fi
    cat > "$OUT/vole_audio.amdgcn.determinism.json" <<EOF
{
  "artifact": "vole_audio.amdgcn.elf",
  "build_a_sha256": "$SHA1",
  "build_b_sha256": "$SHA2",
  "byte_deterministic": $EQUAL,
  "method": "two isolated builds (target/vole-rocm/<rustc-commit>/<gfx> and .../verify), fresh directories",
  "source_tree_sha": "$TREE",
  "created_unix_ms": $(date +%s%3N)
}
EOF
    [ "$EQUAL" = true ] || exit 1
fi

# Provenance sidecar. Canonical name (frozen): <artifact filename>.json
# = vole_audio.amdgcn.elf.json; a legacy <stem>.json copy is written during
# migration for older consumers.
RUSTC_VER=$("$RUSTC" --version | sed 's/ (.*//')
LLVM_VER=$("$RUSTC" -vV | grep -oE 'LLVM version: [0-9.]+' | sed 's/LLVM version: //')
SHA=$SHA1
SIDECAR=$OUT/vole_audio.amdgcn.elf.json
cat > "$SIDECAR" <<EOF
{
  "artifact": "vole_audio.amdgcn.elf",
  "sha256": "$SHA",
  "entries": [
    "vole_render_d0",
    "vole_entropy_decode",
    "vole_upmix_mono_dup"
  ],
  "rustc": "$RUSTC_VER",
  "rustc_commit_hash": "$RUSTC_COMMIT",
  "llvm": "$LLVM_VER",
  "target": "amdgcn-amd-amdhsa",
  "target_cpu": "$GFX (code objects are per-ISA; built via VOLE_ROCM_GFX)",
  "std": "build-std=core (rust-src; rustup ships no prebuilt amdgcn std)",
  "profile": "release; CARGO_INCREMENTAL=0 (fat-LTO ICE avoidance, documented)",
  "build_isolation": "fresh per-build target dir target/vole-rocm/<rustc-commit>/<gfx>; exactly-one core/compiler_builtins enforced",
  "determinism": "$([ "$VERIFY" = 1 ] && echo verified-two-isolated-builds || echo not-verified-this-run)",
  "source_tree_sha": "$TREE",
  "source_dirty": $([ "$DIRTY" = 0 ] && echo false || echo true),
  "loadability": "unvalidated: requires ROCm runtime + matching gfx hardware (Phase J); this artifact is compile evidence only",
  "created_unix_ms": $(date +%s%3N)
}
EOF
# Legacy migration copy (older consumers looked for <stem>.json).
cp "$SIDECAR" "$OUT/vole_audio.amdgcn.json"
echo "$SHA  $OUT/vole_audio.amdgcn.elf" > "$OUT/vole_audio.amdgcn.sha256"

echo "  artifact: $OUT/vole_audio.amdgcn.elf ($(wc -c < "$OUT/vole_audio.amdgcn.elf") bytes, sha256 $SHA)"
echo "  provenance: $SIDECAR (+ legacy vole_audio.amdgcn.json copy)"
