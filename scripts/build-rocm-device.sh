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
# Artifacts (scripts/out/, gitignored):
#   vole_audio.amdgcn.elf     AMDGPU code object (entry: vole_render_d0,
#                             vole_entropy_decode, vole_upmix_mono_dup)
#   vole_audio.amdgcn.sha256  artifact SHA-256
#   vole_audio.amdgcn.json    provenance metadata (rustc/LLVM, target, gfx,
#                             source hash)
set -eu

cd "$(dirname "$0")/.."

OUT=scripts/out
mkdir -p "$OUT"

RUSTC=${RUSTC:-rustc}
GFX=${VOLE_ROCM_GFX:-gfx906}
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

# Compile the shared semantic core + device entry as an AMDGPU code object.
#
# Two steps (cargo's --config cannot override the manifest's lib crate-type):
#   1. cargo -Z build-std=core --release --lib builds the rlib (core built
#      from source; CARGO_INCREMENTAL=0 is mandatory — see the header note).
#   2. rustc links that same src/lib.rs as a cdylib directly against the
#      build-std rlibs (the sysroot has no prebuilt amdgcn std, so core and
#      compiler_builtins are passed explicitly), exactly like the CUDA
#      script's direct-rustc style.
CARGO_INCREMENTAL=0 RUSTFLAGS="-C target-cpu=$GFX" \
    cargo build -Z build-std=core --release --target amdgcn-amd-amdhsa \
    --no-default-features --lib >/dev/null

CORE=$(ls target/amdgcn-amd-amdhsa/release/deps/libcore-*.rlib | head -1)
CB=$(ls target/amdgcn-amd-amdhsa/release/deps/libcompiler_builtins-*.rlib | head -1)
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
    -L dependency=target/amdgcn-amd-amdhsa/release/deps \
    src/lib.rs \
    -o "$OUT/vole_audio.amdgcn.elf"

# Artifact hash + provenance sidecar.
SHA=$(sha256sum "$OUT/vole_audio.amdgcn.elf" | cut -d' ' -f1)
RUSTC_VER=$("$RUSTC" --version | sed 's/ (.*//')
LLVM_VER=$("$RUSTC" -vV | grep -oE 'LLVM version: [0-9.]+' | sed 's/LLVM version: //')
cat > "$OUT/vole_audio.amdgcn.json" <<EOF
{
  "artifact": "vole_audio.amdgcn.elf",
  "sha256": "$SHA",
  "entries": [
    "vole_render_d0",
    "vole_entropy_decode",
    "vole_upmix_mono_dup"
  ],
  "rustc": "$RUSTC_VER",
  "llvm": "$LLVM_VER",
  "target": "amdgcn-amd-amdhsa",
  "target_cpu": "$GFX (code objects are per-ISA; built via VOLE_ROCM_GFX)",
  "std": "build-std=core (rust-src; rustup ships no prebuilt amdgcn std)",
  "profile": "release; CARGO_INCREMENTAL=0 (fat-LTO ICE avoidance, documented)",
  "source_tree_sha": "$TREE",
  "source_dirty": $([ "$DIRTY" = 0 ] && echo false || echo true),
  "loadability": "unvalidated: requires ROCm runtime + matching gfx hardware (Phase J); this artifact is compile evidence only",
  "created_unix_ms": $(date +%s%3N)
}
EOF
echo "$SHA  $OUT/vole_audio.amdgcn.elf" > "$OUT/vole_audio.amdgcn.sha256"

echo "  artifact: $OUT/vole_audio.amdgcn.elf ($(wc -c < "$OUT/vole_audio.amdgcn.elf") bytes, sha256 $SHA)"
echo "  provenance: $OUT/vole_audio.amdgcn.json"
