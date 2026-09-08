#!/bin/sh
# Build the CUDA (nvptx64-nvidia-cuda) device artifact from this same package.
#
# Phase G: emits a PTX module compiled directly from src/lib.rs (the one
# package — no second crate), with `--no-default-features` semantics (the
# crate's std-gated modules are cfg'd out; device code is no_std). The PTX is
# loaded by the CUDA driver API at runtime (backend::cuda); module-load JIT
# happens once at setup, never on a real-time path.
#
# Artifacts (scripts/out/, gitignored):
#   vole_audio.ptx         PTX module (entry: vole_render_d0)
#   vole_audio.ptx.sha256  artifact SHA-256
#   vole_audio.ptx.json    provenance metadata (rustc/LLVM, target, source hash)
#
# Determinism note: the build records the committed source-tree hash and the
# rustc identity; the artifact bytes are reproducible from the recorded
# inputs (same tree + same pinned toolchain).
set -eu

cd "$(dirname "$0")/.."

OUT=scripts/out
mkdir -p "$OUT"

RUSTC=${RUSTC:-rustc}
echo "vole-audio: build-cuda-device.sh"
echo "  rustc: $("$RUSTC" --version)"
echo "  target: nvptx64-nvidia-cuda (PTX, sm_70 baseline; driver JITs for the installed GPU)"

# Source-tree anchor (excludes generated receipts so the attested tree stays
# clean when receipts were just written).
TREE=$(git rev-parse HEAD^{tree} 2>/dev/null || echo "not-a-git-tree")
DIRTY=$(git status --porcelain --untracked-files=all -- . ':(exclude)receipts' 2>/dev/null | wc -l | tr -d ' ')

# Compile the shared semantic core + device entry to PTX. The crate is one
# package; no build-std needed (the toolchain ships rust-std for this Tier-2
# target). `--crate-type cdylib` on nvptx64-nvidia-cuda emits PTX text.
"$RUSTC" \
    --edition 2024 \
    --crate-type cdylib \
    --target nvptx64-nvidia-cuda \
    --crate-name vole_audio \
    -C panic=abort \
    -C opt-level=3 \
    -C codegen-units=1 \
    -C debuginfo=0 \
    src/lib.rs \
    -o "$OUT/vole_audio.ptx"

# Artifact hash + provenance sidecar.
SHA=$(sha256sum "$OUT/vole_audio.ptx" | cut -d' ' -f1)
RUSTC_VER=$("$RUSTC" --version | sed 's/ (.*//')
LLVM_VER=$("$RUSTC" -vV | grep -oE 'LLVM version: [0-9.]+' | sed 's/LLVM version: //')
cat > "$OUT/vole_audio.ptx.json" <<EOF
{
  "artifact": "vole_audio.ptx",
  "sha256": "$SHA",
  "entry": "vole_render_d0",
  "rustc": "$RUSTC_VER",
  "llvm": "$LLVM_VER",
  "target": "nvptx64-nvidia-cuda",
  "target_cpu": "sm_70 (baseline; driver JIT for installed GPU)",
  "profile": "release-equivalent: panic=abort opt-level=3 codegen-units=1",
  "source_tree_sha": "$TREE",
  "source_dirty": $([ "$DIRTY" = 0 ] && echo false || echo true),
  "created_unix_ms": $(date +%s%3N)
}
EOF
echo "$SHA  $OUT/vole_audio.ptx" > "$OUT/vole_audio.ptx.sha256"

echo "  artifact: $OUT/vole_audio.ptx ($(wc -c < "$OUT/vole_audio.ptx") bytes, sha256 $SHA)"
echo "  provenance: $OUT/vole_audio.ptx.json"
