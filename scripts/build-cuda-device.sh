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

# Strip DWARF debug sections from the PTX text: rustc's NVPTX backend emits
# .debug_abbrev/.debug_info/.debug_macinfo blocks even at -C debuginfo=0, and
# the driver's ptxas JIT intermittently fails parsing them on the tested
# driver (rc 218 "Unexpected non-ASCII character", observed inside
# .debug_info) — a process-state-dependent flake that disappears when the
# debug metadata is absent. The metadata is never needed for execution.
python3 - "$OUT/vole_audio.ptx" <<'EOF'
import sys
path = sys.argv[1]
lines = open(path, encoding='utf-8').read().split('\n')
out = []
i = 0
n = len(lines)
while i < n:
    line = lines[i]
    stripped = line.strip()
    if stripped.startswith('.target '):
        # rustc's NVPTX backend marks the target `debug` when any inlined
        # precompiled core/std code carries DWARF (even at -C debuginfo=0);
        # with the debug sections stripped below, ptxas rejects a `debug`
        # target that has no debug data, so request a plain target.
        line = line.replace(', debug', '')
        out.append(line)
        i += 1
        continue
    if stripped.startswith('.section\t.debug_') or stripped.startswith('.section .debug_'):
        # Skip the section header; if it does not open a brace inline, skip
        # until the matching close brace at line start.
        if '{' not in line:
            i += 1
            while i < n and lines[i].strip() != '}':
                i += 1
        i += 1
        continue
    out.append(line)
    i += 1
open(path, 'w', encoding='utf-8').write('\n'.join(out))
EOF
echo "  stripped DWARF debug sections from PTX"

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
