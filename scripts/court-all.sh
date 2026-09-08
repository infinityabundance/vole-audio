#!/bin/sh
# Run the full court battery of the CURRENT build.
#
# Courts arrive by phase; `court-all.sh` runs every always-runnable court this
# build ships (semantic, authored, simd, facts, cuda, d1, rocm, entropy-rans,
# entropy-literal, entropy-residual, entropy-pages, entropy-partial,
# entropy-simd) and reports their verdicts.
# Each court writes its own immutable receipt under receipts/<court>/ and
# exits 0 whether its verdict is SUPPORTED or an honest negative — this
# script fails only on operational errors, never on a negative result.
#
# Evidence binding: the release binary is rebuilt UNCONDITIONALLY before the
# battery (incremental cargo makes this cheap) and the binary's compile-time
# source identity must match the work tree (compiled-from == executed-in-
# worktree, `source_bound`). A stale binary can never bind a receipt to a
# tree it was not built from — receipts record both identities and a seal
# requires them to match.
#
# `court d1` plays to a real ALSA endpoint in silence-safe mode by default
# (opt-in audible content: VOLE_D1_EMIT_AUDIO=1). The CUDA courts need the
# PTX artifact (scripts/build-cuda-device.sh) and a CUDA device; absent
# hardware/artifact they produce UNSUPPORTED_BY_HARDWARE / INCONCLUSIVE
# receipts and still exit 0. `court rocm` needs the amdgcn code object
# (scripts/build-rocm-device.sh) for its compile-surface evidence.
set -eu

cd "$(dirname "$0")/.."

echo "building release binary (all-features)..."
cargo build --release --all-features

# Source-bound check: refuse to run a battery whose binary was built from a
# different tree than the one the receipts would attest.
BIN_VERSION=$(./target/release/vole-audio version)
echo "$BIN_VERSION"
if ! echo "$BIN_VERSION" | grep -q "source_bound: true"; then
    echo "error: release binary is not source-bound to this work tree;" >&2
    echo "rebuild before running a court battery (receipts must attest the" >&2
    echo "tree the binary was actually built from)." >&2
    exit 1
fi

run_court() {
    echo
    echo "== court $1 =="
    ./target/release/vole-audio court "$1" --receipts receipts
}

rc=0
for court in semantic authored simd facts cuda d1 rocm entropy-rans entropy-literal \
    entropy-residual entropy-pages entropy-partial entropy-simd; do
    if ! run_court "$court"; then
        echo "court $court: operational failure" >&2
        rc=1
    fi
done

# Hardware/feature-gated courts (entropy-cuda, entropy-d1, entropyfs,
# dsfb-entropy, h2) are run by the phase seal with --all-features; the
# baseline above is always runnable.

echo
echo "court-all.sh: done (verdicts above; receipts under receipts/<court>/)."
exit "$rc"
