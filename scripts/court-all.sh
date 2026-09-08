#!/bin/sh
# Run the full court battery of the CURRENT build.
#
# Courts arrive by phase; `court-all.sh` runs every court this build ships
# (semantic, authored, simd, facts, cuda, d1, entropy-rans, entropy-literal,
# entropy-residual, entropy-pages, entropy-partial) and reports their verdicts.
# Each court writes its own immutable receipt under receipts/<court>/ and
# exits 0 whether its verdict is SUPPORTED or an honest negative — this
# script fails only on operational errors, never on a negative result.
#
# `court d1` plays to a real ALSA endpoint in silence-safe mode by default
# (opt-in audible content: VOLE_D1_EMIT_AUDIO=1). The CUDA courts need the
# PTX artifact (scripts/build-cuda-device.sh) and a CUDA device; absent
# hardware/artifact they produce UNSUPPORTED_BY_HARDWARE / INCONCLUSIVE
# receipts and still exit 0.
set -eu

cd "$(dirname "$0")/.."

if [ ! -x target/release/vole-audio ]; then
    echo "building release binary..."
    cargo build --release
fi

run_court() {
    echo
    echo "== court $1 =="
    ./target/release/vole-audio court "$1" --receipts receipts
}

rc=0
for court in semantic authored simd facts cuda d1 entropy-rans entropy-literal \
    entropy-residual entropy-pages entropy-partial; do
    if ! run_court "$court"; then
        echo "court $court: operational failure" >&2
        rc=1
    fi
done

echo
echo "court-all.sh: done (verdicts above; receipts under receipts/<court>/)."
exit "$rc"
