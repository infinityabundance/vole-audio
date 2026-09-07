#!/bin/sh
# Build the CUDA (nvptx64-nvidia-cuda) device artifact from this same package.
#
# Phase A status: the device kernel surface lands in Phase G. Until then this
# script exits NOT_IMPLEMENTED (3) instead of producing a misleading artifact.
set -eu

cd "$(dirname "$0")/.."

echo "vole-audio: build-cuda-device.sh"
echo "NOT_IMPLEMENTED: CUDA device kernel surface arrives in Phase G."
echo "This script intentionally refuses to emit a PTX artifact before the"
echo "shared semantic core exists and is differential-tested."
exit 3
