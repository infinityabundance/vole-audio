#!/bin/sh
# Build the ROCm (amdgcn-amd-amdhsa) device artifact from this same package.
#
# Phase A status: the device kernel surface lands in Phase I. Until then this
# script exits NOT_IMPLEMENTED (3) instead of producing a misleading artifact.
set -eu

cd "$(dirname "$0")/.."

echo "vole-audio: build-rocm-device.sh"
echo "NOT_IMPLEMENTED: ROCm device kernel surface arrives in Phase I."
echo "This script intentionally refuses to emit a code object before the"
echo "shared semantic core exists and is differential-tested."
exit 3
