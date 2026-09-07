#!/bin/sh
# Run the full court battery.
#
# Phase A status: courts land phase by phase (semantic court is first, Phase C).
# Until a court exists this script exits NOT_IMPLEMENTED (3); it never pretends
# a court ran.
set -eu

cd "$(dirname "$0")/.."

echo "vole-audio: court-all.sh"
echo "NOT_IMPLEMENTED: no courts exist yet (first court: 'semantic', Phase C)."
exit 3
