#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec "${DRAFTKV_CONTROL_PYTHON:-python3}" -B "$ROOT/scripts/draftkv.py" "$@"
