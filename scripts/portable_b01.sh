#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Portable builds use the original audited B01 source API with a new receipt.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec "${DRAFTKV_INSTALL_PYTHON:-python3}" -B "$ROOT/tools/portable_b01.py" "$@"
