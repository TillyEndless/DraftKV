#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Portable run uses the same original worker checks with a separately approved B01 build identity.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec "${DRAFTKV_INSTALL_PYTHON:-python3}" -B "$ROOT/tools/launch_portable.py" "$@"
