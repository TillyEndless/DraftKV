#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${DRAFTKV_INSTALL_PYTHON:-python3}" -I -B "$SCRIPT_DIR/_install.py" verify "$@"
