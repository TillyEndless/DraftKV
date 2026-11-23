#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Select and bind a Python deployment environment without silent system installs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import urlparse


MODES = ("auto", "conda", "venv")


def resolve_mode(value="auto"):
    mode = value or "auto"
    if mode not in MODES:
        raise ValueError("unknown Python environment mode: " + str(mode))
    return ("conda" if os.environ.get("CONDA_PREFIX") and
            os.environ.get("CONDA_DEFAULT_ENV") != "base" else "venv") if mode == "auto" else mode


def conda_python(requested=None):
    """Require an activated non-base Conda CPython 3.10, never system Python."""
    supplied = os.environ.get("CONDA_PREFIX")
    if not supplied:
        raise ValueError("Conda mode requires an activated environment (conda activate draftkv)")
    prefix = Path(supplied).expanduser().resolve()
    if os.environ.get("CONDA_DEFAULT_ENV") == "base":
        raise ValueError("refusing to install into Conda base; create a dedicated environment")
    if not (prefix / "conda-meta").is_dir():
        raise ValueError("CONDA_PREFIX does not contain conda-meta: " + str(prefix))
    executable = prefix / "bin/python"
    if not executable.is_file():
        raise ValueError("Conda Python not found: " + str(executable))
    if requested:
        found = shutil.which(str(requested))
        if not found or Path(found).resolve() != executable.resolve():
            raise ValueError("--python must resolve to the active Conda environment's bin/python")
    script = ("import json,sys;print(json.dumps({'version':list(sys.version_info[:2]),"
              "'prefix':sys.prefix,'executable':sys.executable}))")
    result = subprocess.check_output([str(executable), "-I", "-B", "-c", script], text=True)
    identity = json.loads(result.strip())
    if identity["version"] != [3, 10] or Path(identity["prefix"]).resolve() != prefix:
        raise ValueError("Conda mode requires an active CPython 3.10 whose sys.prefix equals CONDA_PREFIX")
    return executable


def pip_isolated_flags(mode):
    return ["--require-virtualenv"] if mode == "venv" else []


def binding(base, mode, python, backbone, *, claim=False):
    """Bind a per-Backbone cache to the selected Python; forbid silent mode switches.

    Claiming an active Conda environment prevents mutually incompatible Backbones
    overwriting one another. Read-only checks do not create directories or files.
    """
    base = Path(base)
    python = Path(python).absolute()
    mode = resolve_mode(mode)
    payload = dict(schema=1, mode=mode, python=str(python), backbone=backbone)
    record = base / "runtime-environment.json"
    if record.exists() or record.is_symlink():
        if not record.is_file() or record.is_symlink() or json.loads(record.read_text()) != payload:
            raise ValueError("deployment cache is bound to another Python/environment; change --env-root")
    elif mode == "conda" and (base / "portable-dependency-closure.json").is_file():
        # Older receipts only ever originated from a Backbone venv.
        raise ValueError("legacy venv deployment cache found; select a different --env-root for Conda")
    if mode == "conda":
        prefix = python.parent.parent
        marker = prefix / ".draftkv-backbone.json"
        owner = dict(schema=1, backbone=backbone)
        if marker.exists() or marker.is_symlink():
            if not marker.is_file() or marker.is_symlink() or json.loads(marker.read_text()) != owner:
                raise ValueError("Conda environment already belongs to a different DraftKV Backbone")
        if claim and not marker.exists():
            with marker.open("x", encoding="utf-8") as file:
                json.dump(owner, file, sort_keys=True)
                file.write("\n")
    if claim and not record.exists():
        base.mkdir(parents=True, exist_ok=True)
        with record.open("x", encoding="utf-8") as file:
            json.dump(payload, file, sort_keys=True)
            file.write("\n")
    return payload


def torch_index_url(requested, cuda):
    """Opt-in trusted mirror with the same CUDA-specific index and exact pins."""
    channel = "cu" + str(cuda).replace(".", "")
    official = "https://download.pytorch.org/whl/" + channel
    aliyun = "https://mirrors.aliyun.com/pytorch-wheels/" + channel
    value = (requested or official).rstrip("/")
    parts = urlparse(value)
    if (parts.scheme != "https" or parts.username or parts.password or
            parts.query or parts.fragment or value not in (official, aliyun)):
        raise ValueError("PyTorch index must be the official or Aliyun HTTPS "
                         "repository for " + channel)
    return value
