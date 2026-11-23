#!/usr/bin/env python3
"""GPU-aware FlashAttention installation policy.

The policy selects installation components only. DraftKV's original streaming
attention autotune and frozen kernel implementations are not changed.
"""
import re
import subprocess


def from_compute_capability(compute_capability):
    """SM90 Hopper -> FA3; other NVIDIA GPU architectures -> FA2."""
    match = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", str(compute_capability or ""))
    if match and (int(match.group(1)), int(match.group(2))) == (9, 0):
        return "fa3"
    return "fa2"


def detect(gpu_index=0):
    """Inspect one physical GPU; absence of nvidia-smi falls back to FA2.

    No comparison with the historical H200 profile is made.
    """
    if not isinstance(gpu_index, int) or isinstance(gpu_index, bool) or gpu_index < 0:
        raise ValueError("GPU index must be a nonnegative integer")
    cmd = ["nvidia-smi", "--id=" + str(gpu_index),
           "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=10)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return dict(gpu_index=gpu_index, detected=False, name=None,
                    compute_capability=None, driver_version=None, backend="fa2")
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        return dict(gpu_index=gpu_index, detected=False, name=None,
                    compute_capability=None, driver_version=None, backend="fa2")
    cells = [cell.strip() for cell in rows[0].split(",")]
    if len(cells) != 3 or not re.fullmatch(r"\d+\.\d+", cells[1]):
        return dict(gpu_index=gpu_index, detected=False, name=None,
                    compute_capability=None, driver_version=None, backend="fa2")
    return dict(gpu_index=gpu_index, detected=True, name=cells[0],
                compute_capability=cells[1], driver_version=cells[2],
                backend=from_compute_capability(cells[1]))
