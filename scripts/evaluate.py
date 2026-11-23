#!/usr/bin/env python3
"""Evaluate model outputs using pinned ORBIA and WBench model/checkpoint stacks.

Table 2: run ORBIA's official evaluator unchanged, then write a *separate*
480-case derived T6/Overall sidecar, with explicit non-leaderboard provenance.
Table 3: run WBench's unchanged DA3 reprojection with DraftKV's frozen masked
SSIM/LPIPS observer, requiring a precomputed DA3 cache for the video.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = ROOT / "versions/lingbot13/src"
WBENCH_REV = "edc4e80e642baac4f2b5764dec9f1a491200af8f"
ORBIA_REV = "f7d8ddde3069a6d5ce75a87fa6b4fc896cc78a59"


def run(args, *, cwd=ROOT, env=None):
    print("+ " + " ".join(__import__("shlex").quote(str(x)) for x in args), flush=True)
    subprocess.run([str(a) for a in args], cwd=cwd, env=env, check=True)


def verified_checkout(directory, revision):
    if not (directory / ".git").is_dir():
        raise ValueError(f"Official evaluator source not installed: {directory}")
    sha = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                  text=True).strip()
    if sha != revision:
        raise ValueError(f"Evaluator source does not match pinned revision: {sha}")
    tracked = subprocess.run(
        ["git", "-C", str(directory), "diff", "--no-ext-diff", "--exit-code", "HEAD", "--"],
        capture_output=True, text=True)
    if tracked.returncode != 0:
        raise ValueError(f"Evaluator source has changes relative to its pinned commit: {directory}")


def evaluate_orbia(args):
    root = Path(args.asset_root).expanduser().resolve()
    official = root / "evaluation/Orbia"
    verified_checkout(official, ORBIA_REV)
    # Verify the exact audited ORBIA files, not only its Git HEAD.
    lock_path = ROOT / "scripts/benchmark_locks/orbia480.lock.json"
    lock = json.loads(lock_path.read_text())
    for relative, expected in lock["source_files"].items():
        file = official / relative
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"ORBIA source SHA mismatch: {relative}")
    dataset = root / "datasets/orbia"
    videos = Path(args.videos).expanduser().resolve()
    if not videos.exists():
        raise ValueError(f"Video collection missing: {videos}")
    if not (dataset / "manifest.json").is_file():
        raise ValueError("Official ORBIA dataset was not prepared")
    config = official / "configs/evaluation.local.yaml"
    if not config.is_file():
        raise ValueError("ORBIA model paths missing: run evaluator setup first")
    output = Path(args.output).expanduser().resolve()
    run(["conda", "run", "--no-capture-output", "-n", "draftkv-orbia-eval",
         "python", official / "evaluate.py", "--videos", videos,
         "--dataset", dataset, "--config", config, "--output", output,
         "--devices", str(args.gpu)])
    scores = output / "scores/scores.json"
    if not scores.is_file():
        raise ValueError("Official ORBIA evaluation did not create scores/scores.json")
    env = dict(os.environ, PYTHONPATH=str(VERSION))
    run([sys.executable, "-B", "-c",
         "from draftkv.evaluators.orbia_short import main; import sys; "
         "raise SystemExit(main(sys.argv[1:]))",
         "--official-scores", scores,
         "--output-dir", output / "draftkv_480_derived"], env=env)
    print(json.dumps({
        "status": "ORBIA_OFFICIAL_AND_DRAFTKV_DERIVED_OUTPUTS_CREATED",
        "official_scores": str(scores),
        "derived_scores": str(output / "draftkv_480_derived/short_video_scores.json"),
        "protocol": "draftkv.orbia.short_video_derived.v1",
        "official_full_leaderboard_comparable": False
    }, indent=2))


def evaluate_wbench(args):
    root = Path(args.asset_root).expanduser().resolve()
    wbench = root / "evaluation/WBench"
    verified_checkout(wbench, WBENCH_REV)
    video = Path(args.video).expanduser().resolve()
    da3 = Path(args.da3_cache).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not video.is_file():
        raise ValueError(f"Video input not found: {video}")
    for name in ("depth.npy", "extrinsics.npy", "intrinsics.npy"):
        if not (da3 / name).is_file():
            raise ValueError(f"Missing WBench DA3 cache input: {da3 / name}")
    if output.exists():
        raise ValueError(f"Do not overwrite prior evaluation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    module = wbench / "src/metrics/consistency/reconstruction_consistency.py"
    body = """
import importlib.util, json, pathlib, sys
from draftkv.evaluators.table3_reprojection import verify_wbench_source
from draftkv.evaluators.wbench_reprojection_similarity import ReprojectionObserver
wbench, video, da3, dest = map(pathlib.Path, sys.argv[1:])
verify_wbench_source(wbench)
file = wbench / 'src/metrics/consistency/reconstruction_consistency.py'
spec = importlib.util.spec_from_file_location('pinned_wbench_reprojection', file)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
data = ReprojectionObserver(device='cuda:0').run(original, video, da3, fps=3.0)
data['source'] = dict(wbench_commit='edc4e80e642baac4f2b5764dec9f1a491200af8f',
                      video=str(video), da3_cache=str(da3))
with dest.open('x') as f:
    json.dump(data, f, indent=2, allow_nan=False)
    f.write('\\n')
print(json.dumps(dict(status='WBench_REPROJECTION_OBSERVED',
    scores=data['scores'], counts=data['counts'], output=str(dest)), indent=2))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(VERSION)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run(["conda", "run", "--no-capture-output", "-n", "draftkv-wbench-eval",
         "python", "-B", "-c", body, wbench, video, da3, output], env=env)
    if not output.is_file():
        raise ValueError("WBench observer returned without a valid JSON output")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="table", required=True)
    t2 = sub.add_parser("table2", help="ORBIA official metrics plus separately labeled 480-case derivation")
    t2.add_argument("--asset-root", required=True)
    t2.add_argument("--videos", required=True, help="Official case-id videos folder or JSONL list")
    t2.add_argument("--gpu", type=int, default=0)
    t2.add_argument("--output", required=True)
    t3 = sub.add_parser("table3", help="WBench DA3 reprojected PSNR, masked SSIM, masked LPIPS")
    t3.add_argument("--asset-root", required=True)
    t3.add_argument("--video", required=True, help="One generated video")
    t3.add_argument("--da3-cache", required=True, help="Official WBench DA3 precompute cache for this video")
    t3.add_argument("--gpu", type=int, default=0)
    t3.add_argument("--output", required=True, help="New JSON file for per-case measured scores")
    args = p.parse_args(argv)
    if args.gpu < 0:
        p.error("--gpu must be nonnegative")
    try:
        (evaluate_orbia if args.table == "table2" else evaluate_wbench)(args)
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print("EVALUATION_BLOCKED: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
