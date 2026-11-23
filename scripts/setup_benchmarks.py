#!/usr/bin/env python3
"""Network provisioning of source-pinned ORBIA/WBench/VBench evaluators.

All large resources and Conda environments reside outside the Git checkout.
ORBIA inputs are acquired through the *frozen DraftKV original installer*,
which pins Git-LFS archive identities and validates all 480 inputs.
Do not equate a prepared evaluator with a completed benchmark score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_VERSIONS = {
    "lingbot1.3b": ("lingbot13", "LingBot1.3B"),
    "lingbot14b": ("h200_large", "LingBot14B"),
    "solarwm33b": ("h200_large", "SolarWM33B"),
}
REVISIONS = {
    "orbia": ("https://github.com/worldbench/Orbia.git",
              "f7d8ddde3069a6d5ce75a87fa6b4fc896cc78a59"),
    "wbench": ("https://github.com/meituan-longcat/WBench.git",
               "edc4e80e642baac4f2b5764dec9f1a491200af8f"),
    "vbench": ("https://github.com/Vchitect/VBench.git",
               "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"),
}
ORBIA_LOCK = ROOT / "scripts/benchmark_locks/orbia480.lock.json"
ORBIA_LOCK_SHA256 = "84bd49bc380fee947ab1070c5950912d9be22e376b78c9f0e29170811c60f722"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(args, *, cwd=ROOT, env=None, dry_run=False):
    args = [str(a) for a in args]
    print("+ " + " ".join(__import__("shlex").quote(a) for a in args), flush=True)
    if not dry_run:
        subprocess.run(args, cwd=cwd, check=True, env=env)


def official_checkout(name, directory, *, dry_run=False):
    url, revision = REVISIONS[name]
    if dry_run:
        command(["git", "clone", "--recursive", url, directory], dry_run=True)
        command(["git", "-C", directory, "checkout", "--detach", revision], dry_run=True)
        return
    if directory.exists() and not (directory / ".git").is_dir():
        raise ValueError(f"Refusing to replace a non-Git directory: {directory}")
    if not directory.exists():
        command(["git", "clone", "--recursive", url, directory])
    remote = subprocess.check_output(
        ["git", "-C", str(directory), "remote", "get-url", "origin"], text=True).strip()
    canonical = lambda x: x.lower().removesuffix(".git").removesuffix("/").removeprefix("https://")
    if canonical(remote) != canonical(url):
        raise ValueError(f"Unexpected upstream evaluator remote: {remote}")
    head = subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    if head != revision:
        dirty = subprocess.check_output(
            ["git", "-C", str(directory), "status", "--porcelain"], text=True).strip()
        if dirty:
            raise ValueError(f"Do not overwrite locally modified evaluator: {directory}")
        command(["git", "-C", directory, "fetch", "origin", revision])
        command(["git", "-C", directory, "checkout", "--detach", revision])
    actual = subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise ValueError(f"Expected official revision {revision}, found {actual}")
    if name == "wbench":
        command(["git", "-C", directory, "submodule", "update", "--init", "--recursive"])
    print(f"PASS {name}: official revision {actual}", flush=True)


def frozen_runner(backbone, module, args, *, dry_run):
    version = SOURCE_VERSIONS[backbone][0]
    package_path = ROOT / "versions" / version / "src"
    if not package_path.is_dir():
        raise ValueError(f"Missing frozen DraftKV source: {package_path}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package_path)
    payload = ["import sys; from " + module + " import main; "
               "raise SystemExit(main(sys.argv[1:]))"]
    command([sys.executable, "-B", "-c", payload[0], *args], env=env, dry_run=dry_run)


def official_dataset_lock():
    if sha256(ORBIA_LOCK) != ORBIA_LOCK_SHA256:
        raise ValueError("Pinned original ORBIA metadata lock changed")
    lock = json.loads(ORBIA_LOCK.read_text())
    if (lock["selection"]["short_cases"] != 480 or
            lock["distribution"]["repository"] != "Orbia/Orbia"):
        raise ValueError("Original ORBIA distribution identity changed")
    return lock


def dataset_and_requests(args, root):
    """Reuse SHA-locked original installer, then prepare source-locked requests."""
    lock = official_dataset_lock()
    dataset = root / "datasets" / "orbia"
    cache = root / "downloads" / "orbia-verified"
    receipts = root / "receipts"
    index = receipts / "orbia-installed-files.json"
    prepared = root / "prepared" / args.backbone
    if not args.dry_run:
        receipts.mkdir(parents=True, exist_ok=True)
    report = receipts / ("orbia-install-" +
                         datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    command_args = [
        "--dataset-root", dataset, "--cache", cache, "--index", index,
        "--lock", ORBIA_LOCK, "--stage", "references", "--report", report]
    if index.exists():
        actual = sha256(index)
        if actual != lock["installed_files"]["sha256"]:
            # Existing installed-files receipt can differ if this environment
            # indexes the same official source with different serialization;
            # require explicit validation through the source-owned installer.
            print("Existing ORBIA index checksum differs from historical receipt; "
                  "source-owned verify-only will validate its current identity.")
        command_args.extend(["--verify-only", "--index-sha256", actual])
    # The full parse is expensive; source-owned validation checks indexed files.
    frozen_runner(args.backbone, "draftkv.benchmarks.orbia_install",
                  command_args, dry_run=args.dry_run)
    if args.dry_run:
        index_sha = "<installed index SHA256>"
    else:
        index_sha = sha256(index)
    request_args = [
        "--backbone", SOURCE_VERSIONS[args.backbone][1],
        "--dataset-root", dataset,
        "--index", index,
        "--index-sha256", index_sha,
        "--lock", ORBIA_LOCK,
        "--output", prepared]
    frozen_runner(args.backbone, "draftkv.benchmarks.orbia_requests",
                  request_args, dry_run=args.dry_run)
    if not args.dry_run:
        requests = json.loads((prepared / "requests.json").read_text())["requests"]
        if len(requests) != 480:
            raise ValueError(f"Expected 480 source-locked requests, found {len(requests)}")
        print("PASS 480 ORBIA inference requests prepared and source-verified")


def conda_available(name):
    try:
        probe = subprocess.run(["conda", "run", "-n", name, "python", "--version"],
                               capture_output=True, text=True, timeout=35)
        return probe.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def install_orbia_evaluator(repo, *, dry_run):
    name = "draftkv-orbia-eval"
    if dry_run or not conda_available(name):
        command(["bash", "tools/install_evaluation.sh", "--name", name],
                cwd=repo, dry_run=dry_run)
    command(["conda", "run", "--no-capture-output", "-n", name, "python",
             "tools/download_weights.py", "--workflow", "evaluation"],
            cwd=repo, dry_run=dry_run)
    if not dry_run:
        config = repo / "configs/evaluation.local.yaml"
        if not config.is_file():
            raise ValueError("ORBIA official evaluation config is missing after weight download")


def install_wbench_evaluator(repo, *, dry_run):
    name = "draftkv-wbench-eval"
    if dry_run or not conda_available(name):
        command(["bash", "tools/install.sh", name, "cu124"], cwd=repo, dry_run=dry_run)
    command(["conda", "run", "--no-capture-output", "-n", name, "python",
             "tools/download_weights.py"], cwd=repo, dry_run=dry_run)
    if not dry_run:
        result = subprocess.run(
            ["conda", "run", "-n", name, "python", "-c", "import cv2, numpy, torch, lpips"],
            cwd=repo, capture_output=True, text=True)
        if result.returncode:
            raise ValueError("WBench evaluator CUDA environment is incomplete: "
                             + result.stderr[-700:])


def install_vbench_evaluator(repo, *, dry_run):
    name = "draftkv-vbench-eval"
    if dry_run or not conda_available(name):
        command(["conda", "create", "-y", "-n", name, "python=3.12", "pip"], dry_run=dry_run)
        command(["conda", "run", "--no-capture-output", "-n", name, "python", "-m", "pip",
                 "install", "-r", "requirements.txt"], cwd=repo, dry_run=dry_run)
    print("VBench-Long upstream sources and base environment requested; "
          "exact original compatibility overlay and score weights NOT_VERIFIED.")


def setup(args):
    root = Path(args.asset_root).expanduser().resolve()
    if root.is_relative_to(ROOT):
        raise ValueError("Benchmark assets must be stored outside the Git checkout")
    evalroot = root / "evaluation"
    if args.backbone == "longlive1.3b":
        path = evalroot / "VBench"
        official_checkout("vbench", path, dry_run=args.dry_run)
        install_vbench_evaluator(path, dry_run=args.dry_run)
        print("VBench-Long original prompt/conditioning release unavailable; "
              "no generation input was substituted.")
        return
    dataset_and_requests(args, root)
    orb = evalroot / "Orbia"
    wbench = evalroot / "WBench"
    official_checkout("orbia", orb, dry_run=args.dry_run)
    official_checkout("wbench", wbench, dry_run=args.dry_run)
    install_orbia_evaluator(orb, dry_run=args.dry_run)
    install_wbench_evaluator(wbench, dry_run=args.dry_run)
    print("Official evaluator source revisions and model dependencies prepared. "
          "Full GPU scoring and paper-table equivalence are NOT_VERIFIED.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--backbone", dest="backbone", required=True,
                        choices=("longlive1.3b", *SOURCE_VERSIONS))
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        setup(args)
        print(json.dumps({
            "status": "PLAN_ONLY" if args.dry_run else "BENCHMARK_INSTALL_STAGES_COMPLETED",
            "benchmark_references": "PINNED_OFFICIAL_SOURCES",
            "gpu_scoring": "NOT_EXECUTED",
            "paper_table_equivalence": "NOT_VERIFIED",
        }, indent=2))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError, KeyError) as error:
        print("BENCHMARK_SETUP_BLOCKED: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
