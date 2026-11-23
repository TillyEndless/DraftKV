#!/usr/bin/env python3
"""Concise, source-preserving bootstrap/inference facade for DraftKV.

This wrapper never rewrites model configuration, fabricates benchmark inputs, or
claims inference success without a completed video. It delegates to the audited
native DraftKV CLI and uses separate benchmark provisioning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = ("longlive1.3b", "lingbot1.3b", "lingbot14b", "solarwm33b")
MODEL_NAMES = {
    "lingbot1.3b": "LingBot1.3B",
    "lingbot14b": "LingBot14B",
    "solarwm33b": "SolarWM33B",
    "longlive1.3b": "LongLive1.3B",
}
DEFAULT_ASSET_ROOT = Path.home() / ".cache" / "draftkv" / "assets"
DEFAULT_ENV_ROOT = Path.home() / ".cache" / "draftkv-install"
ORBIA_CASE_COUNT = 480


def absolute_path(value):
    return Path(value).expanduser().resolve()


def run_command(command, dry_run=False):
    command = [str(part) for part in command]
    print("+ " + " ".join(shlex.quote(s) for s in command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def setup(args):
    cli = ROOT / "scripts/draftkv.sh"
    command = [cli, "bootstrap", "--backbone", args.backbone,
               "--gpu", str(args.gpu), "--env-mode", args.env_mode,
               "--asset-root", absolute_path(args.asset_root),
               "--env-root", absolute_path(args.env_root)]
    if args.with_dataset:
        command.append("--with-dataset")
    if args.python:
        command.extend(("--python", args.python))
    if args.dry_run:
        command.append("--dry-run")
    run_command(command)
    if args.with_evaluators:
        command = [sys.executable, ROOT / "scripts/setup_benchmarks.py",
                   "--backbone", args.backbone,
                   "--asset-root", absolute_path(args.asset_root),
                   "--gpu", str(args.gpu)]
        if args.dry_run:
            command.append("--dry-run")
        run_command(command)
    print(json.dumps({
        "status": "SETUP_PLAN_ONLY" if args.dry_run else "SETUP_STAGES_COMPLETED",
        "backbone": args.backbone,
        "asset_root": str(absolute_path(args.asset_root)),
        "environment_root": str(absolute_path(args.env_root)),
        "evaluators_requested": args.with_evaluators,
        "video_generation": "NOT_EXECUTED",
    }, indent=2))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def requests_for_backbone(root, backbone):
    """Legacy single-case discovery: the native planner revalidates each input."""
    base = root / "prepared" / backbone
    if not base.is_dir():
        return []
    candidates = []
    for path in sorted(base.rglob("request.json")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if isinstance(payload, dict) and payload.get("case_id"):
            candidates.append((str(payload["case_id"]), path))
    return candidates


def verified_batch_requests(root, backbone):
    """Validate all original, frozen ORBIA requests against their complete index."""
    if backbone == "longlive1.3b":
        raise ValueError("LongLive batch inputs are not published as a complete "
                         "source-locked cohort; --all cannot run without them.")
    base = root / "prepared" / backbone
    index_path = base / "requests.json"
    if not index_path.is_file() or index_path.is_symlink():
        raise ValueError(f"Missing source-locked ORBIA request index: {index_path}; "
                         "run 'reproduce.sh setup' with authorized ORBIA access.")
    index = json.loads(index_path.read_text())
    requests = index.get("requests")
    if (index.get("schema_version") != 1 or index.get("backbone") != MODEL_NAMES[backbone]
            or index.get("input_mode") != "orbia" or not isinstance(requests, dict)
            or len(requests) != ORBIA_CASE_COUNT):
        raise ValueError(f"Expected an unmodified, complete {ORBIA_CASE_COUNT}-case "
                         f"ORBIA index for {backbone}; refusing partial batch.")
    receipt_path = base / "preparation_receipt.json"
    receipt_sha = index.get("preparation_receipt_sha256")
    if (not receipt_path.is_file() or not isinstance(receipt_sha, str)
            or sha256_file(receipt_path) != receipt_sha):
        raise ValueError("ORBIA preparation receipt SHA-256 mismatch")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("status") != "READY" or receipt.get("backbone") != MODEL_NAMES[backbone]
            or receipt.get("case_ids") != list(requests)
            or receipt.get("dataset_manifest_sha256") != index.get("dataset_manifest_sha256")
            or receipt.get("dataset_index_sha256") != index.get("dataset_index_sha256")):
        raise ValueError("ORBIA preparation receipt differs from frozen requests index")
    entries = []
    for case_id, binding in requests.items():
        if (not isinstance(case_id, str) or not case_id
                or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
                       for c in case_id)
                or case_id in (".", "..") or ".." in case_id):
            raise ValueError(f"Unsafe case ID in ORBIA request index: {case_id!r}")
        if not isinstance(binding, dict) or set(binding) != {"request", "request_sha256"}:
            raise ValueError(f"Invalid request binding for {case_id}")
        path = base / "cases" / case_id / "request.json"
        stored_path = Path(binding["request"])
        if not stored_path.is_absolute() or stored_path.resolve() != path.resolve():
            raise ValueError(f"Case {case_id} request path differs from prepared index")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing ORBIA case request: {path}")
        digest = sha256_file(path)
        if digest != binding["request_sha256"]:
            raise ValueError(f"ORBIA case {case_id} request SHA-256 mismatch")
        payload = json.loads(path.read_text())
        if (payload.get("case_id") != case_id or
                payload.get("backbone") != MODEL_NAMES[backbone] or
                payload.get("input_mode") != "orbia" or
                payload.get("dataset_manifest_sha256") != index["dataset_manifest_sha256"] or
                payload.get("dataset_index_sha256") != index["dataset_index_sha256"]):
            raise ValueError(f"ORBIA case {case_id} request identity mismatch")
        entries.append((case_id, path, digest))
    return index_path, sha256_file(index_path), entries


def inference_command(backbone, gpu, profile, request, output, env_root, asset_root, env_mode, *, dry_run):
    command = [ROOT / "scripts/draftkv.sh", "run",
               "--backbone", backbone,
               "--gpu", str(gpu),
               "--profile", profile,
               "--input", request, "--output", output,
               "--env-root", env_root, "--asset-root", asset_root,
               "--env-mode", env_mode]
    if dry_run:
        command.append("--dry-run")
    return command


def _atomic_json(path, data):
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         prefix=".batch-", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            name = stream.name
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def infer_all(args):
    asset_root = absolute_path(args.asset_root)
    env_root = absolute_path(args.env_root)
    index_path, index_sha, cases = verified_batch_requests(asset_root, args.backbone)
    if args.resume and not args.output_dir:
        raise ValueError("--resume requires explicit --output-dir")
    if args.output_dir:
        batch_root = absolute_path(args.output_dir)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        batch_root = asset_root / "runs" / args.backbone / ("batch-" + timestamp)
    if batch_root.is_relative_to(ROOT):
        raise ValueError("Batch outputs must be outside the Git checkout")
    if batch_root.is_symlink():
        raise ValueError(f"Do not use a symlinked batch output directory: {batch_root}")
    manifest_path = batch_root / "batch.json"
    identity = {
        "schema_version": 1,
        "backbone": args.backbone,
        "gpu": args.gpu,
        "profile": args.profile,
        "env_mode": args.env_mode,
        "asset_root": str(asset_root),
        "env_root": str(env_root),
        "requests_index": str(index_path),
        "requests_index_sha256": index_sha,
        "case_ids": [case_id for case_id, _, _ in cases],
    }
    if args.dry_run:
        print(f"PLAN {len(cases)} verified ORBIA cases → {batch_root}")
        for case_id, request, _ in cases:
            output = batch_root / "cases" / case_id / "attempt-0001" / "video.mp4"
            run_command(inference_command(
                args.backbone, args.gpu, args.profile, request, output,
                env_root, asset_root, args.env_mode, dry_run=True), dry_run=True)
        return
    if args.resume:
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError(f"Cannot resume missing batch manifest: {manifest_path}")
        state = json.loads(manifest_path.read_text())
        if any(state.get(key) != value for key, value in identity.items()):
            raise ValueError("Batch identity/request index changed; refuse unsafe resume")
        if not isinstance(state.get("cases"), dict):
            raise ValueError("Existing batch manifest is invalid")
    else:
        if batch_root.exists():
            raise ValueError(f"Batch output directory already exists: {batch_root}; "
                             "use --resume with --output-dir to continue an existing batch")
        batch_root.mkdir(parents=True)
        state = dict(identity, cases={})
        _atomic_json(manifest_path, state)
    # Exclusive advisory lock avoids two writers overwriting the same manifest.
    import fcntl
    with (batch_root / ".batch.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another process is already running this batch") from error
        completed = skipped = failed = 0
        for i, (case_id, request, digest) in enumerate(cases, 1):
            existing = state["cases"].get(case_id, {})
            if existing.get("status") == "completed":
                video = batch_root / existing.get("output", "")
                if (existing.get("request_sha256") != digest or not video.is_file()
                        or video.stat().st_size <= 0):
                    raise ValueError(f"Completed case output or request has changed: {case_id}")
                print(f"SKIP [{i}/{len(cases)}] {case_id}: completed")
                skipped += 1
                continue
            attempt = int(existing.get("attempt", 0)) + 1
            output = batch_root / "cases" / case_id / f"attempt-{attempt:04d}" / "video.mp4"
            if output.parent.exists():
                raise ValueError(f"Output directory already exists: {output.parent}")
            print(f"RUN [{i}/{len(cases)}] {case_id} attempt={attempt}", flush=True)
            state["cases"][case_id] = {
                "status": "running", "attempt": attempt,
                "request_sha256": digest,
                "output": str(output.relative_to(batch_root)),
            }
            _atomic_json(manifest_path, state)
            try:
                run_command(inference_command(
                    args.backbone, args.gpu, args.profile, request, output,
                    env_root, asset_root, args.env_mode, dry_run=False))
                if not output.is_file() or output.stat().st_size <= 0:
                    raise ValueError(f"No completed video was written for {case_id}")
            except (OSError, ValueError, subprocess.CalledProcessError) as error:
                state["cases"][case_id]["status"] = "failed"
                state["cases"][case_id]["error"] = str(error)
                failed += 1
                _atomic_json(manifest_path, state)
                print(f"FAIL [{i}/{len(cases)}] {case_id}: {error}", file=sys.stderr)
                if not args.keep_going:
                    break
            else:
                state["cases"][case_id]["status"] = "completed"
                state["cases"][case_id]["video_bytes"] = output.stat().st_size
                completed += 1
                _atomic_json(manifest_path, state)
                print(f"PASS [{i}/{len(cases)}] {case_id}")
        finished = sum(v.get("status") == "completed" for v in state["cases"].values())
        print(json.dumps({
            "status": "COMPLETE" if finished == len(cases) and failed == 0 else "INCOMPLETE",
            "backbone": args.backbone, "expected_cases": len(cases),
            "completed_total": finished, "completed_this_run": completed,
            "skipped_this_run": skipped, "failed_this_run": failed,
            "batch_manifest": str(manifest_path),
        }, indent=2))
        if finished != len(cases) or failed:
            raise ValueError(f"Batch incomplete: {finished}/{len(cases)} completed; "
                             f"resume with --all --resume --output-dir {batch_root}")


def infer(args):
    if getattr(args, "all", False):
        return infer_all(args)
    asset_root = absolute_path(args.asset_root)
    env_root = absolute_path(args.env_root)
    if args.input:
        request = absolute_path(args.input)
        if not request.is_file() and not args.dry_run:
            raise ValueError(f"Input plan is missing: {request}")
        case_id = request.parent.name
    else:
        candidates = requests_for_backbone(asset_root, args.backbone)
        if args.case_id:
            candidates = [(case_id, path) for case_id, path in candidates
                          if case_id == args.case_id]
        if not candidates:
            raise ValueError(
                "No source-verified prepared request was found for "
                f"{args.backbone} under {asset_root / 'prepared' / args.backbone}. "
                "Bootstrap downloads are not generation-ready requests. "
                "Use the original 'draftkv.sh prepare' command with an authorized "
                "case index/source lock, or provide --input /path/to/request.json."
            )
        case_id, request = candidates[0]
        print(f"Selecting installed case: {case_id} ({request})", flush=True)
    if args.output:
        output = absolute_path(args.output)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        safe_case = "".join(c if c.isalnum() or c in "-_" else "_" for c in case_id)
        output = asset_root / "runs" / args.backbone / (safe_case + "-" + stamp) / "video.mp4"
    command = inference_command(
        args.backbone, args.gpu, args.profile, request, output,
        env_root, asset_root, args.env_mode, dry_run=args.dry_run)
    run_command(command)
    print(json.dumps({
        "status": "INFERENCE_PLAN_ONLY" if args.dry_run else "INFERENCE_COMMAND_COMPLETED",
        "backbone": args.backbone, "case_id": case_id,
        "output": str(output), "gpu": args.gpu, "profile": args.profile,
    }, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Two-stage DraftKV reproduction: prepare online assets; run verified cases.")
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("setup", "infer"):
        p = commands.add_parser(name)
        p.add_argument("--model", "--backbone", dest="backbone", required=True, choices=BACKBONES)
        p.add_argument("--gpu", type=int, default=0)
        p.add_argument("--asset-root", default=str(DEFAULT_ASSET_ROOT))
        p.add_argument("--env-root", default=str(DEFAULT_ENV_ROOT))
        p.add_argument("--env-mode", default="auto", choices=("auto", "conda", "venv"))
        p.add_argument("--dry-run", action="store_true")
        if name == "setup":
            p.add_argument("--python")
            p.add_argument("--with-dataset", action=argparse.BooleanOptionalAction, default=True)
            p.add_argument("--with-evaluators", action=argparse.BooleanOptionalAction, default=True)
        else:
            p.add_argument("--profile", choices=("on", "off"), default="off")
            p.add_argument("--input", help="Explicit verified request or source-locked plan")
            p.add_argument("--case-id", help="Select from installed verified requests")
            p.add_argument("--output", help="Output video path (single-case mode)")
            p.add_argument("--all", action="store_true",
                           help="Run all 480 source-verified ORBIA cases")
            p.add_argument("--output-dir", help="Batch root directory (batch mode)")
            p.add_argument("--resume", action="store_true", help="Resume batch using --output-dir")
            p.add_argument("--keep-going", action="store_true",
                           help="Continue remaining batch cases after a failed case")
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    if args.action == "infer":
        if args.all and (args.input or args.case_id or args.output):
            parser.error("--all cannot be combined with --input, --case-id or --output")
        if not args.all and (args.output_dir or args.resume or args.keep_going):
            parser.error("--output-dir/--resume/--keep-going require --all")
    try:
        {"setup": setup, "infer": infer}[args.action](args)
        return 0
    except (ValueError, subprocess.CalledProcessError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
