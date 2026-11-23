#!/usr/bin/env python3
"""Build an original source-locked DraftKV input plan from a prepared request.

Delegates request verification and plan construction to the frozen model
source. Does not invent poses, protected-frame budgets, offsets or attention.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PUBLIC = Path(__file__).resolve().parents[1]
MODELS = {"longlive1.3b": ("longlive", "LongLive1.3B", "bench1", "longlive13"),
          "lingbot1.3b": ("lingbot13", "LingBot1.3B", "bench2", "lingbot14"),
          "lingbot14b": ("h200_large", "LingBot14B", "bench2", "lingbot14"),
          "solarwm33b": ("h200_large", "SolarWM33B", "bench2", None)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(cond, why):
    if not cond:
        raise ValueError(why)


def external(path):
    p = Path(path)
    require(p.is_absolute() and not p.resolve().is_relative_to(PUBLIC),
            "input, output, model files must have absolute external paths")
    return p.resolve()


def interpreter_path(path):
    # A venv discovers pyvenv.cfg through the *invoked* interpreter path.
    # Resolve it only for security checks: never discard a bin/python symlink.
    real = Path(path).expanduser()
    require(real.is_absolute() and not real.resolve().is_relative_to(PUBLIC),
            "model interpreter must be an absolute external virtualenv path")
    invoked = Path(os.path.abspath(str(real)))
    require(invoked.is_file(), "model Python interpreter missing")
    return invoked


def plan(args):
    version, model, bench, native = MODELS[args.backbone]
    root = PUBLIC / "versions" / version
    require((root / "src/draftkv").is_dir(), "frozen model adapter missing")
    request = external(args.request)
    checkpoint = external(args.checkpoint)
    assets = external(args.assets_dir)
    python = interpreter_path(args.python)
    spec_path = external(args.b01_spec)
    cache = external(args.b01_cache)
    target = external(args.output_plan)
    require(request.is_file() and checkpoint.exists() and assets.exists()
            and python.is_file() and spec_path.is_file() and cache.is_dir(),
            "original request/weights/native B01 environment incomplete")
    require(not target.exists(), "source plan output exists; refuse overwrite")
    if native is None:
        require(args.native_source, "SolarWM requires the source-locked authorized original native root")
        source = external(args.native_source)
    else:
        source = root / "src/draftkv/vendor/backbones" / native
    require(source.is_dir(), "original native source missing")
    lock = root / ("configs/backbones/solarwm_source_lock.json"
                   if native is None else "configs/backbones/source_locks.json")
    require(lock.is_file(), "frozen original native source lock missing")
    req = json.loads(request.read_text())
    require(req.get("execution_scope") in ("diagnostic", "release_candidate"),
            "original request has no approved source-bound execution scope")
    require(req.get("backbone", model) == model, "prepared request belongs to another model")
    require(req.get("dataset_manifest_sha256") and req.get("case_id"),
            "request must identify immutable dataset row and manifest")

    # Execute inside the selected model's own interpreter, NOT system Python.
    # This is CPU-only planning; the original code checks source, request,
    # checkpoint and B01 ABI/receipt without loading CUDA or generating.
    code = """
import json,sys,hashlib
import numpy as np  # Load the model-venv dependency before source-owned isolation.
from pathlib import Path
root=Path(sys.argv[1]);model=sys.argv[2];bench=sys.argv[3]
req=Path(sys.argv[4]);checkpoint=sys.argv[5]
assets=Path(sys.argv[6]);python=sys.argv[7]
native=Path(sys.argv[8]);lock=Path(sys.argv[9])
spec_path=Path(sys.argv[10]);cache=Path(sys.argv[11])
output=Path(sys.argv[12]);lora=sys.argv[13];silence=sys.argv[14];profile=sys.argv[15]
from draftkv.registry import Registry
from draftkv.release.audit import release_source_files
from draftkv.backbones.catalog import MODELS
from draftkv.backbones.sequence_parallel import local_heads
from draftkv.backbones.planning import make_plan
from draftkv.streaming.config import select_config
group=Registry().select(bench,model,'draftkv')[0]
source_files=release_source_files(root)
native_parameters=select_config('H200',local_heads=local_heads(
    MODELS[model].heads,MODELS[model].family,1))['parameters']
if not isinstance(native_parameters,dict):
    raise RuntimeError('no canonical H200 source-bound B01 runtime profile')
cfg=dict(request=str(req),request_sha256=hashlib.sha256(req.read_bytes()).hexdigest(),
    gpu_profile=profile=='summary',
    source_root=str(native),source_lock=str(lock),assets_dir=str(assets),python=python,
    algorithm_plugin=dict(root=str(root),
        entry='src/draftkv/algorithms/draftkv/entry.py',
        files=source_files,
        config=dict(importance_config='configs/algorithms/draftkv/scorer.json',
            importance_config_sha256=hashlib.sha256(
                (root/'configs/algorithms/draftkv/scorer.json').read_bytes()).hexdigest(),
            packing_policy='chronological_frame_id',
            profile_admission='runtime',host_pages_policy='quality',
            device_archive_plan=True,gpu_krope_8x16=True,
            b01=dict(build_spec_path=str(spec_path),cache_directory=str(cache),
                     build_spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                     parameters=native_parameters))))
if lora:cfg['lora_checkpoint']=lora
if silence:cfg['silence_path']=silence
planned=make_plan(root,group,1,cfg,checkpoint,output.parent)
planned['selected_devices']=[0]
planned['gpu_profile']=profile=='summary'
planned['measurement_mode']='deployment'
planned['hard_timeout_s']=7200
# The runtime source/selection contract, not the quickstart script,
# owns all algorithm and retrieval settings.
print(json.dumps(planned,allow_nan=False,sort_keys=True))
"""
    env = dict(os.environ, PYTHONPATH=str(root / "src"),
               PATH=str(python.parent) + os.pathsep + os.environ.get("PATH", ""))
    command = [str(python), "-B", "-c", code, str(root), model, bench, str(request),
               str(checkpoint), str(assets), str(python), str(source), str(lock),
               str(spec_path), str(cache), str(target), args.lora or "",
               args.silence or "", args.profile]
    completed = subprocess.run(command, env=env, text=True, capture_output=True)
    require(completed.returncode == 0, "source-owned planner rejected input: " +
            completed.stderr[-1800:])
    data = json.loads(completed.stdout)
    require(data.get("backbone") == model and data.get("algorithm") == "draftkv"
            and data.get("request", {}).get("case_id") == req["case_id"],
            "frozen plan builder returned a different model or input")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(status="SOURCE_LOCKED_INPUT_PLAN_PREPARED",
                          backbone=args.backbone,
                          input=str(request), source_plan=str(target),
                          source_plan_sha256=sha(target),
                          gpu_executed=False), indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", choices=MODELS, required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--assets-dir", required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--b01-spec", required=True)
    p.add_argument("--b01-cache", required=True)
    p.add_argument("--output-plan", required=True)
    p.add_argument("--native-source", help="source-locked native SolarWM root")
    p.add_argument("--profile", choices=("off","summary"), default="off")
    p.add_argument("--lora", help="LongLive LoRA checkpoint")
    p.add_argument("--silence", help="SolarWM original silence asset")
    args = p.parse_args(argv)
    try:
        plan(args)
    except (OSError, ValueError, subprocess.SubprocessError, KeyError, TypeError) as exc:
        p.exit(2, "BLOCKED: " + str(exc) + "\n")


if __name__ == "__main__":
    sys.exit(main())
