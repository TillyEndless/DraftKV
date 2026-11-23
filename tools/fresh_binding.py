"""Independent fresh-install DraftKV B01 plan binder; not historical experiment replay.

Consumes a source-locked plan produced by the original model's native planner,
rebinds only external run/profile paths and verified B01 cache, then seals
both input and deployment plans with independent SHA-256 provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"lingbot1.3b": ("lingbot13", "LingBot1.3B"),
          "lingbot14b": ("h200_large", "LingBot14B"),
          "solarwm33b": ("h200_large", "SolarWM33B"),
          "longlive1.3b": ("longlive", "LongLive1.3B")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def external(path):
    p = Path(path)
    require(p.is_absolute() and not p.resolve().is_relative_to(ROOT),
            "all input/output/receipt paths must be absolute and outside checkout")
    return p.resolve()


def source_identity():
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                     text=True).strip()
    status = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"],
                                     text=True)
    require(not status, "checkout must be clean and committed")
    return commit


def deployment_source_check(plan, selected):
    version, canonical_model = MODELS[selected]
    source = ROOT / "versions" / version
    require(plan.get("backbone") == canonical_model
            and plan.get("sp") == 1 and plan.get("algorithm") == "draftkv",
            "fresh binding only supports the selected single-GPU DraftKV backbone")
    require(Path(plan.get("repo_root", "")).resolve() == source.resolve(),
            "prepared input plan must use the current source tree")
    plugin = plan.get("algorithm_plugin") or {}
    require(plugin.get("entry") == "src/draftkv/algorithms/draftkv/entry.py"
            and Path(plugin.get("root", "")).resolve() == source.resolve(),
            "plan must use the canonical source-locked DraftKV plugin")
    import importlib.util
    loader = importlib.util.spec_from_file_location(
        "draftkv_runtime_sources", Path(__file__).resolve().parents[1] / "scripts/runtime_sources.py")
    current = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(current)
    for fields in (plan.get("adapter_source_files"), plugin.get("files")):
        current.check_inventory(ROOT, version, fields)
    for key in ("checkpoint", "assets_dir", "python", "source_root", "native_root"):
        require(Path(plan.get(key, "")).exists(), "input plan lacks native dependency: " + key)
    request = plan.get("request") or {}
    require(isinstance(request, dict) and isinstance(request.get("dataset_manifest"), str),
            "original planner must provide an immutable request and dataset manifest")
    manifest = external(request["dataset_manifest"])
    require(manifest.is_file() and sha(manifest) == request.get("dataset_manifest_sha256"),
            "prepared dataset manifest SHA differs")
    if request.get("image"):
        image = external(request["image"])
        require(image.is_file() and sha(image) == request.get("asset_sha256", {}).get("image"),
                "prepared image SHA differs")
    if request.get("camera_dir"):
        camera = external(request["camera_dir"])
        for name in ("poses", "intrinsics"):
            path = camera / (name + ".npy")
            require(path.is_file() and sha(path) == request.get("asset_sha256", {}).get(name),
                    "prepared camera SHA differs: " + name)
    require(isinstance(plugin.get("config", {}).get("b01"), dict),
            "prepared plan lacks canonical B01 parameters")


def stripped(plan):
    result = json.loads(json.dumps(plan))
    for k in ("run_dir", "output", "repo_commit", "gpu_profile"):
        result.pop(k, None)
    b01 = result["algorithm_plugin"]["config"]["b01"]
    for k in ("cache_directory", "build_spec_path", "build_spec_sha256"):
        b01.pop(k, None)
    return result


def native_receipt(spec_path, cache_path):
    import importlib.util
    module = importlib.util.spec_from_file_location("_portable_b01", ROOT / "tools/portable_b01.py")
    original = importlib.util.module_from_spec(module)
    module.loader.exec_module(original)
    spec = read(spec_path)
    require(spec["clean_commit"] == source_identity(),
            "new B01 build spec belongs to a different checkout")
    receipt_path = original.verify_receipt(spec, cache_path)
    return sha(receipt_path)


def bind(args):
    original_path = external(args.input_plan)
    spec_path = external(args.spec)
    cache_path = external(args.cache_directory)
    run_dir = external(args.run_dir)
    target = run_dir / "plan.json"
    evidence_path = run_dir / "deployment-evidence.json"
    require(original_path.is_file() and spec_path.is_file() and cache_path.is_dir(),
            "prepared plan, fresh B01 spec or native receipt directory missing")
    require(not run_dir.exists(), "run directory already exists; refuse overwrite")
    original = read(original_path)
    deployment_source_check(original, args.backbone)
    receipt_sha = native_receipt(spec_path, cache_path)
    prepared = json.loads(json.dumps(original))
    b01 = prepared["algorithm_plugin"]["config"]["b01"]
    b01.update(cache_directory=str(cache_path), build_spec_path=str(spec_path),
               build_spec_sha256=sha(spec_path))
    prepared["repo_commit"] = source_identity()
    prepared["run_dir"] = str(run_dir)
    prepared["output"] = str(external(args.output))
    prepared["gpu_profile"] = args.profile == "summary"
    require(Path(prepared["output"]).parent == run_dir,
            "output must be directly inside the new run directory")
    require(not Path(prepared["output"]).exists(), "output exists")
    # No changes to any retrieval budget, model source, camera or input semantics.
    require(stripped(prepared) == stripped(original), "new run changed scientific configuration")
    run_dir.mkdir(parents=True)
    target.write_text(json.dumps(prepared, indent=2, sort_keys=True) + "\n")
    evidence = dict(schema=1, status="FRESH_B01_DEPLOYMENT_APPROVED",
                    scope="NEW_INSTALL_NOT_HISTORICAL_BENCHMARK",
                    source_commit=source_identity(), backbone=args.backbone,
                    input_plan=str(original_path), input_plan_sha256=sha(original_path),
                    deployed_plan_sha256=sha(target),
                    build_spec=str(spec_path), build_spec_sha256=sha(spec_path),
                    native_cache=str(cache_path), native_receipt_sha256=receipt_sha,
                    gpu_profile=prepared["gpu_profile"])
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(status=evidence["status"], plan=str(target),
                          evidence=str(evidence_path)), indent=2))


def verify(plan_path, evidence_path):
    path = external(plan_path)
    proof = read(external(evidence_path))
    require(proof.get("schema") == 1
            and proof.get("status") == "FRESH_B01_DEPLOYMENT_APPROVED",
            "invalid fresh deployment attestation")
    require(proof["source_commit"] == source_identity()
            and proof["deployed_plan_sha256"] == sha(path)
            and proof["input_plan_sha256"] == sha(proof["input_plan"]),
            "fresh deployment source/input/output changed")
    original, deployed = read(proof["input_plan"]), read(path)
    deployment_source_check(original, proof["backbone"])
    require(stripped(original) == stripped(deployed),
            "deployed input or algorithm settings changed")
    require(deployed["repo_commit"] == source_identity()
            and deployed["gpu_profile"] == proof["gpu_profile"],
            "fresh run profile or commit changed")
    require(Path(deployed["run_dir"]).resolve() == path.parent
            and Path(deployed["output"]).parent == path.parent,
            "fresh run directory or video output location changed")
    b01 = deployed["algorithm_plugin"]["config"]["b01"]
    require(Path(b01["build_spec_path"]).resolve() == Path(proof["build_spec"])
            and Path(b01["cache_directory"]).resolve() == Path(proof["native_cache"])
            and b01["build_spec_sha256"] == proof["build_spec_sha256"] == sha(proof["build_spec"]),
            "fresh B01 binding changed")
    require(native_receipt(proof["build_spec"], proof["native_cache"])
            == proof["native_receipt_sha256"], "fresh native receipt modified")
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", choices=MODELS, required=True)
    p.add_argument("--input-plan", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--cache-directory", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--profile", choices=("off", "summary"), default="off")
    args = p.parse_args(argv)
    try:
        bind(args)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        p.exit(2, "BLOCKED: " + str(exc) + "\n")


if __name__ == "__main__":
    sys.exit(main())
