#!/usr/bin/env python3
"""Portable B01 native binding without rewriting the original historical identity.

This is a deployment tool, NOT a claim of historical byte-identical binaries.
Existing original run.sh paths remain unchanged without --portable-evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
VERSIONS = {"lingbot1.3b": "lingbot13", "lingbot14b": "h200_large",
            "solarwm33b": "h200_large", "longlive1.3b": "longlive"}
B01_FIELDS = ("cache_directory", "build_spec_path", "build_spec_sha256")
SPEC_SEMANTICS = ("arch", "source_generation",
                  "cxx_flags", "nvcc_flags", "modules")
ABI_NON_PATH = ("torch_version", "torch_cuda", "cxx11_abi", "python_version",
                "soabi", "ext_suffix")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def absolute_external(path):
    p = Path(path)
    require(p.is_absolute() and not p.resolve().is_relative_to(REPO.resolve()),
            "output/build paths must be absolute and outside the source repository")
    return p.resolve()


def clean_commit():
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                      text=True).strip()
    status = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"],
                                      text=True)
    require(not status, "portable B01 build requires a clean, committed release checkout")
    return commit


def version_root(backbone):
    require(backbone in VERSIONS, "unsupported backbone")
    return REPO / "versions" / VERSIONS[backbone]


def get_spec_from_python(backbone, model_python, cuda_home, artifact_root, original_spec):
    root = version_root(backbone)
    require(Path(model_python).is_file(), "selected Python interpreter missing")
    require(Path(cuda_home).is_absolute(), "CUDA_HOME must be absolute")
    original = read_json(original_spec)
    commit = clean_commit()
    # The frozen B01 source computes and checks the real ABI and all source locks.
    # Never infer current_abi from an old receipt or manually edit the ABI.
    code = """
import json,sys,torch
from pathlib import Path
from draftkv.vendor.b01.build import current_abi,dry_run
from draftkv.streaming.binding import _native_build_compatible
old=json.loads(Path(sys.argv[6]).read_text())
obj=dry_run(sys.argv[1],current_abi(torch),sys.argv[2],
            sys.argv[3],sys.argv[4],sys.argv[5])
print(json.dumps(dict(spec=obj['spec'],
  old_bridge_compatible=_native_build_compatible(old),
  new_bridge_compatible=_native_build_compatible(obj['spec'])),sort_keys=True))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    p = subprocess.run([str(Path(model_python).absolute()), "-B", "-c", code,
                        original["arch"], commit, str(cuda_home),
                        str(artifact_root / "cache"), str(artifact_root / "build-spec.json"),
                        str(Path(original_spec).resolve())],
                       env=env, text=True, capture_output=True)
    require(p.returncode == 0, "original B01 dry_run failed: " + p.stderr[-1500:])
    result = json.loads(p.stdout)
    require(result["old_bridge_compatible"] and result["new_bridge_compatible"],
            "frozen NativeBinding rejects original or portable compatibility generation")
    spec = result["spec"]
    require(spec["abi"]["python_executable"] == str(Path(model_python).absolute()),
            "new ABI does not match selected interpreter identity")
    compare_specs(original, spec)
    return spec


def compare_specs(old, new):
    for key in SPEC_SEMANTICS:
        require(old.get(key) == new.get(key), "B01 scientific/build source changed: " + key)
    for key in ABI_NON_PATH:
        require(old["abi"].get(key) == new["abi"].get(key),
                "B01 ABI changed beyond interpreter location: " + key)
    require(old["abi"]["python_executable"] != new["abi"]["python_executable"],
            "expected a distinct portable interpreter path")
    require(old.get("schema") == new.get("schema") == 1, "B01 spec schema changed")


def prepare(args):
    external = absolute_external(args.artifact_root)
    original = Path(args.original_spec).resolve()
    require(original.is_file(), "original B01 spec missing")
    spec = get_spec_from_python(args.backbone, args.python, args.cuda_home, external,
                                original)
    external.mkdir(parents=True, exist_ok=True)
    target = external / "build-spec.json"
    require(not target.exists(), "portable spec already exists; do not overwrite")
    target.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
    bridge = {"schema": 1, "original_spec_sha256": sha(original),
              "portable_spec_sha256": sha(target),
              "binding_source_sha256": sha(version_root(args.backbone) / "src/draftkv/streaming/binding.py"),
              "old_compatibility_generation": read_json(original)["compatibility_generation"],
              "new_compatibility_generation": spec["compatibility_generation"],
              "native_bridge_accepted": True}
    (external / "compatibility-bridge.json").write_text(json.dumps(bridge, indent=2) + "\n")
    print(json.dumps({"status": "SPEC_PREPARED_BUILD_NOT_RUN",
                      "spec_path": str(target), "spec_sha256": sha(target),
                      "original_spec_sha256": sha(original),
                      "clean_commit": spec["clean_commit"],
                      "new_interpreter": spec["abi"]["python_executable"]}, indent=2))


def prepare_fresh(args):
    """Create a new deployment spec from the frozen B01 source, not a historical copy.

    The public user does not need or possess a historical machine's spec.
    A fresh build identity is distinct from historical scientific evidence.
    """
    artifact = absolute_external(args.artifact_root)
    require(args.arch == "9.0", "fresh B01 H200/H100 SM90 mode is the only GPU-accepted target")
    require(Path(args.python).is_file(), "selected deployment Python interpreter missing")
    require(Path(args.cuda_home).is_absolute(), "CUDA_HOME must be absolute")
    commit = clean_commit()
    root = version_root(args.backbone)
    code = """
import json,sys,torch
from draftkv.vendor.b01.build import current_abi,dry_run
result=dry_run(sys.argv[1],current_abi(torch),sys.argv[2],
               sys.argv[3],sys.argv[4],sys.argv[5])
print(json.dumps(result['spec'],sort_keys=True))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    process = subprocess.run([str(Path(args.python).absolute()), "-B", "-c", code,
                              args.arch, commit, args.cuda_home,
                              str(artifact / "cache"), str(artifact / "build-spec.json")],
                             capture_output=True, text=True, env=env)
    require(process.returncode == 0, "fresh B01 spec generation failed: " + process.stderr[-1200:])
    spec = json.loads(process.stdout)
    require(spec["clean_commit"] == commit
            and spec["arch"] == args.arch
            and spec["abi"]["python_executable"] == str(Path(args.python).absolute()),
            "fresh spec does not match selected source/runtime identity")
    require(not (artifact / "build-spec.json").exists(), "fresh B01 spec exists; refuse overwrite")
    artifact.mkdir(parents=True, exist_ok=True)
    output = artifact / "build-spec.json"
    output.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    evidence = dict(schema=1, scope="CURRENT_SOURCE_DEPLOYMENT_NOT_HISTORICAL_REPRODUCTION",
                    spec_sha256=sha(output), source_commit=commit,
                    arch=args.arch, backbone=args.backbone)
    (artifact / "fresh-source-identity.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(status="FRESH_B01_SPEC_READY_BUILD_NOT_RUN",
                          spec=str(output), **evidence), indent=2))


def build(args):
    root = version_root(args.backbone)
    spec_path = Path(args.spec).resolve()
    spec = read_json(spec_path)
    require(spec["clean_commit"] == clean_commit(), "B01 spec commit is stale")
    require(spec["abi"]["python_executable"] == str(Path(args.python).absolute()),
            "selected Python does not match portable spec")
    require(spec_path == absolute_external(spec_path), "spec must be external")
    cache_root = absolute_external(args.cache_root)
    require(cache_root.resolve() != REPO.resolve(), "cache root invalid")
    env = dict(os.environ, PYTHONPATH=str(root / "src"), CUDA_HOME=spec["cuda_home"],
               TORCH_CUDA_ARCH_LIST=spec["arch"], B01_BUILD_SPEC=str(spec_path),
               B01_CACHE_ROOT=str(cache_root))
    # CPATH, e.g. matching Python.h and pyconfig.h, is a caller-provisioned build
    # prerequisite. The frozen vendor rejects unrecorded CFLAGS/CXXFLAGS overrides.
    p = subprocess.run([str(Path(args.python).absolute()), "-B",
                        str(root / "src/draftkv/vendor/b01/setup_compat.py"), "build_ext"],
                       env=env, check=False)
    require(p.returncode == 0, "original B01 build API failed (exit " + str(p.returncode) + ")")
    candidates = [x for x in cache_root.glob("*/receipt.json")
                  if x.parent.is_dir() and read_json(x).get("spec") == spec]
    require(len(candidates) == 1, "expected one complete verified B01 build receipt")
    print(json.dumps({"status": "B01_BUILT", "receipt": str(candidates[0]),
                      "receipt_sha256": sha(candidates[0])}, indent=2))


def no_binding(config):
    value = json.loads(json.dumps(config))
    b01 = value.get("b01", {})
    for key in B01_FIELDS:
        b01.pop(key, None)
    return value


def scientific_identity(plan, keys):
    # Identical canonical serialization to tools/launch_original.py, except
    # callers can verify the old plan before allowing a new B01 binding.
    config = json.loads(json.dumps(plan["algorithm_plugin"]["config"]))
    for key in ("cache_directory", "build_spec_path"):
        config.get("b01", {}).pop(key, None)
    value = {"original_algorithm_plugin_config_except_external_b01_paths": config,
             **{key: plan[key] for key in keys if key in plan}}
    return canonical(value)


def verify_receipt(spec, directory):
    receipt_path = Path(directory) / "receipt.json"
    require(receipt_path.is_file(), "portable B01 build receipt missing")
    record = read_json(receipt_path)
    require(record.get("schema") == 1
            and record.get("validation") == "BUILD_EXPORTS_ONLY"
            and record.get("spec") == spec, "portable B01 receipt/spec mismatch")
    require(record.get("key") == canonical(dict(spec=spec, toolchain=record.get("toolchain")))
            == Path(directory).name, "portable B01 content-addressed cache key mismatch")
    for name, details in record["binaries"].items():
        p = Path(directory) / details["path"]
        require(details["path"] == name + spec["abi"]["ext_suffix"]
                and p.is_file() and p.stat().st_size and sha(p) == details["sha256"],
                "portable B01 native binary mismatch: " + name)
    require(len(record["binaries"]) == len(spec["modules"])
            and set(record["binaries"]) == set(spec["modules"]),
            "portable B01 module set incomplete")
    return receipt_path


def approve(reference, portable, original_spec, new_spec, cache_directory,
            mapping, reference_sha=None):
    require(reference["algorithm"] == portable["algorithm"] == "draftkv",
            "only Full DraftKV may use portable B01")
    require(reference["backbone"] == portable["backbone"]
            and reference["sp"] == portable["sp"], "backbone/topology changed")
    require(reference["algorithm_plugin"]["entry"] ==
            portable["algorithm_plugin"]["entry"], "plugin entry changed")
    keys = mapping["scientific_keys"]
    require(scientific_identity(reference, keys) ==
            mapping["scientific_sections_sha256"], "reference is not the original approved scientific plan")
    require(no_binding(reference["algorithm_plugin"]["config"]) ==
            no_binding(portable["algorithm_plugin"]["config"]),
            "portable plan changed non-B01 algorithm settings")
    for key in keys:
        if key != "algorithm_plugin":
            require(reference.get(key) == portable.get(key),
                    "portable plan changed approved scientific field: " + key)
    oldb = reference["algorithm_plugin"]["config"]["b01"]
    newb = portable["algorithm_plugin"]["config"]["b01"]
    require(oldb["build_spec_sha256"] == sha(original_spec)
            and newb["build_spec_sha256"] == sha(new_spec),
            "old/new B01 spec SHA mismatch")
    require(Path(newb["build_spec_path"]).resolve() == Path(new_spec).resolve()
            and Path(newb["cache_directory"]).resolve() == Path(cache_directory).resolve(),
            "portable B01 path bindings do not match the approved artifact")
    orig_spec, portable_spec = read_json(original_spec), read_json(new_spec)
    compare_specs(orig_spec, portable_spec)
    if orig_spec["compatibility_generation"] != portable_spec["compatibility_generation"]:
        bridge_path = Path(new_spec).with_name("compatibility-bridge.json")
        require(bridge_path.is_file(), "B01 compatibility bridge attestation missing")
        bridge = read_json(bridge_path)
        version = {"LingBot1.3B": "lingbot13", "LingBot14B": "h200_large",
                   "SolarWM33B": "h200_large", "LongLive1.3B": "longlive"}[portable["backbone"]]
        binding = REPO / "versions" / version / "src/draftkv/streaming/binding.py"
        require(bridge.get("schema") == 1 and bridge.get("native_bridge_accepted") is True
                and bridge.get("original_spec_sha256") == sha(original_spec)
                and bridge.get("portable_spec_sha256") == sha(new_spec)
                and bridge.get("old_compatibility_generation") == orig_spec["compatibility_generation"]
                and bridge.get("new_compatibility_generation") == portable_spec["compatibility_generation"]
                and bridge.get("binding_source_sha256") == sha(binding),
                "B01 compatibility bridge evidence/source mismatch")
    require(portable_spec["clean_commit"] == clean_commit(),
            "portable B01 spec does not bind the current clean checkout")
    receipt = verify_receipt(read_json(new_spec), cache_directory)
    return {"schema": 1, "status": "PORTABLE_B01_APPROVED_NOT_GPU_VALIDATED",
            "reference_plan_sha256": reference_sha, "reference_spec_sha256": sha(original_spec),
            "portable_spec_sha256": sha(new_spec), "native_receipt_sha256": sha(receipt),
            "native_receipt": str(receipt), "original_spec": str(Path(original_spec).resolve()),
            "portable_spec": str(Path(new_spec).resolve()),
            "cache_directory": str(Path(cache_directory).resolve())}


def bind(args):
    reference_path = Path(args.reference_plan).resolve()
    reference = read_json(reference_path)
    old_spec = Path(args.original_spec).resolve()
    new_spec = Path(args.spec).resolve()
    cache = Path(args.cache_directory).resolve()
    require(args.backbone in VERSIONS, "unknown backbone")
    mapping = read_json(REPO / "ORIGINAL_COMMAND_MAPPING.json")[args.backbone + ":sp" + str(reference["sp"])]
    require(reference["backbone"] == {"lingbot1.3b": "LingBot1.3B",
              "lingbot14b": "LingBot14B", "solarwm33b": "SolarWM33B",
              "longlive1.3b": "LongLive1.3B"}[args.backbone], "reference backbone mismatch")
    require(args.output_plan and args.evidence, "output-plan and evidence required")
    new = json.loads(json.dumps(reference))
    b01 = new["algorithm_plugin"]["config"]["b01"]
    b01.update(cache_directory=str(cache), build_spec_path=str(new_spec),
               build_spec_sha256=sha(new_spec))
    new["repo_commit"] = clean_commit()
    new["run_dir"] = str(absolute_external(args.run_dir))
    new["output"] = str(absolute_external(args.output))
    require(not Path(new["output"]).exists(), "output exists")
    result = approve(reference, new, old_spec, new_spec, cache, mapping, sha(reference_path))
    result["portable_plan_sha256"] = canonical(new)
    result["source_commit"] = new["repo_commit"]
    target_plan = absolute_external(args.output_plan)
    evidence = absolute_external(args.evidence)
    require(not target_plan.exists() and not evidence.exists(), "refusing overwrite of plan/evidence")
    require(target_plan.parent == evidence.parent == Path(new["run_dir"]).resolve(),
            "plan and evidence must be directly inside the selected run_dir")
    require(target_plan.name == "plan.json", "original worker requires run_dir/plan.json")
    target_plan.parent.mkdir(parents=True, exist_ok=True)
    target_plan.write_text(json.dumps(new, sort_keys=True, indent=2) + "\n")
    result["portable_plan_sha256"] = sha(target_plan)
    result["reference_plan"] = str(reference_path)
    evidence.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "PORTABLE_PLAN_BOUND", "plan": str(target_plan),
                      "evidence": str(evidence), "GPU_generation": "NOT_TESTED"}, indent=2))


def verify_portable_plan(portable_plan, evidence_path, mapping):
    evidence = read_json(evidence_path)
    require(evidence.get("schema") == 1
            and evidence.get("status") == "PORTABLE_B01_APPROVED_NOT_GPU_VALIDATED",
            "unknown portable B01 evidence schema/status")
    require(evidence["portable_plan_sha256"] == sha(portable_plan),
            "portable plan bytes changed after approval")
    reference = read_json(evidence["reference_plan"])
    require(evidence["reference_plan_sha256"] == sha(evidence["reference_plan"]),
            "reference plan changed after approval")
    new = read_json(portable_plan)
    result = approve(reference, new, evidence["original_spec"], evidence["portable_spec"],
                     evidence["cache_directory"], mapping, sha(evidence["reference_plan"]))
    require(result["native_receipt_sha256"] == evidence["native_receipt_sha256"],
            "B01 portable receipt changed after approval")
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--backbone", required=True, choices=VERSIONS)
    prep.add_argument("--python", required=True)
    prep.add_argument("--cuda-home", required=True)
    prep.add_argument("--original-spec", required=True)
    prep.add_argument("--artifact-root", required=True)
    fresh = sub.add_parser("prepare-fresh")
    fresh.add_argument("--backbone", choices=VERSIONS, required=True)
    fresh.add_argument("--python", required=True)
    fresh.add_argument("--cuda-home", required=True)
    fresh.add_argument("--arch", choices=("9.0",), default="9.0")
    fresh.add_argument("--artifact-root", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--backbone", required=True, choices=VERSIONS)
    build_parser.add_argument("--python", required=True)
    build_parser.add_argument("--spec", required=True)
    build_parser.add_argument("--cache-root", required=True)
    bind_parser = sub.add_parser("bind")
    bind_parser.add_argument("--backbone", required=True, choices=VERSIONS)
    bind_parser.add_argument("--reference-plan", required=True)
    bind_parser.add_argument("--original-spec", required=True)
    bind_parser.add_argument("--spec", required=True)
    bind_parser.add_argument("--cache-directory", required=True)
    bind_parser.add_argument("--run-dir", required=True)
    bind_parser.add_argument("--output", required=True)
    bind_parser.add_argument("--output-plan", required=True)
    bind_parser.add_argument("--evidence", required=True)
    a = p.parse_args(argv)
    try:
        {"prepare": prepare, "prepare-fresh": prepare_fresh, "build": build, "bind": bind}[a.action](a)
    except (KeyError, ValueError, OSError, json.JSONDecodeError, subprocess.CalledProcessError) as e:
        p.exit(2, "BLOCKED: " + str(e) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
