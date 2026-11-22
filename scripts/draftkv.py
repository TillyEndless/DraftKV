#!/usr/bin/env python3
"""One-command deployment and generation facade over the unmodified DraftKV worker.

'run --input' requires an audited original-schema prepared input plan, not a
raw photo or arbitrary prompt. No implicit camera, retrieval or budget inference.
"""
import argparse
import hashlib
import re
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = ("longlive1.3b", "lingbot1.3b", "lingbot14b", "solarwm33b")
# Model-specific *portable* additions; these are not reconstructed historical pins.
# In particular SolarWM uses Pillow 9.5.0 for Transformers 5.12's Qwen3VL
# Resampling API, rather than modifying the frozen original Pillow 9.0.1 pin.
PORTABLE_EXTRAS = {
    "longlive1.3b": {"scipy": "1.15.3", "imageio": "2.37.4", "imageio-ffmpeg": "0.6.0"},
    "lingbot1.3b": {},
    "lingbot14b": {"ftfy": "6.3.1", "wcwidth": "0.2.14",
                     "scipy": "1.15.3", "imageio": "2.38.0"},
    "solarwm33b": {"Pillow": "9.5.0", "imageio-ffmpeg": "0.6.0"},
}
TORCHVISION_CU124 = {"lingbot14b", "solarwm33b"}

# Exactly the same private cache contract as _install.install_env_vars.
# A fresh install is not usable for admitted cold-runtime profiling when the
# launching worker silently falls back to unrelated /tmp or user Triton caches.
PORTABLE_CACHE_LAYOUT = {
    "PIP_CACHE_DIR": "cache/pip",
    "TMPDIR": "tmp",
    "TORCH_EXTENSIONS_DIR": "cache/torch_extensions",
    "DRAFTKV_PREPACK_BUILD": "cache/prepack",
    "TRITON_CACHE_DIR": "cache/triton",
    "TORCHINDUCTOR_CACHE_DIR": "cache/torchinductor",
    "CUDA_CACHE_PATH": "cache/cuda",
    "HF_HOME": "cache/huggingface",
}



def require(test, message):
    if not test:
        raise ValueError(message)


def external(path):
    p = Path(path).expanduser()
    require(p.is_absolute() and not p.resolve().is_relative_to(ROOT),
            "paths must be absolute and outside the source checkout")
    return p.resolve()


def commands(parts, env=None, capture=False):
    if capture:
        return subprocess.run([str(v) for v in parts], text=True, capture_output=True,
                              env=env, check=True)
    subprocess.run([str(v) for v in parts], env=env, check=True)
    return None


def paths(args):
    env_root = external(args.env_root)
    base = env_root / args.backbone
    return base, base / "venv/bin/python", base / "native/b01", base / "native/fa3"


def cache_paths_for(python):
    """Restore the setup cache identity without resolving the venv/bin symlink."""
    base = Path(python).absolute().parent.parent.parent
    root = base.resolve()
    paths = {name: base / suffix for name, suffix in PORTABLE_CACHE_LAYOUT.items()}
    require(all(path.resolve().is_relative_to(root) for path in paths.values()),
            "portable runtime cache escapes model environment via symlink")
    return {name: str(path) for name, path in paths.items()}


def env_for(python, *, headers=None, cuda_home=None):
    """Carry the audited build prerequisites into runtime torch JIT kernels."""
    env = dict(os.environ)
    py = Path(python)
    env["PATH"] = str(py.parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_CONFIG_FILE"] = os.devnull
    env["PIP_EXTRA_INDEX_URL"] = ""
    caches = cache_paths_for(python)
    env.update(caches)
    receipt = py.parent.parent.parent / "native/build-env.json"
    if receipt.is_file():
        attestation = json.loads(receipt.read_text())
        require(attestation.get("schema_version") == 1 and
                attestation.get("python") == str(py), "wrong portable build environment receipt")
        headers = headers or attestation["python_includes"]
        cuda_home = cuda_home or attestation["cuda_home"]
        require(attestation["torch_extensions_dir"] == caches["TORCH_EXTENSIONS_DIR"],
                "portable native receipt and runtime extension cache disagree")
        if "runtime_cache_paths" in attestation:
            require(attestation["runtime_cache_paths"] == caches,
                    "portable native receipt and runtime cache identity disagree")
    if headers:
        for directory in headers:
            require(Path(directory).is_dir(), "Python development include path disappeared: " + str(directory))
        env["CPATH"] = os.pathsep.join([*map(str, headers),
                                         *([env["CPATH"]] if env.get("CPATH") else [])])
    if cuda_home:
        require((Path(cuda_home) / "bin/nvcc").is_file(), "CUDA compiler vanished: " + str(cuda_home))
        env["CUDA_HOME"] = str(cuda_home)
        env["PATH"] = str(Path(cuda_home) / "bin") + os.pathsep + env["PATH"]
    return env


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def verify_user_fa2_wheel(wheel, digest):
    require(wheel and digest, "install requires --fa2-wheel and --fa2-sha256; "
            "FA3 cannot replace the native FlashAttention-2 distribution")
    path = external(wheel)
    require(path.is_file() and path.suffix == ".whl", "authorized FlashAttention-2 wheel missing")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", digest)), "--fa2-sha256 must be a lowercase SHA-256")
    require(path.name.startswith("flash_attn-"), "wheel must be the flash_attn distribution")
    require(sha256(path) == digest, "supplied FA2 wheel SHA-256 mismatch; refusing installation")
    return path


def resolve_python_headers(python, supplied_root=None):
    """Require matching Python.h + pyconfig.h; never substitute an ABI."""
    result = commands([python, "-B", "-c",
                       "import json,sys,sysconfig; print(json.dumps({"
                       "'python':list(sys.version_info[:2]),"
                       "'include':sysconfig.get_path('include')}))"], capture=True)
    identity = json.loads(result.stdout)
    require(identity["python"] == [3, 10], "portable B01 requires CPython 3.10")
    version = "python3.10"
    if supplied_root:
        root = external(supplied_root)
        options = [root / version, root, root / "include" / version]
        configs = [root / "x86_64-linux-gnu" / version, root / "include/x86_64-linux-gnu" / version,
                   root / version, root]
    else:
        include = Path(identity["include"])
        options = [include]
        configs = [include, Path("/usr/include/x86_64-linux-gnu") / version]
    python_dir = next((p for p in options if (p / "Python.h").is_file()), None)
    config_dir = next((p for p in configs if (p / "pyconfig.h").is_file()), None)
    require(python_dir is not None and config_dir is not None,
            "matching CPython 3.10 Python.h/pyconfig.h missing; use --python-dev-root")
    return list(dict.fromkeys((str(python_dir.resolve()), str(config_dir.resolve()))))


def install_model_closure(args, python, base, headers):
    """Install source-verified user-owned FA2 wheel and explicit portable extras."""
    wheel = verify_user_fa2_wheel(args.fa2_wheel, args.fa2_sha256)
    pip = [python, "-m", "pip", "--require-virtualenv", "install", "--no-deps"]
    commands([*pip, "--force-reinstall", wheel])
    extras = {"setuptools": "84.0.0", "ninja": "1.13.2", **PORTABLE_EXTRAS[args.backbone]}
    commands([*pip, "--only-binary=:all:", "--index-url", "https://pypi.org/simple",
              *(name + "==" + version for name, version in extras.items())])
    if args.backbone in TORCHVISION_CU124:
        commands([*pip, "--only-binary=:all:", "--index-url", "https://download.pytorch.org/whl/cu124",
                  "torchvision==0.21.0+cu124"])
    commands([python, "-m", "pip", "check"])
    if args.backbone in {"solarwm33b", "longlive1.3b"}:
        # The original SolarWM and LongLive VAE writers both need the FFmpeg MP4 backend.
        # A Python import and pip check alone cannot detect a missing writer.
        commands([python, "-B", "-c", "import imageio,numpy as np,tempfile; "
                  "from pathlib import Path; "
                  "folder=tempfile.TemporaryDirectory(); "
                  "target=str(Path(folder.name)/'video.mp4'); "
                  "writer=imageio.get_writer(target,fps=16,codec='libx264'); "
                  "writer.append_data(np.zeros((64,64,3),dtype=np.uint8)); "
                  "writer.close(); "
                  "assert Path(target).is_file() and Path(target).stat().st_size>0; "
                  "folder.cleanup()"])
    expected = json.loads((ROOT / "scripts/install_manifest.json").read_text())[
        "environments"][args.backbone]["packages"]
    probe = commands([python, "-B", "-c",
       "import importlib.metadata as m,json,sys; print(json.dumps({"
       "'python':list(sys.version_info[:2]),"
       "'versions':{n:m.version(n) for n in json.loads(sys.argv[1])}}))",
       json.dumps(["torch", "flash-attn", "setuptools", "ninja", *PORTABLE_EXTRAS[args.backbone]])],
       capture=True)
    observed = json.loads(probe.stdout)
    require(observed["python"] == [3, 10] and
            observed["versions"]["torch"] == expected["torch"] and
            observed["versions"]["flash-attn"] == expected["flash-attn"],
            "new FA2 distribution does not match backbone's original Torch/FA2 ABI pins")
    for name, version in extras.items():
        require(observed["versions"][name] == version, "portable native extra version mismatch: " + name)
    return dict(schema_version=1, python=str(python),
                python_includes=headers, cuda_home=str(Path(args.cuda_home).resolve()),
                torch_extensions_dir=str((base / "cache/torch_extensions").resolve()),
                fa2_wheel_sha256=args.fa2_sha256,
                fa2_version=expected["flash-attn"], torch_version=expected["torch"],
                portable_extras=extras, runtime_cache_paths=cache_paths_for(python),
                historical_fa3_binary_equivalence=False)


def find_receipt(cache_dir):
    import importlib.util
    spec = importlib.util.spec_from_file_location("port_b01", ROOT / "tools/portable_b01.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    p = external(cache_dir)
    receipts = list(p.glob("*/receipt.json"))
    require(len(receipts) == 1, "exactly one built B01 receipt required")
    b01_spec = p.parent / "build-spec.json"
    require(b01_spec.is_file(), "missing current-source B01 build spec")
    module.verify_receipt(json.loads(b01_spec.read_text()), receipts[0].parent)
    return b01_spec, receipts[0].parent


def gpu_uuid(index):
    require(index >= 0, "GPU index must be nonnegative")
    result = commands(["nvidia-smi", "-i", str(index), "--query-gpu=uuid",
                       "--format=csv,noheader"], capture=True)
    devices = [x.strip() for x in result.stdout.splitlines() if x.strip()]
    require(len(devices) == 1 and devices[0].startswith("GPU-"),
            "no unique physical GPU UUID found; assign a supported GPU")
    return devices[0]


def install(args):
    base, python, b01, fa3 = paths(args)
    require(Path(args.cuda_home).is_absolute(), "CUDA_HOME must be absolute")
    if args.dry_run:
        print(json.dumps(dict(status="PLAN_ONLY", backbone=args.backbone, base=str(base),
                              python=str(python), b01=str(b01), fa3=str(fa3),
                              cuda_home=args.cuda_home, asset_root=args.asset_root,
                              gated_login_is_user_owned=True,
                              fresh_install_and_gpu_generation="NOT_VALIDATED"),
                         indent=2))
        return
    require(args.asset_root, "install requires --asset-root for model asset deployment")
    require(args.dependency_root,
            "install requires --dependency-root of user-authorized SHA-locked native sources")
    require(args.build_fa3,
            "complete installation requires FA3 compilation; --no-build-fa3 cannot be marked successful")
    # Preflight the third-party binary before downloading or changing anything.
    verify_user_fa2_wheel(args.fa2_wheel, args.fa2_sha256)
    # Existing source installer returns code 3 for unknown original optional
    # dependency provenance even when required wheels were installed.
    # Do not pretend it is historical reconstruction; continue only with a
    # real installed model venv and fail on actual errors.
    installed = subprocess.run([str(ROOT / "scripts/setup_env.sh"),
        "--backbone", args.backbone, "--env-root", args.env_root,
        "--python", args.python, "--portable"], check=False)
    require(installed.returncode in (0, 3),
            "base environment installation failed; inspect setup_env.sh output")
    require(python.is_file(), "model venv missing after setup")
    headers = resolve_python_headers(python, args.python_dev_root)
    runtime_receipt = install_model_closure(args, python, base, headers)
    environment = env_for(python, headers=headers, cuda_home=args.cuda_home)
    commands([sys.executable, ROOT / "scripts/provision_sources.py",
              "--backbone", args.backbone,
              "--dependency-root", args.dependency_root])
    if args.asset_root:
        for category in ("--models", "--datasets") if args.with_dataset else ("--models",):
            argv = [ROOT / "scripts/prepare_assets.sh", "--backbone", args.backbone,
                    "--env-root", args.env_root, "--asset-root", args.asset_root, category]
            if args.authorized_gated:
                argv.append("--authorized-gated")
            commands(argv)
    if args.build_fa3:
        # Uses pinned upstream reference; does not claim historically modified
        # FA3 binary equivalence or permission to redistribute built artifacts.
        fa3.mkdir(parents=True, exist_ok=True)
        receipt = fa3 / "build-receipt.json"
        if not receipt.is_file():
            commands([python, ROOT / "scripts/build_fa3.py",
                      "--backbone", args.backbone, "--python", python,
                      "--artifact-root", fa3, "--cuda-home", args.cuda_home,
                      "--jobs", str(args.jobs), "--install"], env=environment)
        else:
            data = json.loads(receipt.read_text())
            require(data.get("status") == "BUILT_AND_INSTALLED" and data.get("backbone") == args.backbone,
                    "existing FA3 receipt does not attest completed installation")
    b01.mkdir(parents=True, exist_ok=True)
    spec = b01 / "build-spec.json"
    if not spec.exists():
        commands([ROOT / "scripts/portable_b01.sh", "prepare-fresh",
                  "--backbone", args.backbone, "--python", python,
                  "--cuda-home", args.cuda_home, "--artifact-root", b01],
                 env=environment)
    if not list((b01 / "cache").glob("*/receipt.json")):
        commands([ROOT / "scripts/portable_b01.sh", "build",
                  "--backbone", args.backbone, "--python", python,
                  "--spec", spec, "--cache-root", b01 / "cache"], env=environment)
    find_receipt(b01 / "cache")
    native_receipt = base / "native/build-env.json"
    native_receipt.parent.mkdir(parents=True, exist_ok=True)
    require(not native_receipt.is_symlink(), "refusing build environment symlink")
    with native_receipt.open("w", encoding="utf-8") as stream:
        json.dump(runtime_receipt, stream, indent=2)
        stream.write("\n")
    for folder in runtime_receipt["runtime_cache_paths"].values():
        Path(folder).mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(status="INSTALLATION_COMPONENTS_PREPARED",
                          backbone=args.backbone, python=str(python),
                          source_identity="fresh B01 + pinned upstream current-reference FA3",
                          video_generation="NOT_TESTED_BY_INSTALL_COMMAND"), indent=2))


def model_paths(args):
    root = external(args.asset_root)
    layouts = {
        "longlive1.3b": ("models/longlive/models/longlive_base.pt", "models/wan",
                         "models/longlive/models/lora.pt", None),
        "lingbot1.3b": ("models/lingbot13", "models/lingbot14", None, None),
        "lingbot14b": ("models/lingbot14", "models/lingbot14", None, None),
        "solarwm33b": ("models/solar/SolarWM-h3-33B-sgf-stage2-158f",
                       "models/solar/SolarWM-h3-33B-base", None,
                       "models/solar/support/h3_silence_153_158_170.safetensors"),
    }
    return tuple(str(root / name) if name else None for name in layouts[args.backbone])


def prepare_inputs(args):
    base, python, _, _ = paths(args)
    require(python.is_file(), "selected model Python is missing")
    version = {"longlive1.3b": "longlive", "lingbot1.3b": "lingbot13",
               "lingbot14b": "h200_large", "solarwm33b": "h200_large"}[args.backbone]
    names = {"longlive1.3b": "LongLive1.3B", "lingbot1.3b": "LingBot1.3B",
             "lingbot14b": "LingBot14B", "solarwm33b": "SolarWM33B"}
    source = ROOT / "versions" / version
    output = external(args.output)
    require(not output.exists(), "prepared inputs directory exists; refuse overwrite")
    require(not args.case_id.startswith("-") and all(x.isalnum() or x in "._-" for x in args.case_id)
            and ".." not in args.case_id, "invalid case ID")
    if args.backbone != "longlive1.3b":
        require(args.index and args.index_sha256, "ORBIA requires an authorized dataset file index and SHA")
        # Frozen source exports main(argv) but has no __main__ guard.
        # Invoking it through -m silently exits 0 without preparing a request.
        cmd = [python, "-B", "-c",
               "from draftkv.benchmarks.orbia_requests import main; "
               "import sys; raise SystemExit(main(sys.argv[1:]))",
               "--backbone", names[args.backbone], "--dataset-root", external(args.dataset_root),
               "--index", external(args.index), "--index-sha256", args.index_sha256,
               "--case-ids", args.case_id, "--output", output]
        if args.lock:
            require(external(args.lock).is_file(), "authorized original ORBIA lock missing")
            cmd += ["--lock", external(args.lock)]
        else:
            lock = source / "configs/datasets/orbia480.lock.json"
            require(lock.is_file(), "original ORBIA source lock absent; supply --lock with authorized original")
    else:
        manifest = (external(args.manifest) if args.manifest else
                    source / "configs/datasets/bench1/worldattention160.manifest.json")
        require(manifest.is_file(),
                "source-bound LongLive original manifest absent; use --manifest to supply the SHA-verified authorized source")
        cmd = [python, "-B", "-c",
               "from draftkv.backbones.bench1_inputs import main; "
               "import sys; raise SystemExit(main(sys.argv[1:]))",
               "--backbone", names[args.backbone], "--manifest", manifest,
               "--runtime-dataset", external(args.dataset_root),
               "--case-ids", args.case_id, "--output", output]
        if args.native_outputs:
            cmd += ["--native-outputs", external(args.native_outputs)]
    environment = env_for(python)
    environment["PYTHONPATH"] = str(source / "src")
    commands(cmd, env=environment)
    request = output / "cases" / args.case_id / "request.json"
    require(request.is_file(), "original input preparation did not create a case request")
    print(json.dumps(dict(status="ORIGINAL_INPUT_PREPARED",
                          request=str(request), backbone=args.backbone), indent=2))


def run(args):
    require(not args.startup_only or args.profile == "off",
            "native startup-only check does not produce an admitted latency profile")
    base, python, b01, fa3 = paths(args)
    run_dir = external(args.output).parent
    input_plan = external(args.input)
    require(input_plan.is_file(), "source-locked request/plan missing")
    require(python.is_file(), "selected model environment missing; run install first")
    require(not run_dir.exists(), "output directory already exists; choose a fresh output directory")
    spec, cache = find_receipt(b01 / "cache")
    original = json.loads(input_plan.read_text())
    is_request = "algorithm_plugin" not in original and "case_id" in original
    if is_request:
        require(args.asset_root, "--asset-root is required for source-locked request input")
        checkpoint, assets, lora, silence = model_paths(args)
        source_plan = run_dir.parent / (run_dir.name + ".source-plan.json")
        require(not source_plan.exists(), "prepared source plan exists; refuse overwrite")
        if not args.dry_run:
            cmd = [python, "-B", ROOT / "tools/prepare_input_plan.py",
                   "--backbone", args.backbone, "--request", input_plan,
                   "--checkpoint", checkpoint, "--assets-dir", assets,
                   "--python", python, "--b01-spec", spec, "--b01-cache", cache,
                   "--output-plan", source_plan, "--profile", args.profile]
            if lora:
                cmd += ["--lora", lora]
            if silence:
                cmd += ["--silence", silence]
            if args.native_source:
                cmd += ["--native-source", args.native_source]
            commands(cmd, env=env_for(python))
            input_plan = source_plan
    else:
        require("algorithm_plugin" in original and "request" in original,
                "--input must be a source-locked original request or original-schema plan")
    if args.dry_run:
        print(json.dumps(dict(status="PLAN_ONLY", backbone=args.backbone,
                              prepared_input=str(input_plan), output=args.output,
                              gpu_index=args.gpu, profile=args.profile,
                              fresh_b01_spec=str(spec),
                              worker="source-locked draftkv.backbones.worker",
                              video_generation="NOT_EXECUTED"), indent=2))
        return
    device = gpu_uuid(args.gpu)
    environment = env_for(python)
    # The independent input binder verifies the source-locked original plan,
    # all asset hashes, immutable algorithm fields and rebuilt native receipts.
    commands([sys.executable, ROOT / "tools/fresh_binding.py", "--backbone", args.backbone,
              "--input-plan", input_plan, "--spec", spec, "--cache-directory", cache,
              "--run-dir", run_dir, "--output", args.output,
              "--profile", args.profile], env=environment)
    evidence = run_dir / "deployment-evidence.json"
    argv = [ROOT / "scripts/portable_run.sh", "run", "--backbone", args.backbone,
            "--sp", "1", "--devices", device, "--python", python,
            "--plan", run_dir / "plan.json", "--portable-evidence", evidence,
            "--builtin-worker",
            "--profile", "portable-summary" if args.profile == "summary" else "off"]
    if args.startup_only:
        argv.append("--native-startup-only")
    commands(argv, env=environment)
    if args.startup_only:
        receipt = run_dir / "traces/portable_native_startup_rank0.json"
        require(receipt.is_file() and not Path(args.output).exists(),
                "native startup precompile did not complete cleanly")
        ready = json.loads(receipt.read_text())
        require(ready.get("status") == "NATIVE_EXTENSION_READY" and
                ready.get("backbone") == {"lingbot1.3b": "LingBot1.3B",
                                           "lingbot14b": "LingBot14B",
                                           "solarwm33b": "SolarWM33B",
                                           "longlive1.3b": "LongLive1.3B"}[args.backbone],
                "native precompilation attestation did not match requested backbone")
        print(json.dumps(dict(status="PORTABLE_NATIVE_STARTUP_READY",
                              output_video="NOT_GENERATED", profile_admission="NOT_REQUESTED",
                              native_receipt=str(receipt)), indent=2))
        return
    require(Path(args.output).is_file() and Path(args.output).stat().st_size > 0,
            "worker returned without a nonempty generated video")
    print(json.dumps(dict(status="VIDEO_GENERATION_COMPLETED",
                          backbone=args.backbone, output=args.output,
                          profile=str(run_dir / "portable_profile_summary.json")
                          if args.profile == "summary" else None), indent=2))


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="list supported backbone selections")
    for action in ("install", "run"):
        c = sub.add_parser(action)
        c.add_argument("--backbone", choices=BACKBONES, required=True)
        c.add_argument("--env-root", default=str(Path.home() / ".cache/draftkv-install"))
        c.add_argument("--dry-run", action="store_true")
        if action == "install":
            c.add_argument("--asset-root")
            c.add_argument("--dependency-root", help="authorized source tree for SHA-verified native files")
            c.add_argument("--python", default="python3.10", help="base Python 3.10")
            c.add_argument("--cuda-home", required=True)
            c.add_argument("--with-dataset", action="store_true")
            c.add_argument("--authorized-gated", action="store_true",
                           help="user has independently obtained upstream approval")
            c.add_argument("--build-fa3", action=argparse.BooleanOptionalAction, default=True)
            c.add_argument("--jobs", type=int, default=4)
            c.add_argument("--fa2-wheel", help="user-authorized, ABI-matching flash_attn 2.x wheel")
            c.add_argument("--fa2-sha256", help="lowercase SHA-256 of supplied FA2 wheel")
            c.add_argument("--python-dev-root", help="matching CPython 3.10 include root (Python.h and pyconfig.h)")
        else:
            c.add_argument("--asset-root", help="download root for request->plan preparation")
            c.add_argument("--native-source", help="authorized original SolarWM native source")
            c.add_argument("--input", required=True,
                           help="audited prepared original-schema input plan (not a raw image)")
            c.add_argument("--output", required=True)
            c.add_argument("--gpu", type=int, default=0)
            c.add_argument("--profile", choices=("off", "summary"), default="off")
            c.add_argument("--startup-only", action="store_true",
                           help="compile/load original native extension and exit before model generation")
    prep = sub.add_parser("prepare", help="prepare original authorized benchmark inputs")
    prep.add_argument("--backbone", choices=BACKBONES, required=True)
    prep.add_argument("--env-root", default=str(Path.home() / ".cache/draftkv-install"))
    prep.add_argument("--dataset-root", required=True)
    prep.add_argument("--case-id", required=True)
    prep.add_argument("--index", help="ORBIA original installed-files.json")
    prep.add_argument("--lock", help="authorized original ORBIA distribution lock when absent in source tree")
    prep.add_argument("--index-sha256", help="ORBIA original index SHA256")
    prep.add_argument("--manifest", help="authorized original LongLive WorldAttention manifest")
    prep.add_argument("--native-outputs", help="verified original LongLive native output policy")
    prep.add_argument("--output", required=True, help="new external directory for prepared requests")
    return p


def main(argv=None):
    p = make_parser()
    args = p.parse_args(argv)
    try:
        if args.action == "list":
            print("\n".join(BACKBONES))
        elif args.action == "install":
            install(args)
        elif args.action == "prepare":
            prepare_inputs(args)
        else:
            run(args)
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as exc:
        p.exit(2, "BLOCKED: " + str(exc) + "\n")


if __name__ == "__main__":
    sys.exit(main())
