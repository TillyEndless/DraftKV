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
import platform
import tempfile
import urllib.request
import importlib.util
import shutil
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
# Load helpers by path: offline tests also import this module directly by file spec.
_ENV_SPEC = importlib.util.spec_from_file_location(
    "_draftkv_env_backend_" + "draftkv_py", ROOT / "scripts/env_backend.py")
env_backend = importlib.util.module_from_spec(_ENV_SPEC)
_ENV_SPEC.loader.exec_module(env_backend)
_GPU_SPEC = importlib.util.spec_from_file_location("_draftkv_gpu_backend", ROOT / "scripts/gpu_backend.py")
gpu_backend = importlib.util.module_from_spec(_GPU_SPEC)
_GPU_SPEC.loader.exec_module(gpu_backend)
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
    mode = env_backend.resolve_mode(getattr(args, "env_mode", "auto"))
    python = (env_backend.conda_python(getattr(args, "python", None))
              if mode == "conda" else base / "venv/bin/python")
    env_backend.binding(base, mode, python, args.backbone)
    return base, python, base / "native/b01", base / "native/fa3"


def cache_paths_for(python, base=None):
    """Use the per-Backbone cache, independent of the Python installation root."""
    base = Path(base) if base is not None else Path(python).absolute().parent.parent.parent
    root = base.resolve()
    paths = {name: base / suffix for name, suffix in PORTABLE_CACHE_LAYOUT.items()}
    require(all(path.resolve().is_relative_to(root) for path in paths.values()),
            "portable runtime cache escapes model environment via symlink")
    return {name: str(path) for name, path in paths.items()}


def env_for(python, *, base=None, headers=None, cuda_home=None):
    """Carry the audited build prerequisites into runtime torch JIT kernels."""
    env = dict(os.environ)
    py = Path(python)
    env["PATH"] = str(py.parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_CONFIG_FILE"] = os.devnull
    env["PIP_EXTRA_INDEX_URL"] = ""
    if base is None:
        # Legacy venv callers may infer the base. Native Conda calls pass it explicitly.
        base = py.absolute().parent.parent.parent
    caches = cache_paths_for(python, base)
    env.update(caches)
    receipt = Path(base) / "native/build-env.json"
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


def pinned_official_fa2_wheel(backbone):
    """Read an official URL and immutable wheel SHA from the existing install manifest."""
    manifest = json.loads((ROOT / "scripts/install_manifest.json").read_text())
    environment = manifest["environments"][backbone]["packages"]
    selected = manifest["fa2_official_wheels"].get(environment["torch"])
    require(selected is not None, "no pinned official FA2 wheel for this Torch environment; "
            "supply --fa2-wheel and --fa2-sha256 for a separately authorized build")
    require(selected["version"] == environment["flash-attn"],
            "official FA2 package version differs from source environment lock")
    name = selected["filename"]
    require(re.fullmatch(r"flash_attn-[A-Za-z0-9.+_-]+-cp310-cp310-linux_x86_64\.whl", name),
            "pinned FA2 wheel must target Linux x86_64 CPython 3.10")
    require(selected["url"] ==
            "https://github.com/Dao-AILab/flash-attention/releases/download/v"
            + selected["version"] + "/" + name.replace("+", "%2B"),
            "pinned official FA2 URL does not match the upstream release")
    require("cxx11abiTRUE" in name or "cxx11abiFALSE" in name,
            "pinned official FA2 wheel C++ ABI tag is missing")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", selected["sha256"])) and
            isinstance(selected["bytes"], int) and selected["bytes"] > 0,
            "pinned FA2 SHA-256 or byte count is invalid")
    return selected


def verify_official_fa2_abi(python, backbone):
    """Prevent installing a prebuilt FA2 binary into an incompatible runtime."""
    manifest = json.loads((ROOT / "scripts/install_manifest.json").read_text())
    torch_pin = manifest["environments"][backbone]["packages"]["torch"]
    require(platform.system() == "Linux" and platform.machine() == "x86_64",
            "official FA2 auto-download requires Linux x86_64")
    asset = pinned_official_fa2_wheel(backbone)
    expected_cxx11abi = "cxx11abiTRUE" in asset["filename"]
    result = commands([python, "-B", "-c",
        "import json,sys,torch; print(json.dumps({"
        "'python':list(sys.version_info[:2]),"
        "'torch':torch.__version__,"
        "'cuda':torch.version.cuda,"
        "'cxx11abi':bool(torch._C._GLIBCXX_USE_CXX11_ABI)}))"], capture=True)
    facts = json.loads(result.stdout)
    require(facts.get("python") == [3, 10] and
            facts.get("torch") == torch_pin and
            isinstance(facts.get("cuda"), str) and facts["cuda"].startswith("12.") and
            facts.get("cxx11abi") is expected_cxx11abi,
            "official FA2 wheel requires Python 3.10, the locked Torch/CUDA 12 "
            "runtime and a matching cxx11abi" + str(expected_cxx11abi).upper())


def obtain_fa2_wheel(args, python, base):
    """Use an explicitly supplied checked wheel, or download an official SHA-locked one."""
    if args.fa2_wheel is not None or args.fa2_sha256 is not None:
        wheel = verify_user_fa2_wheel(args.fa2_wheel, args.fa2_sha256)
        return wheel, args.fa2_sha256
    asset = pinned_official_fa2_wheel(args.backbone)
    verify_official_fa2_abi(python, args.backbone)
    directory = base / "cache/verified_wheels"
    require(directory.resolve().is_relative_to(base.resolve()) and
            not directory.is_symlink(), "FA2 wheel cache escapes model environment")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / asset["filename"]
    require(not destination.is_symlink(), "refusing symlinked FA2 wheel cache entry")
    if destination.exists():
        require(destination.is_file() and destination.stat().st_size == asset["bytes"] and
                sha256(destination) == asset["sha256"],
                "cached official FA2 wheel SHA-256 mismatch; refusing to use it")
        return destination, asset["sha256"]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".fa2-", suffix=".part",
                                         delete=False) as out:
            temporary = Path(out.name)
            digest = hashlib.sha256()
            count = 0
            print("Downloading SHA-256-pinned FlashAttention-2 from official GitHub release:", asset["url"],
                  flush=True)
            with urllib.request.urlopen(asset["url"], timeout=45) as stream:
                while True:
                    block = stream.read(8 * 1024 * 1024)
                    if not block:
                        break
                    count += len(block)
                    require(count <= asset["bytes"], "official FA2 wheel exceeds pinned size")
                    digest.update(block)
                    out.write(block)
        require(count == asset["bytes"] and digest.hexdigest() == asset["sha256"],
                "official FA2 wheel SHA-256/size mismatch; refusing installation")
        require(not destination.exists(), "FA2 wheel cache target appeared during download")
        temporary.replace(destination)
        print("Verified official FlashAttention-2 wheel SHA-256:", digest.hexdigest(), flush=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination, asset["sha256"]


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
    """Install a verified official or user-supplied FA2 wheel and portable extras."""
    wheel, wheel_sha256 = obtain_fa2_wheel(args, python, base)
    flags = env_backend.pip_isolated_flags(env_backend.resolve_mode(getattr(args, "env_mode", "auto")))
    pip = [python, "-m", "pip", *flags, "install", "--no-deps"]
    commands([*pip, "--force-reinstall", wheel])
    extras = {"setuptools": "84.0.0", "ninja": "1.13.2", **PORTABLE_EXTRAS[args.backbone]}
    commands([*pip, "--only-binary=:all:", "--index-url", "https://pypi.org/simple",
              *(name + "==" + version for name, version in extras.items())])
    if args.backbone in TORCHVISION_CU124:
        commands([*pip, "--only-binary=:all:", "--index-url",
                  env_backend.torch_index_url(getattr(args, "torch_index_url", None), "12.4"),
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
            observed["versions"]["flash-attn"].split("+", 1)[0] == expected["flash-attn"],
            "new FA2 distribution does not match backbone's original Torch/FA2 ABI pins")
    for name, version in extras.items():
        require(observed["versions"][name] == version, "portable native extra version mismatch: " + name)
    return dict(schema_version=1, python=str(python),
                python_includes=headers, cuda_home=str(Path(args.cuda_home).resolve()),
                torch_extensions_dir=str((base / "cache/torch_extensions").resolve()),
                fa2_wheel_sha256=wheel_sha256,
                fa2_version=expected["flash-attn"], torch_version=expected["torch"],
                portable_extras=extras, runtime_cache_paths=cache_paths_for(python, base),
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


def inspect_native_dependencies(args):
    """Resolve/download only the original source-locked selection for one model."""
    command = [sys.executable, ROOT / "scripts/provision_sources.py",
               "--backbone", args.backbone, "--profile", args.profile]
    if args.dependency_root:
        command += ["--dependency-root", args.dependency_root]
    if args.dependency_cache:
        command += ["--cache-root", args.dependency_cache]
    if args.dry_run:
        command.append("--dry-run")
    commands(command)


def download_assets(args):
    """Download all pinned online model assets using the caller's existing HF login."""
    root = external(args.asset_root)
    # The low-level asset helper uses mutually exclusive category flags.
    # Download categories individually and skip original inputs for which
    # neither an upstream URL nor a verifiable content lock exists.
    categories = ["--models", *(["--datasets"] if args.with_dataset else [])]
    for category in categories:
        cmd = [ROOT / "scripts/prepare_assets.sh", "--backbone", args.backbone,
               "--asset-root", str(root), category, "--online-only"]
        if args.dry_run:
            cmd.append("--dry-run")
        # HF_TOKEN and standard Hugging Face login are used automatically.
        commands(cmd)
    print(json.dumps(dict(
        status="ASSET_DOWNLOAD_PLAN_ONLY" if args.dry_run else "PINNED_ONLINE_ASSETS_READY",
        backbone=args.backbone,
        asset_root=str(root),
        categories=["models", *(["datasets"] if args.with_dataset else [])],
        original_benchmark_inputs="SEPARATELY_REQUIRED",
    ), indent=2))


def _cuda_toolkit_root(explicit=None):
    """Use a caller-specified or discovered nvcc for Hopper FA3 builds."""
    candidates = [explicit, os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")]
    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(str(Path(nvcc).resolve().parents[1]))
    candidates.extend(("/usr/local/cuda", "/usr/local/cuda-12.8",
                       "/usr/local/cuda-12.6"))
    for value in candidates:
        if not value:
            continue
        root = Path(value).expanduser()
        if root.is_absolute() and (root / "bin/nvcc").is_file():
            return str(root.resolve())
    raise ValueError("Hopper FA3 source build needs an installed CUDA toolkit "
                     "with nvcc (set CUDA_HOME or --cuda-home)")


def bootstrap_attention(args):
    """Install model-native FA2; automatically build FA3 additionally on Hopper."""
    base, python, _, fa3 = paths(args)
    gpu = gpu_backend.detect(args.gpu)
    backend = gpu["backend"]
    if args.dry_run:
        return dict(selected_backend=backend, gpu=gpu,
                    fa2="PLAN_ONLY", fa3="PLAN_ONLY" if backend == "fa3" else "NOT_APPLICABLE",
                    runtime_validation="NOT_EXECUTED")
    require(python.is_file(), "model Python environment missing before attention install")
    lock = pinned_official_fa2_wheel(args.backbone)
    receipt = base / "native/attention-install.json"
    if receipt.is_file():
        previous = json.loads(receipt.read_text())
        require(previous.get("selected_backend") == backend and
                previous.get("fa2_sha256") == lock["sha256"] and
                previous.get("backbone") == args.backbone,
                "existing attention installation receipt disagrees with GPU or pinned FA2")
        if backend == "fa3":
            fa3_receipt = fa3 / "build-receipt.json"
            require(fa3_receipt.is_file() and
                    json.loads(fa3_receipt.read_text()).get("status") == "BUILT_AND_INSTALLED",
                    "existing Hopper FA3 build receipt unavailable")
        commands([python, "-m", "pip", "check"])
        return dict(selected_backend=backend, gpu=gpu,
                    fa2="REUSED_VERIFIED_INSTALL", fa3="REUSED" if backend == "fa3" else "NOT_APPLICABLE",
                    runtime_validation="NOT_EXECUTED")
    override = SimpleNamespace(backbone=args.backbone, fa2_wheel=None, fa2_sha256=None)
    wheel, fa2_sha256 = obtain_fa2_wheel(override, python, base)
    flags = env_backend.pip_isolated_flags(env_backend.resolve_mode(getattr(args, "env_mode", "auto")))
    commands([python, "-m", "pip", *flags, "install", "--no-deps", wheel])
    check = commands([python, "-B", "-c",
        "import importlib.metadata as m,json; print(json.dumps({'version':m.version('flash-attn')}))"],
        capture=True)
    require(json.loads(check.stdout)["version"].split("+", 1)[0] == lock["version"],
            "installed FA2 distribution differs from pinned upstream wheel")
    commands([python, "-m", "pip", "check"])
    if backend == "fa3":
        cuda_home = _cuda_toolkit_root(args.cuda_home)
        # The public Hopper source build needs pinned packaging/build tools;
        # do not rely on whatever happens to be on the host Python path.
        commands([python, "-m", "pip", *flags, "install",
                  "--only-binary=:all:", "--index-url", "https://pypi.org/simple",
                  "setuptools==84.0.0", "ninja==1.13.2"])
        commands([python, "-m", "pip", "check"])
        fa3.mkdir(parents=True, exist_ok=True)
        fa3_receipt = fa3 / "build-receipt.json"
        if not fa3_receipt.is_file():
            commands([python, ROOT / "scripts/build_fa3.py",
                      "--backbone", args.backbone, "--python", python,
                      "--artifact-root", fa3, "--cuda-home", cuda_home,
                      "--jobs", str(args.jobs), "--install"],
                     env=env_for(python, base=base, cuda_home=cuda_home))
        data = json.loads(fa3_receipt.read_text())
        require(data.get("status") == "BUILT_AND_INSTALLED" and
                data.get("backbone") == args.backbone,
                "Hopper FA3 build did not produce its completed install receipt")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    require(not receipt.exists() and not receipt.is_symlink(),
            "refusing to overwrite existing attention install receipt")
    receipt.write_text(json.dumps(dict(schema=1, backbone=args.backbone,
                                       selected_backend=backend,
                                       gpu=gpu, fa2_sha256=fa2_sha256,
                                       fa3_installed=backend == "fa3",
                                       runtime_validation="NOT_EXECUTED"),
                                  indent=2) + "\n")
    return dict(selected_backend=backend, gpu=gpu, fa2="INSTALLED",
                fa3="BUILT_AND_INSTALLED" if backend == "fa3" else "NOT_APPLICABLE",
                runtime_validation="NOT_EXECUTED")


def bootstrap(args):
    """Install the selected FA2/FA3 backend and available verified online assets."""
    setup_environment(args)
    if not args.dry_run:
        base, _, _, _ = paths(args)
        require((base / "portable-dependency-closure.json").is_file(),
                "Python dependency resolution is incomplete; cannot report a prepared environment")
    attention = bootstrap_attention(args)
    download_assets(args)
    # The public upstream subset is fetched without pretending that modified,
    # unredistributed B01/model sources have become available.
    cmd = [sys.executable, ROOT / "scripts/provision_sources.py",
           "--backbone", args.backbone, "--profile", "inference"]
    cmd.append("--dry-run" if args.dry_run else "--download-available")
    evidence = commands(cmd, capture=True)
    native = json.loads(evidence.stdout)
    print(json.dumps(dict(
        status="NETWORK_BOOTSTRAP_PLAN_ONLY" if args.dry_run
               else "ONLINE_ENVIRONMENT_AND_ASSETS_PREPARED",
        backbone=args.backbone,
        env_root=str(external(args.env_root)),
        asset_root=str(external(args.asset_root)),
        attention_backend=attention,
        native_sources=native["status"],
        native_sources_unavailable=native["unavailable_count"],
        native_sources_verified=native.get("source_hashes_verified", 0),
        original_input_cases="NOT_PROVISIONED",
        end_to_end_generation="NOT_EXECUTED",
    ), indent=2))


def setup_environment(args):
    """Bootstrap the pinned Python package environment from online indices."""
    base, python, _, _ = paths(args)
    if args.dry_run:
        print(json.dumps(dict(status="ENVIRONMENT_PLAN_ONLY",
                              backbone=args.backbone, python=args.python,
                              env_root=str(base)), indent=2))
        return
    result = subprocess.run([str(ROOT / "scripts/setup_env.sh"),
                             "--backbone", args.backbone, "--env-root", args.env_root,
                              "--python", str(env_backend.conda_python(getattr(args, "python", None)))
                              if env_backend.resolve_mode(getattr(args, "env_mode", "auto")) == "conda"
                              else (args.python or "python3.10"),
                              "--env-mode", env_backend.resolve_mode(getattr(args, "env_mode", "auto")),
                              "--gpu", str(getattr(args, "gpu", 0)), "--portable",
                              *(["--torch-index-url", args.torch_index_url]
                                if getattr(args, "torch_index_url", None) else [])], check=False)
    require(result.returncode in (0, 3) and python.is_file(),
            "base environment installation failed; inspect setup_env.sh report")
    env_backend.binding(base, env_backend.resolve_mode(getattr(args, "env_mode", "auto")),
                        python, args.backbone)
    # This stage does not claim FlashAttention native readiness or an E2E GPU run.
    print(json.dumps(dict(status="BASE_ENVIRONMENT_PREPARED",
                          backbone=args.backbone, python=str(python),
                          native_dependencies="NOT_INSTALLED"), indent=2))


def should_build_fa3(args, gpu):
    """Hopper SM90 receives FA3; other NVIDIA devices use FA2 only."""
    if gpu["backend"] == "fa3":
        require(args.build_fa3,
                "Hopper requires FA3 compilation; --no-build-fa3 cannot be marked successful")
        return True
    return False


def install(args):
    base, python, b01, fa3 = paths(args)
    require(Path(args.cuda_home).is_absolute(), "CUDA_HOME must be absolute")
    gpu = gpu_backend.detect(args.gpu)
    attention_backend = gpu["backend"]
    if args.dry_run:
        print(json.dumps(dict(status="PLAN_ONLY", backbone=args.backbone, base=str(base),
                              python=str(python), b01=str(b01), fa3=str(fa3),
                              cuda_home=args.cuda_home, asset_root=args.asset_root,
                              gpu=gpu, selected_attention_backend=attention_backend,
                              gated_login_is_user_owned=True,
                              fresh_install_and_gpu_generation="NOT_VALIDATED"),
                         indent=2))
        return
    require(args.asset_root, "install requires --asset-root for model asset deployment")
    build_fa3 = should_build_fa3(args, gpu)
    # Validate the pinned official source or explicit user-provided binary
    # before changing the environment. Actual download follows Torch setup.
    if args.fa2_wheel is not None or args.fa2_sha256 is not None:
        verify_user_fa2_wheel(args.fa2_wheel, args.fa2_sha256)
    else:
        pinned_official_fa2_wheel(args.backbone)
    # Refuse BEFORE the large PyTorch/model downloads if the source-locked
    # files cannot be recovered via exact upstream URLs or the optional local tree.
    native_cmd = [sys.executable, ROOT / "scripts/provision_sources.py",
                  "--backbone", args.backbone, "--profile", "inference"]
    if args.dependency_root:
        native_cmd += ["--dependency-root", args.dependency_root]
    if args.dependency_cache:
        native_cmd += ["--cache-root", args.dependency_cache]
    inspection = subprocess.run([str(x) for x in native_cmd] +
                                ["--dry-run", "--require-available"],
                                capture_output=True, text=True, check=False)
    require(inspection.returncode == 0,
            "native dependencies are not all downloadable: " +
            inspection.stderr.strip().removeprefix("BLOCKED: ") +
            "; obtain missing originals via --dependency-root")
    # Existing source installer returns code 3 for unknown original optional
    # dependency provenance even when required wheels were installed.
    # Do not pretend it is historical reconstruction; continue only with a
    # real installed model venv and fail on actual errors.
    installed = subprocess.run([str(ROOT / "scripts/setup_env.sh"),
        "--backbone", args.backbone, "--env-root", args.env_root,
        "--python", str(env_backend.conda_python(getattr(args, "python", None)))
        if env_backend.resolve_mode(getattr(args, "env_mode", "auto")) == "conda"
        else (args.python or "python3.10"),
        "--env-mode", env_backend.resolve_mode(getattr(args, "env_mode", "auto")),
        "--gpu", str(args.gpu), "--portable",
        *(["--torch-index-url", args.torch_index_url]
          if getattr(args, "torch_index_url", None) else [])], check=False)
    require(installed.returncode in (0, 3),
            "base environment installation failed; inspect setup_env.sh output")
    require(python.is_file(), "model Python missing after setup")
    headers = resolve_python_headers(python, args.python_dev_root)
    runtime_receipt = install_model_closure(args, python, base, headers)
    runtime_receipt["gpu"] = gpu
    runtime_receipt["selected_attention_backend"] = attention_backend
    environment = env_for(python, base=base, headers=headers, cuda_home=args.cuda_home)
    commands(native_cmd)
    if args.asset_root:
        for category in ("--models", "--datasets") if args.with_dataset else ("--models",):
            argv = [ROOT / "scripts/prepare_assets.sh", "--backbone", args.backbone,
                    "--env-root", args.env_root, "--asset-root", args.asset_root, category]
            # Authentication is handled by the configured Hugging Face account.
            # The assets helper auto-attempts gated downloads and reports 401/403.
            commands(argv)
    if build_fa3:
        # Hopper SM90 uses FA3; non-Hopper GPUs use the installed FA2
        # without attempting to compile Hopper-only FA3.
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
                          gpu=gpu, selected_attention_backend=attention_backend,
                          flash_attention_2="INSTALLED_FOR_MODEL_NATIVE_DEPENDENCIES",
                          flash_attention_3=("BUILT_FOR_HOPPER" if attention_backend == "fa3"
                                             else "NOT_BUILT_ON_NON_HOPPER"),
                          source_identity="fresh B01 + architecture-selected official attention backend",
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
    environment = env_for(python, base=base)
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
            commands(cmd, env=env_for(python, base=base))
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
    environment = env_for(python, base=base)
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
            c.add_argument("--dependency-root", help="optional original SHA-verified source tree; "
                           "needed only for dependencies not downloadable from exact upstream")
            c.add_argument("--dependency-cache", help="optional external cache for source-verified "
                           "public upstream files")
            c.add_argument("--python", help="Python 3.10; defaults to activated Conda Python in Conda mode")
            c.add_argument("--cuda-home", required=True)
            c.add_argument("--gpu", type=int, default=0,
                           help="physical GPU index for automatic FA3 (SM90) / FA2 selection")
            c.add_argument("--with-dataset", action="store_true")
            c.add_argument("--authorized-gated", action="store_true",
                           help="deprecated compatibility flag; HF login is used automatically")
            c.add_argument("--build-fa3", action=argparse.BooleanOptionalAction, default=True,
                           help="Hopper-only FA3 build; non-Hopper always skips FA3")
            c.add_argument("--jobs", type=int, default=4)
            c.add_argument("--fa2-wheel", help="optional offline/override ABI-matched FA2 wheel; "
                           "default: automatically download SHA-locked official release")
            c.add_argument("--fa2-sha256", help="SHA-256 required only with --fa2-wheel")
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
    native = sub.add_parser("native-deps",
                            help="inspect or provision per-backbone SHA-locked native sources")
    native.add_argument("--backbone", choices=BACKBONES, required=True)
    native.add_argument("--dependency-root", help="optional original source archive tree")
    native.add_argument("--dependency-cache", help="external verified upstream file cache")
    native.add_argument("--profile", choices=("inference", "full"), default="inference")
    native.add_argument("--dry-run", action="store_true")
    boot = sub.add_parser("bootstrap",
                          help="download pinned Python environment and available model/data assets from the network")
    boot.add_argument("--backbone", choices=BACKBONES, required=True)
    boot.add_argument("--env-root", default=str(Path.home() / ".cache/draftkv-install"))
    boot.add_argument("--asset-root", default=str(Path.home() / ".cache/draftkv/assets"))
    boot.add_argument("--python", help="Python 3.10; defaults to activated Conda Python")
    boot.add_argument("--gpu", type=int, default=0,
                      help="physical GPU index for attention backend detection")
    boot.add_argument("--cuda-home",
                      help="optional CUDA toolkit root for Hopper FA3 build; otherwise auto-detected")
    boot.add_argument("--jobs", type=int, default=4,
                      help="parallel jobs for the Hopper FA3 source build")
    boot.add_argument("--with-dataset", action="store_true",
                      help="also download verified online dataset assets; original cases may not be published")
    boot.add_argument("--dry-run", action="store_true")

    assets = sub.add_parser("assets", help="download SHA-locked model assets from Hugging Face")
    assets.add_argument("--backbone", choices=BACKBONES, required=True)
    assets.add_argument("--asset-root", required=True)
    assets.add_argument("--with-dataset", action="store_true",
                        help="also fetch pinned online dataset assets; original cases may still be unavailable")
    assets.add_argument("--dry-run", action="store_true")
    environment = sub.add_parser("environment", help="install per-backbone pinned Python packages")
    environment.add_argument("--backbone", choices=BACKBONES, required=True)
    environment.add_argument("--env-root", default=str(Path.home() / ".cache/draftkv-install"))
    environment.add_argument("--python", help="Python 3.10; defaults to activated Conda Python")
    environment.add_argument("--gpu", type=int, default=0,
                             help="physical GPU index for attention backend detection")
    environment.add_argument("--dry-run", action="store_true")
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
    for command in (boot, environment, sub.choices["install"]):
        command.add_argument("--torch-index-url",
                             help="optional official or Aliyun CUDA-specific PyTorch wheel index")
    for command in (boot, environment, prep, *(
            sub.choices[name] for name in ("install", "run"))):
        command.add_argument("--env-mode", choices=env_backend.MODES, default="auto",
                             help="auto uses activated Conda, otherwise legacy venv")
    return p


def main(argv=None):
    p = make_parser()
    args = p.parse_args(argv)
    try:
        if args.action == "list":
            print("\n".join(BACKBONES))
        elif args.action == "install":
            install(args)
        elif args.action == "native-deps":
            inspect_native_dependencies(args)
        elif args.action == "bootstrap":
            bootstrap(args)
        elif args.action == "assets":
            download_assets(args)
        elif args.action == "environment":
            setup_environment(args)
        elif args.action == "prepare":
            prepare_inputs(args)
        else:
            run(args)
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as exc:
        p.exit(2, "BLOCKED: " + str(exc) + "\n")


if __name__ == "__main__":
    sys.exit(main())
