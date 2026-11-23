#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Small, offline-verifiable helpers for the three installation shell scripts."""
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("install_manifest.json")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def external(path):
    path = Path(path).expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("environment, cache and asset paths must be outside the source checkout")
    return path


def child(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("asset paths must be relative and cannot contain '..'")
    # Reject symlinks into the checkout, including symlinked parent directories.
    return external(root / relative)


class Report:
    def __init__(self):
        self.statuses = []

    def item(self, status, label, detail=""):
        self.statuses.append(status)
        print(f"{status} {label}" + (f": {detail}" if detail else ""), flush=True)

    def finish(self, dry=False):
        state = ("FAIL" if "FAIL" in self.statuses else
                 "MANUAL_REQUIRED" if any(s in self.statuses for s in ("MANUAL_REQUIRED", "NOT_CHECKED")) else "PASS")
        print(f"{state} OVERALL (installation checks only; GPU generation/performance NOT_CHECKED)")
        return 1 if state == "FAIL" else 0 if dry or state == "PASS" else 3


def run(cmd, **kwargs):
    return subprocess.run([str(x) for x in cmd], check=True, **kwargs)


def package_key(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def selected_packages(env):
    # Historical pins win even when a supplement uses a spelling alias.
    packages = dict(env["packages"])
    names = {package_key(name) for name in packages}
    for name, version in env.get("installation_packages", {}).items():
        if package_key(name) not in names:
            packages[name] = version
            names.add(package_key(name))
    return packages


def portable_packages(env):
    """Explicit portable-only pins; original observed package pins are immutable."""
    packages = selected_packages(env)
    for name, version in env.get("portable_overrides", {}).items():
        existing = next((key for key in packages if package_key(key) == package_key(name)), None)
        packages[existing or name] = version
    return packages


def matches(path, asset):
    return (path.is_file() and (asset.get("bytes") is None or path.stat().st_size == asset["bytes"])
            and digest(path) == asset["sha256"])


def hf_token():
    # Standard Hugging Face authentication; never print or save the token.
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    token_path = Path(os.environ.get("HF_TOKEN_PATH", str(Path(os.environ.get(
        "HF_HOME", str(Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface"))) / "token")))
    return token_path.read_text().strip() if token_path.is_file() else None


def provision_file(asset, root, report, download=False, gated=False, dry=False, online_only=False):
    path = child(root, asset["path"])
    if path.exists():
        if matches(path, asset):
            report.item("PASS", asset["id"], "existing SHA-256 verified; reused")
        else:
            report.item("FAIL", asset["id"], "existing file has wrong SHA/size; preserved without overwrite")
        return
    if not download:
        report.item("FAIL", asset["id"], "missing " + asset["path"])
        return
    url = asset.get("url")
    if not url:
        report.item("MANUAL_REQUIRED", asset["id"], "provide original content at " + asset["path"])
        return
    if asset.get("gated") and not gated:
        report.item("MANUAL_REQUIRED", asset["id"], "obtain upstream access, use standard HF login, then --authorized-gated; " + asset.get("source", ""))
        return
    if dry:
        report.item("NOT_CHECKED", asset["id"], f"would download {asset.get('bytes') or 'unknown-size'} bytes to {asset['path']}; {asset.get('license', '')}")
        return
    if root != path.parent and root not in path.parents:
        raise ValueError("new downloads cannot write through a symlink outside --asset-root")
    token = hf_token() if urlparse(url).hostname == "huggingface.co" else None
    if asset.get("gated") and not token:
        report.item("MANUAL_REQUIRED", asset["id"], "standard Hugging Face login/HF_TOKEN required; no access request is submitted")
        return
    # curl resumes the .part only when the server supports Range. A Range failure
    # stays a failure; the existing target is never truncated or replaced.
    path.parent.mkdir(parents=True, exist_ok=True)
    part = child(root, asset["path"] + ".part")
    if root not in part.parents:
        raise ValueError("partial-download path escapes --asset-root through a symlink")
    # Share a lock on the partial inode; never write a published hardlink again.
    with part.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            report.item("PASS" if matches(path, asset) else "FAIL", asset["id"], "concurrent target preserved and verified")
            return
        if part.exists() and matches(part, asset):
            os.link(part, path)  # atomic no-clobber publication, including concurrent runs
            report.item("PASS", asset["id"], "reused fully verified .part")
            return
        cmd = ["curl", "--disable", "--fail", "--location", "--silent", "--show-error", "--connect-timeout", "30",
               "--proto", "=https", "--proto-redir", "=https", "--continue-at", "-", "--output", str(part), "--config", "-", url]
        config = ""
        if token:
            if any(c in token for c in ('\r', '\n', '"', '\\')):
                raise ValueError("invalid Hugging Face token format")
            config = f'header = "Authorization: Bearer {token}"\n'
        try:
            # Authorization is on stdin, not in argv or a persisted config file.
            run(cmd, input=config, text=True)
            if not matches(part, asset):
                report.item("FAIL", asset["id"], "download SHA/size mismatch; .part retained for inspection")
                return
            os.link(part, path)  # keep .part for resumption/reuse; no valid file is overwritten
            report.item("PASS", asset["id"], "download SHA-256 verified; " + asset.get("license", ""))
        except (OSError, subprocess.CalledProcessError):
            report.item("FAIL", asset["id"], "download/publication failed; existing target preserved, .part retained; gated sources may require upstream approval")


def asset_checks(data, backbone, root, categories, report, **options):
    for asset in data["assets"]:
        if backbone in asset["backbones"] and asset["category"] in categories:
            provision_file(asset, root, report, **options)
    for asset in data["manual_assets"]:
        if backbone in asset["backbones"] and asset["category"] in categories:
            if options.get("online_only"):
                report.item("SKIPPED", asset["id"],
                            "original dataset/source has no verified online URL; "
                            "not included in online-only asset bootstrap")
                continue
            path = child(root, asset["path"])
            missing = [p for p in asset["required"] if not child(path, p).exists()]
            for relative, expected in asset.get("known_files", {}).items():
                known = child(path, relative)
                status = ("PASS" if digest(known) == expected else "FAIL") if known.is_file() else (
                    "MANUAL_REQUIRED" if options.get("download") else "FAIL")
                report.item(status,
                            asset["id"] + ":" + relative, "original SHA-256 required")
            report.item("MANUAL_REQUIRED", asset["id"], asset["reason"] +
                        (" Missing required paths: " + ", ".join(missing) if missing else ""))


def source_checks(data, report, dependency_root=None, version=None, backbone=None):
    bad = []
    for path, expected in data["protected_files"].items():
        p = ROOT / path
        if not p.is_file() or digest(p) != expected:
            bad.append(path)
    records = json.loads((ROOT / "RETAINED_SOURCE_SHA256.json").read_text())
    for record in records:
        p = ROOT / "versions" / record["version"] / record["path"]
        if not p.is_file() or digest(p) != record["sha256"]:
            bad.append(str(p.relative_to(ROOT)))
    report.item("FAIL" if bad else "PASS", "original source/config/CLI identity",
                f"{len(records)} retained files and {len(data['protected_files'])} protected files; mismatches={len(bad)} " + ", ".join(bad[:3]))
    if version is None:
        return
    if (ROOT / "scripts/native_sources.json").is_file():
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("_draftkv_compact_sources", ROOT / "scripts/native_sources.py")
        source_index = module_from_spec(spec)
        spec.loader.exec_module(source_index)
        locks = source_index.selected(ROOT, version, backbone=backbone)
    else:
        locks = [r for r in json.loads((ROOT / "EXTERNAL_DEPENDENCY_SHA256.json").read_text())
                 if r["version"] == version]
    missing, wrong = [], []
    for rec in locks:
        # A prior source-verified online install is already in the local,
        # git-ignored model tree; otherwise check the optional external root.
        dest = ROOT / "versions" / version / rec["path"]
        supplied = (child(dependency_root, version + "/" + rec["path"])
                    if dependency_root is not None else None)
        path = dest if dest.is_file() else supplied
        if path is None or not path.is_file():
            missing.append(rec["path"])
        elif digest(path) != rec["sha256"]:
            wrong.append(rec["path"])
    report.item("FAIL" if wrong else "MANUAL_REQUIRED" if missing else "PASS", "external source/B01 hashes",
                f"expected={len(locks)}, missing={len(missing)}, wrong={len(wrong)}; " + ", ".join((wrong + missing)[:3]))
    report.item("NOT_CHECKED", "external source integration", "hashing an external inventory does not install files into frozen versions/, establish licenses or prove original worker lookup paths")


def install_env_vars(base):
    paths = dict(PIP_CACHE_DIR=base / "cache/pip", TMPDIR=base / "tmp",
                 TORCH_EXTENSIONS_DIR=base / "cache/torch_extensions",
                 DRAFTKV_PREPACK_BUILD=base / "cache/prepack",
                 TRITON_CACHE_DIR=base / "cache/triton",
                 TORCHINDUCTOR_CACHE_DIR=base / "cache/torchinductor",
                 CUDA_CACHE_PATH=base / "cache/cuda", HF_HOME=base / "cache/huggingface")
    if any(base not in path.resolve().parents for path in paths.values()):
        raise ValueError("venv/cache path escapes --env-root through a symlink")
    return dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                PIP_CONFIG_FILE=os.devnull, PIP_EXTRA_INDEX_URL="",
                **{key: str(path) for key, path in paths.items()})


def hardware_checks(env, report):
    system = platform.system() + " " + platform.machine()
    report.item("PASS" if system == env["platform"] else "FAIL", "platform", system + "; required " + env["platform"])
    cuda_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = str(Path(cuda_root) / "bin/nvcc") if cuda_root else "nvcc"
    for command, pattern, expected in [([nvcc, "--version"], r"release (\d+\.\d+)", env["compiler_cuda"]),
                                       ([os.environ.get("CXX", "c++"), "-dumpfullversion"], r"(\d+\.\d+)", env["compiler_gcc"])]:
        if expected is None:
            report.item("MANUAL_REQUIRED", "C++ compiler version", "exact historical LongLive compiler pin not recovered")
            continue
        try:
            out = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
            found = re.search(pattern, out)
            actual = found.group(1) if found else "unknown"
            report.item("PASS" if actual == expected else "FAIL", command[0], f"{actual}; original {expected}")
        except (OSError, subprocess.CalledProcessError):
            report.item("MANUAL_REQUIRED", command[0], f"install original toolchain {expected} externally; no system changes are made")
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version", "--format=csv,noheader"], text=True, stderr=subprocess.STDOUT)
        rows = [line.strip() for line in out.splitlines() if line.strip()]
        if not rows:
            raise ValueError("no devices")
        for row in rows:
            parts = [p.strip() for p in row.split(",")]
            report.item("PASS" if len(parts) == 4 and parts[2] == env["recorded_compute_capability"] else "MANUAL_REQUIRED",
                        "GPU architecture/driver", row + "; original profile SM90, no compatibility patch is applied")
    except (OSError, ValueError, subprocess.CalledProcessError):
        report.item("NOT_CHECKED", "GPU architecture/driver", "nvidia-smi unavailable; no generation is run")
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        version = ctypes.c_int()
        rc = driver.cuDriverGetVersion(ctypes.byref(version))
        report.item("PASS" if rc == 0 and version.value >= 12080 else "FAIL",
                    "CUDA driver API >=12.8", str(version.value))
    except (OSError, AttributeError):
        report.item("NOT_CHECKED", "CUDA driver library", "libcuda.so.1 could not be loaded; no system driver is installed by these scripts")
    # Original slot loader searches CUDA driver headers separately from nvcc.
    roots = [Path(cuda_root)] if cuda_root else []
    if shutil.which(nvcc):
        roots.append(Path(shutil.which(nvcc)).resolve().parents[1])
    roots += [Path("/usr/local/cuda"), *Path("/usr/local").glob("cuda-*")]
    headers = []
    for root in roots:
        for sub in ("include/cuda.h", "targets/x86_64-linux/include/cuda.h"):
            p = root / sub
            if p.is_file():
                found = re.search(r"^#define CUDA_VERSION\s+(\d+)", p.read_text(), re.M)
                if found and int(found.group(1)) >= 12080:
                    headers.append(str(p))
    report.item("PASS" if headers else "MANUAL_REQUIRED", "CUDA driver headers >=12.8", ", ".join(headers) or "provide original loader-visible SDK headers; nvcc and driver-header versions are separate")


def receipt_is_override(name, env, effective):
    key = package_key(name)
    overrides = {package_key(item) for item in env.get("portable_overrides", {})}
    if key not in overrides:
        return False
    active = {package_key(item): version for item, version in effective.items()}
    original = {package_key(item): version for item, version in selected_packages(env).items()}
    return active.get(key) != original.get(key)


def probe_python(python, env, report, base):
    if not Path(python).is_file():
        report.item("FAIL", "environment Python", "missing; run setup_env.sh with a Python 3.10 interpreter")
        return
    expected_packages = selected_packages(env)
    portable_verified = False
    portable_receipt = base / "portable-dependency-closure.json"
    if portable_receipt.is_file():
        try:
            receipt = json.loads(portable_receipt.read_text())
        except (OSError, json.JSONDecodeError):
            report.item("FAIL", "portable dependency receipt", "unreadable or malformed")
            return
        if (receipt.get("schema") != 1 or
                receipt.get("scope") != "CURRENT_PORTABLE_ENVIRONMENT_NOT_HISTORICAL" or
                receipt.get("top_level_pins") != selected_packages(env)):
            report.item("FAIL", "portable dependency receipt", "original package identity mismatch")
            return
        if receipt.get("portable_top_level_pins") is not None:
            if receipt["portable_top_level_pins"] != portable_packages(env):
                report.item("FAIL", "portable dependency receipt", "portable compatibility pins changed; reinstall required")
                return
            expected_packages = portable_packages(env)
            portable_verified = True
            report.item("MANUAL_REQUIRED", "portable dependency provenance",
                        "verified new-installation compatibility pins, not reconstructed historical environment")
        elif env.get("portable_overrides"):
            report.item("MANUAL_REQUIRED", "portable compatibility receipt",
                        "older venv predates recorded portable compatibility overrides; reinstall before inference")
    # No project imports or JIT loader calls; native import only loads installed artifacts.
    probe = '''import importlib,importlib.metadata as m,json,os,sys,sysconfig
def failure(e):
 detail=type(e).__name__
 message=str(e.name or "") if isinstance(e,ModuleNotFoundError) else str(e) if isinstance(e,(ImportError,OSError)) else ""
 for key,value in os.environ.items():
  if value and any(tag in key.upper() for tag in ("TOKEN","SECRET","PASSWORD","API_KEY")):
   message=message.replace(value,"<redacted>")
 message=" ".join(message.splitlines()[0].split())[:240] if message else ""
 return detail+(": "+message if message else "")
paths=sysconfig.get_paths();include=paths.get("include")
r={"python":"%s.%s"%sys.version_info[:2],"python_full":sys.version.split()[0],"versions":{},"imports":{},"site":paths["purelib"],
   "include":include,"python_header":bool(include and os.path.isfile(os.path.join(include,"Python.h")))}
for name in json.loads(sys.argv[1]):
 try:r["versions"][name]=m.version(name)
 except m.PackageNotFoundError:r["versions"][name]=None
for name in json.loads(sys.argv[2]):
 try:
  mod=importlib.import_module(name);r["imports"][name]="OK"
  if name=="torch":r.update(torch=str(mod.__version__),cuda=mod.version.cuda,abi=bool(mod._C._GLIBCXX_USE_CXX11_ABI))
 except Exception as e:r["imports"][name]=failure(e)
print(json.dumps(r))'''
    try:
        imports = [{"flash-attn": "flash_attn", "Pillow": "PIL"}.get(name, name.replace("-", "_")) for name in env["packages"]]
        supplemental = {package_key(name) for name in env.get("installation_packages", {})}
        imports += [name for name in ("torchvision", "ftfy", "scipy", "imageio") if name in supplemental and name not in imports]
        imports += [name for name in ("torchvision", "ftfy", "scipy", "imageio")
                    if any(package_key(name) == package_key(p) for p in expected_packages) and name not in imports]
        if "imageio-ffmpeg" in {package_key(item) for item in expected_packages}:
            imports += ["imageio_ffmpeg"]
        if env.get("version_tree") == "h200_large" and portable_receipt.is_file():
            imports += ["transformers.models.qwen3_vl.modeling_qwen3_vl"]
        imports += ["flash_attn_interface", "flash_attn_3._C"]
        result = subprocess.check_output([str(python), "-I", "-B", "-c", probe, json.dumps(list(expected_packages)), json.dumps(imports)],
                                         text=True, env=dict(install_env_vars(base), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"), timeout=90)
        observed = json.loads(result.splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        report.item("FAIL", "Python/native imports", "interpreter probe failed; no JIT build attempted")
        return
    report.item("PASS" if observed["python"] == env["python"] else "FAIL", "Python", observed["python"] + "; expected " + env["python"])
    report.item("NOT_CHECKED", "Python microversion", "not recorded in original profile; not guessed")
    active_python = observed.get("python_full", observed["python"])
    header = str(Path(observed["include"]) / "Python.h") if observed.get("include") else "sysconfig include path unavailable"
    report.item("PASS" if observed.get("python_header") else "MANUAL_REQUIRED", "Python.h availability",
                header + "; active Python " + active_python + ("" if observed.get("python_header") else
                "; provide matching Python " + active_python + " development headers externally before native compilation"))
    historical = {package_key(name) for name in env["packages"]}
    for name, expected in expected_packages.items():
        actual = observed.get("torch") if package_key(name) == "torch" else observed["versions"].get(name)
        provenance = ("portable override (NOT historical)"
                      if receipt_is_override(name, env, expected_packages)
                      else "original" if package_key(name) in historical
                      else "supplemental (NOT recovered historical lock)")
        report.item("PASS" if actual == expected else "FAIL", name, f"observed={actual}, {provenance}={expected}")
    report.item("PASS" if observed.get("cuda") == env["torch_cuda"] else "FAIL", "PyTorch CUDA runtime", f"observed={observed.get('cuda')}, original={env['torch_cuda']}")
    expected_abi = env["torch_cuda"] == "12.8"
    report.item("PASS" if observed.get("abi") == expected_abi else "FAIL", "Torch C++ ABI", f"observed={observed.get('abi')}, original={expected_abi}")
    for name, status in observed["imports"].items():
        report.item("PASS" if status == "OK" else "FAIL", "import " + name, status)
    site = Path(observed["site"])
    for path, h in env["site_files"].items():
        p = site / path
        identical = p.is_file() and digest(p) == h
        status = "PASS" if identical else "MANUAL_REQUIRED" if portable_verified else "FAIL"
        report.item(status, "installed source/binary " + path,
                    "historical SHA-256 matched" if identical else
                    "historical source/binary hash not recovered; portable inference requires separate GPU validation"
                    if portable_verified else "original SHA-256 required")
    if env.get("fa3_version"):
        report.item("MANUAL_REQUIRED", "historical FA3 binary identity", "version 3.0.0b1 recorded; original independent binary hash not recovered for this backbone")
    checked = subprocess.run([str(python), "-I", "-B", "-m", "pip", "check"], capture_output=True, text=True, env=install_env_vars(base))
    report.item("PASS" if checked.returncode == 0 else "FAIL", "Python dependency closure", "pip check" + ("; missing/incompatible transitive packages require original version evidence" if checked.returncode else ""))


def native_checks(env, native_root, python, report):
    if native_root is None:
        report.item("MANUAL_REQUIRED", "project native libraries", "provide --native-root with authorized original compiled route/slot .so artifacts; scripts never invoke JIT compilation")
        return
    for role, h in env["original_native_sha256"].items():
        paths = [p for p in native_root.rglob("draftkv_*" + role + "*.so") if p.is_file()]
        found = next((p for p in paths if digest(p) == h), None)
        if found is None:
            report.item("MANUAL_REQUIRED", "original native " + role, "no matching original binary SHA-256 " + h + "; a rebuilt artifact needs separate provenance")
            continue
        probe = '''import torch,importlib.util,sys
p=sys.argv[1];s=importlib.util.spec_from_file_location(p.rsplit("/",1)[-1].split(".")[0],p);s.loader.exec_module(importlib.util.module_from_spec(s))'''
        try:
            run([python, "-I", "-B", "-c", probe, found], capture_output=True, timeout=60)
            report.item("PASS", "load native " + role, "original SHA-256 and ABI import; no forward executed")
        except (OSError, subprocess.SubprocessError):
            report.item("FAIL", "load native " + role, "dynamic library/ABI import failed")


def evaluator_checks(args, data, report, base):
    if args.backbone != "longlive1.3b" or not args.evaluator_python:
        report.item("MANUAL_REQUIRED", "evaluator environment", "provide original evaluator and its separate environment; LongLive recorded preflight can be checked with --evaluator-python")
        return
    code = '''import importlib.metadata as m,json,sys
r={"python":"%s.%s.%s"%sys.version_info[:3],"packages":{}}
for name in json.loads(sys.argv[1]):
 try:r["packages"][name]=m.version(name)
 except m.PackageNotFoundError:r["packages"][name]=None
print(json.dumps(r))'''
    try:
        out = subprocess.check_output([args.evaluator_python, "-I", "-B", "-c", code,
             json.dumps(list(data["evaluator_observed_environment"]))], text=True, timeout=60,
             env=install_env_vars(base))
        observed = json.loads(out.splitlines()[-1])
        report.item("PASS" if observed["python"] == "3.12.3" else "FAIL", "evaluator Python", observed["python"] + "; preflight 3.12.3")
        for name, expected in data["evaluator_observed_environment"].items():
            actual = observed["packages"].get(name)
            report.item("PASS" if actual == expected else "FAIL", "evaluator " + name, f"observed={actual}, preflight={expected}")
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        report.item("FAIL", "evaluator environment", "interpreter/package probe failed")
    report.item("MANUAL_REQUIRED", "evaluator overlays/protocol", "the recorded environment is CPU preflight only; complete original compatibility overlay, source inventory and metric cache still required")


def setup(args, data, report, base):
    env = data["environments"][args.backbone]
    packages = portable_packages(env) if getattr(args, "portable", False) else selected_packages(env)
    source_checks(data, report)
    if getattr(args, "portable", False) and env.get("portable_overrides"):
        report.item("MANUAL_REQUIRED", "new portable compatibility pins",
                    env["portable_overrides_provenance"] + "; original package pins retained")
    if "FAIL" in report.statuses:
        return
    if env.get("installation_packages"):
        report.item("MANUAL_REQUIRED", "supplemental package provenance",
                    str(env.get("installation_packages_provenance", "current H200 reference closure/upstream Torch requirements")) + "; NOT recovered historical lock")
    if args.dry_run:
        for name, version in packages.items():
            report.item("MANUAL_REQUIRED" if name in env["manual_distributions"] else "NOT_CHECKED", name, version)
        report.item("NOT_CHECKED", "setup selection", args.backbone + " -> " + str(base / "venv") + f"; Python {env['python']}, torch CUDA {env['torch_cuda']}, toolkit {env['compiler_cuda']}")
        return
    hardware_checks(env, report)
    if "FAIL" in report.statuses:
        return
    interpreter = shutil.which(args.python or "python3.10")
    if not interpreter:
        report.item("MANUAL_REQUIRED", "Python", "provide --python /path/to/python3.10; system Python is not changed")
        return
    version = subprocess.check_output([interpreter, "-I", "-B", "-c", "import sys;print('%s.%s'%sys.version_info[:2])"], text=True).strip()
    if version != env["python"]:
        report.item("FAIL", "Python", "requires " + env["python"] + "; no alternate version installed")
        return
    if args.env_root and external(args.env_root) not in base.parents:
        raise ValueError("venv/cache path escapes --env-root through a symlink")
    venv = child(base, "venv")
    if base not in venv.parents:
        raise ValueError("venv/cache path escapes --env-root through a symlink")
    install_env_vars(base)  # Validate every cache destination before creating directories.
    if venv.exists() and not (venv / "pyvenv.cfg").is_file():
        report.item("FAIL", "venv", "existing path is not a venv; preserved")
        return
    if venv.exists() and re.search(r"include-system-site-packages\s*=\s*true", (venv / "pyvenv.cfg").read_text(), re.I):
        report.item("FAIL", "venv isolation", "existing venv includes global packages; preserved without modification")
        return
    bootstrap_pip = None
    if not venv.exists():
        try:
            subprocess.check_output([interpreter, "-I", "-B", "-m", "ensurepip", "--version"],
                                    text=True, stderr=subprocess.STDOUT)
        except (OSError, subprocess.SubprocessError):
            # Debian/Ubuntu may disable ensurepip on system Python even when a
            # trusted system/user pip supports pip --python. Install pip into
            # the brand-new isolated venv, never into system/site packages.
            # This is a deployment bootstrap, not a historical dependency pin.
            try:
                output = subprocess.check_output([interpreter, "-B", "-m", "pip", "--version"],
                                                 text=True, stderr=subprocess.STDOUT)
                found = re.match(r"pip (\d+\.\d+(?:\.\d+)?)\b", output)
                if not found or tuple(map(int, found.group(1).split("."))) < (22, 3):
                    raise ValueError("bootstrap pip must support --python (pip >=22.3)")
                bootstrap_pip = found.group(1)
            except (OSError, subprocess.SubprocessError, ValueError):
                report.item("MANUAL_REQUIRED", "Python ensurepip",
                    "Python has neither ensurepip nor a trusted pip >=22.3 with --python; "
                    "provide an isolated Python environment or install OS python3.10-venv")
                return
    for name in ("tmp", "cache/pip"):
        child(base, name).mkdir(parents=True, exist_ok=True)
    if not venv.exists():
        command = [interpreter, "-I", "-B", "-m", "venv"]
        if bootstrap_pip:
            command.append("--without-pip")
        run([*command, venv], env=install_env_vars(base))
        if bootstrap_pip:
            # Use the user's own bootstrap interpreter and pinned PyPI wheel.
            # Do not let the process select a private additional package index.
            bootstrap_env = dict(os.environ, PIP_CONFIG_FILE=os.devnull,
                                 PIP_EXTRA_INDEX_URL="", PIP_DISABLE_PIP_VERSION_CHECK="1")
            bootstrap_env.pop("PYTHONNOUSERSITE", None)
            run([interpreter, "-B", "-m", "pip", "--python", str(venv / "bin/python"),
                 "install", "--no-deps", "--only-binary=:all:", "--no-input",
                 "--index-url", "https://pypi.org/simple", "pip==" + bootstrap_pip],
                env=bootstrap_env)
            report.item("PASS", "isolated pip bootstrap",
                "venv seeded with the invoking user's trusted pip " + bootstrap_pip
                + "; no global installation; distinct from historical package identity")
    python = venv / "bin/python"
    actual = subprocess.check_output([str(python), "-I", "-B", "-c", "import sys;print('%s.%s'%sys.version_info[:2])"], text=True).strip()
    if actual != env["python"]:
        report.item("FAIL", "existing venv Python", "requires " + env["python"] + "; no packages changed")
        return
    try:
        subprocess.check_output([str(python), "-I", "-B", "-m", "pip", "--version"],
                                text=True, stderr=subprocess.STDOUT, env=install_env_vars(base))
    except (OSError, subprocess.SubprocessError):
        report.item("MANUAL_REQUIRED", "venv pip", "environment is incomplete or pip cannot run; provide a complete isolated venv externally; existing venv preserved and no packages changed")
        return
    observed = json.loads(subprocess.check_output([str(python), "-I", "-B", "-c",
        "import importlib.metadata as m,json,re;print(json.dumps({re.sub(r'[-_.]+','-',d.metadata['Name']).lower():d.version for d in m.distributions()}))"], text=True))
    for name, version in packages.items():
        if name in env["manual_distributions"]:
            report.item("MANUAL_REQUIRED", name, f"provide authorized original {version} wheel/build environment; no replacement is compiled")
            continue
        if observed.get(package_key(name)) == version:
            report.item("PASS", name, "correct installed version; skipped")
            continue
        command = [python, "-I", "-B", "-m", "pip", "--require-virtualenv", "install", "--no-deps", "--only-binary=:all:", f"{name}=={version}"]
        command += ["--index-url", "https://download.pytorch.org/whl/cu" + env["torch_cuda"].replace(".", "")] if package_key(name) in ("torch", "torchvision") else ["--index-url", "https://pypi.org/simple"]
        run(command, env=install_env_vars(base))
    if getattr(args, "portable", False):
        # A new *deployment* environment is not a historical byte-identical
        # profile. Resolve transitive requirements of the known exact direct
        # pins, then verify the direct pins did not silently change.
        pinned = [f"{name}=={version}" for name, version in packages.items()
                  if name not in env["manual_distributions"]]
        pip_args = [str(python), "-I", "-B", "-m", "pip", "--require-virtualenv",
                    "install", "--no-input", "--upgrade-strategy", "only-if-needed",
                    "--index-url", "https://pypi.org/simple", *pinned]
        run(pip_args, env=install_env_vars(base))
        subprocess.check_output([str(python), "-I", "-B", "-m", "pip", "check"],
                                text=True, stderr=subprocess.STDOUT, env=install_env_vars(base))
        final = json.loads(subprocess.check_output([str(python), "-I", "-B", "-c",
            "import importlib.metadata as m,json,re;print(json.dumps({re.sub(r'[-_.]+','-',d.metadata['Name']).lower():d.version for d in m.distributions()}))"],
            text=True, env=install_env_vars(base)))
        for name, version in packages.items():
            if name not in env["manual_distributions"] and final.get(package_key(name)) != version:
                raise ValueError("portable pip resolver changed known pinned direct package: "+name)
        receipt = child(base, "portable-dependency-closure.json")
        closure = dict(schema=1, scope="CURRENT_PORTABLE_ENVIRONMENT_NOT_HISTORICAL",
                       installed_distributions=final, top_level_pins=selected_packages(env),
                       portable_top_level_pins=packages,
                       portable_compatibility_overrides=env.get("portable_overrides", {}))
        if receipt.exists():
            if not receipt.is_file() or receipt.is_symlink():
                raise ValueError("existing portable closure receipt is not a regular file")
            earlier = json.loads(receipt.read_text())
            # Installed transitive versions may grow (e.g. FA2 installed later).
            # Retain the original immutable installation receipt, but only if
            # current package pinning, ABI identity and profile still match.
            for key in ("schema", "scope", "top_level_pins", "portable_top_level_pins",
                        "portable_compatibility_overrides"):
                if earlier.get(key) != closure[key]:
                    raise ValueError("existing portable dependency receipt conflicts with current locks")
            report.item("PASS", "portable dependency closure",
                        "reused existing receipt after pip check and current direct-pin verification")
        else:
            receipt.write_text(json.dumps(closure, sort_keys=True, indent=2)+"\n")
            report.item("PASS", "portable dependency closure",
                        "pip check passed, direct package pins preserved; recorded dependency receipt")
        report.item("MANUAL_REQUIRED", "native FlashAttention and B01",
                    "build via pinned source; checked by the subsequent native install stages")
    else:
        report.item("MANUAL_REQUIRED", "unrecorded transitive dependencies",
                    "selected exact pins installed with --no-deps; historical transitive versions unknown")
        probe_python(python, env, report, base)


def main(argv=None):
    data = json.loads(MANIFEST.read_text())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("setup", "assets", "verify"))
    p.add_argument("--backbone", required=True, choices=tuple(data["environments"]))
    p.add_argument("--env-root", default=str(Path.home() / ".cache/draftkv-install"))
    p.add_argument("--asset-root", help="external models/datasets/evaluator root (required for asset preparation)")
    p.add_argument("--python", help="setup: original Python 3.10; verify: environment interpreter")
    p.add_argument("--evaluator-python", help="verify: separate original evaluator interpreter (LongLive recorded CPU preflight only)")
    p.add_argument("--dependency-root", help="external original files laid out as <version>/<package-relative path>")
    p.add_argument("--native-root", help="external original compiled route/slot libraries (read only)")
    p.add_argument("--scope", choices=("inference", "draftkv", "benchmark"), default="benchmark",
                   help="verify: select the required readiness scope; benchmark retains legacy broad checks")
    p.add_argument("--dry-run", action="store_true", help="display selection only; no readiness claim or downloads")
    p.add_argument("--portable", action="store_true",
                   help="setup only: resolve transitive deps in an isolated fresh environment and audit final pins; not historical identity")
    p.add_argument("--online-only", action="store_true",
                   help="assets only: download available verified URLs; report non-public original case inputs as skipped")
    p.add_argument("--authorized-gated", action="store_true",
                   help="deprecated: asset downloads already use existing Hugging Face login/HF_TOKEN")
    selection = p.add_mutually_exclusive_group()
    for name in ("all", "models", "datasets", "evaluator"):
        selection.add_argument("--" + name, action="store_true")
    args = p.parse_args(argv)
    if args.action != "verify" and args.scope != "benchmark":
        p.error("--scope is valid only for verify_install.sh")
    if args.portable and args.action != "setup":
        p.error("--portable is valid only for environment setup")
    if args.online_only and args.action != "assets":
        p.error("--online-only is valid only for asset preparation")
    if args.action == "assets" and (not args.asset_root or not any(getattr(args, k) for k in ("all", "models", "datasets", "evaluator"))):
        p.error("prepare_assets requires --asset-root and --all/--models/--datasets/--evaluator")
    report = Report()
    try:
        base = child(external(args.env_root), args.backbone)
        if args.action == "setup":
            setup(args, data, report, base)
        else:
            env = data["environments"][args.backbone]
            if args.action == "verify":
                source_checks(data, report, external(args.dependency_root) if args.dependency_root else None,
                              env["version_tree"], backbone=args.backbone)
                hardware_checks(env, report)
                python = Path(args.python).expanduser() if args.python else base / "venv/bin/python"
                probe_python(python, env, report, base)
                if args.scope in ("draftkv", "benchmark"):
                    native_checks(env, external(args.native_root) if args.native_root else None, python, report)
                if args.scope == "benchmark":
                    evaluator_checks(args, data, report, base)
            if args.asset_root:
                categories = (["models"] if args.action == "verify" and args.scope != "benchmark"
                              else ["models", "datasets", "evaluator"] if args.all or args.action == "verify"
                              else [k for k in ("models", "datasets", "evaluator") if getattr(args, k)])
                asset_checks(data, args.backbone, external(args.asset_root), categories, report,
                             download=args.action == "assets",
                             gated=(args.action == "assets" or args.authorized_gated),
                             dry=args.dry_run, online_only=args.online_only)
            else:
                report.item("MANUAL_REQUIRED", "assets", "provide --asset-root; no source-tree files are downloaded")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        # Do not print command lines or authenticated subprocess input.
        report.item("FAIL", "installation operation", str(exc) if isinstance(exc, (OSError, ValueError)) else type(exc).__name__)
    return report.finish(dry=args.dry_run and args.action != "verify")


if __name__ == "__main__":
    sys.exit(main())
