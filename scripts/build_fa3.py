#!/usr/bin/env python3
"""Build an auditable current-reference Hopper FA3 wheel from pinned upstream archives.

Not an assertion of an unrecovered historical modified FA3 binary identity.
No GPU computation is performed while compiling. Never alters frozen DraftKV sources.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from urllib import request

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "flash-attention": {
        "commit": "060c9188beec3a8b62b33a3bfa6d5d2d44975fab",
        "url": "https://codeload.github.com/Dao-AILab/flash-attention/legacy.tar.gz/060c9188beec3a8b62b33a3bfa6d5d2d44975fab",
        "sha256": "7b524b210f3f3e391936540def73047151214fb21dd4cbc045fb4c21a363400c",
    },
    "cutlass": {
        "commit": "dc4817921edda44a549197ff3a9dcf5df0636e7b",
        "url": "https://codeload.github.com/NVIDIA/cutlass/legacy.tar.gz/dc4817921edda44a549197ff3a9dcf5df0636e7b",
        "sha256": "16fd94cd54a6755ee0ca591ad3b7966cc8a3e71fa3a59c655d7e67d37a6d6e48",
    },
}
BACKBONES = ("longlive1.3b", "lingbot1.3b", "lingbot14b", "solarwm33b")


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def outside(path):
    p = Path(path)
    require(p.is_absolute() and not p.resolve().is_relative_to(ROOT),
            "build and asset roots must be absolute and outside repository")
    return p.resolve()


def json_save(path, obj):
    path = Path(path)
    require(not path.exists(), "receipt already exists; refuse overwrite: " + str(path))
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def obtain(info, target, local=None):
    if local:
        src = Path(local).resolve()
        require(src.is_file() and digest(src) == info["sha256"], "local source archive hash mismatch")
        return src
    if target.exists():
        require(digest(target) == info["sha256"], "cached archive SHA mismatch; not overwriting")
        return target
    partial = Path(str(target) + ".part")
    if partial.exists():
        partial.unlink()  # previous incomplete content must not silently be trusted
    # urllib follows HTTPS GitHub redirects. The downloaded archive is not
    # extracted or executed until the pinned SHA-256 check succeeds.
    with request.urlopen(info["url"], timeout=120) as source, partial.open("xb") as output:
        shutil.copyfileobj(source, output)
    require(digest(partial) == info["sha256"], "download archive SHA mismatch")
    partial.replace(target)
    return target


def safe_extract(archive_path, destination):
    """Reject links, devices and traversal. Archives have exactly one top-level folder."""
    destination = Path(destination)
    require(not destination.exists(), "extraction target already exists")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        require(bool(members) and len(members) < 20000, "invalid source archive")
        roots = set()
        for member in members:
            parts = Path(member.name).parts
            require(parts and not Path(member.name).is_absolute()
                    and ".." not in parts and "\\" not in member.name,
                    "archive contains unsafe path")
            require(member.isdir() or member.isfile(),
                    "archive contains a link or nonregular member")
            roots.add(parts[0])
        require(len(roots) == 1, "archive must have exactly one top-level root")
        destination.mkdir(parents=True)
        top = next(iter(roots))
        for member in members:
            relative = Path(*Path(member.name).parts[1:])
            if not relative.parts:
                continue
            target = destination / relative
            require(target.resolve().is_relative_to(destination.resolve()),
                    "archive extraction escaped destination")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)


def probe_env(python):
    code = """import json,torch,sys
print(json.dumps(dict(python=sys.version.split()[0],torch=torch.__version__,
cuda=torch.version.cuda,cxx_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI))))"""
    completed = subprocess.run([python, "-B", "-c", code], text=True,
                               capture_output=True, check=True)
    return json.loads(completed.stdout)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", "--backbone", dest="backbone", choices=BACKBONES, required=True)
    p.add_argument("--python", required=True, help="isolated selected model Python")
    p.add_argument("--artifact-root", required=True, help="absolute external build directory")
    p.add_argument("--cuda-home", required=True)
    p.add_argument("--source-archive", help="existing official FlashAttention source tar.gz")
    p.add_argument("--cutlass-archive", help="existing official CUTLASS tar.gz")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--install", action="store_true", help="explicitly install newly built wheel")
    a = p.parse_args(argv)
    try:
        root = outside(a.artifact_root)
        require(a.jobs in range(1, 17), "--jobs must be 1..16")
        cuda = Path(a.cuda_home)
        require(cuda.is_absolute(), "CUDA_HOME must be absolute")
        manifest = json.loads((ROOT / "scripts/install_manifest.json").read_text())
        version = manifest["environments"][a.backbone]["packages"]["torch"]
        evidence = dict(backbone=a.backbone, original_torch_pin=version, sources=SOURCES,
                        cuda_home=str(cuda), artifact_root=str(root),
                        scope="CURRENT_REFERENCE_UPSTREAM_SOURCE_NOT_HISTORICAL_BINARY")
        if a.dry_run:
            print(json.dumps(dict(status="DRY_RUN_NO_INSTALL_OR_BUILD", **evidence), indent=2))
            return 0
        require(Path(a.python).is_file(), "model Python interpreter missing")
        require((cuda / "bin/nvcc").is_file(), "CUDA nvcc executable missing")
        require(not (root / "build-receipt.json").exists(), "previous completed build already exists")
        info = probe_env(a.python)
        require(info["torch"] == version, "Torch differs from pinned backbone environment")
        require(info["cuda"] == manifest["environments"][a.backbone]["torch_cuda"],
                "Torch CUDA runtime differs from backbone manifest")
        evidence["runtime"] = info
        root.mkdir(parents=True, exist_ok=True)
        src = obtain(SOURCES["flash-attention"], root / "source.tar.gz", a.source_archive)
        cutlass = obtain(SOURCES["cutlass"], root / "cutlass.tar.gz", a.cutlass_archive)
        evidence["source_tar_sha256"] = digest(src)
        evidence["cutlass_tar_sha256"] = digest(cutlass)
        source = root / "source"
        safe_extract(src, source)
        cutlass_root = source / "csrc/cutlass"
        require(not cutlass_root.exists() or not any(cutlass_root.iterdir()),
                "CUTLASS extraction target already has files")
        if cutlass_root.exists():
            cutlass_root.rmdir()
        safe_extract(cutlass, cutlass_root)
        require((cutlass_root / "include/cutlass/cutlass.h").is_file(),
                "pinned CUTLASS headers missing")
        require((source / "hopper/setup.py").is_file(),
                "pinned Hopper setup.py missing")
        wheel_dir = root / "wheels"
        wheel_dir.mkdir(exist_ok=True)
        tmp = root / "tmp"
        tmp.mkdir(exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CUDA_HOME=str(cuda),
                   TORCH_CUDA_ARCH_LIST="9.0a", MAX_JOBS=str(a.jobs),
                   NVCC_THREADS="2", FLASH_ATTENTION_FORCE_BUILD="TRUE",
                   FLASH_ATTENTION_DISABLE_SM80="TRUE",
                   FLASH_ATTENTION_HOME=str(root / "compiler_cache"), TMPDIR=str(tmp))
        env["PATH"] = str(Path(a.python).parent) + os.pathsep + env.get("PATH", "")
        before = dict(info)
        log = root / "build.log"
        with log.open("x") as stream:
            result = subprocess.run([str(Path(a.python).absolute()), "-B", "setup.py",
                                     "bdist_wheel", "--dist-dir", str(wheel_dir)],
                                    cwd=source / "hopper", env=env, stdout=stream,
                                    stderr=subprocess.STDOUT)
        require(result.returncode == 0, "FA3 build failed; see " + str(log))
        wheels = list(wheel_dir.glob("flash_attn_3-*.whl"))
        require(len(wheels) == 1, "expected exactly one FlashAttention3 wheel")
        wheel = wheels[0]
        evidence["wheel"] = dict(path=str(wheel), sha256=digest(wheel),
                                 bytes=wheel.stat().st_size)
        require(probe_env(a.python) == before,
                "build unexpectedly changed protected Torch/Python runtime")
        if a.install:
            subprocess.run([str(Path(a.python).absolute()), "-B", "-m", "pip",
                            "install", "--no-deps", "--force-reinstall", str(wheel)],
                           check=True, env=env)
            code = """import json,torch,importlib.metadata as m,flash_attn_interface as fa3,flash_attn_3._C as ext
print(json.dumps(dict(version=m.version('flash-attn-3'),interface=fa3.__file__,
native=ext.__file__,cuda_initialized=torch.cuda.is_initialized())))"""
            completed = subprocess.run([str(Path(a.python).absolute()), "-B", "-c", code],
                                       env=env, capture_output=True, text=True, check=True)
            evidence["import_check"] = json.loads(completed.stdout)
            require(evidence["import_check"]["cuda_initialized"] is False,
                    "import validation unexpectedly initialized CUDA")
        json_save(root / "build-receipt.json", dict(status="BUILT_AND_INSTALLED" if a.install
                  else "BUILT_NOT_INSTALLED", **evidence))
        print(json.dumps(dict(status="PASS", receipt=str(root / "build-receipt.json"),
                              wheel_sha256=evidence["wheel"]["sha256"],
                              installed=a.install), indent=2))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError, tarfile.TarError) as exc:
        p.exit(2, "BLOCKED: " + str(exc) + "\n")


if __name__ == "__main__":
    sys.exit(main())
