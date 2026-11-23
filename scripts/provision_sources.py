#!/usr/bin/env python3
"""Install source-locked native dependencies using a compact per-backbone plan.

Only URLs tied to a pinned public upstream commit and SHA-256 are fetched.
No custom DraftKV sources or B01 evidence archives are synthesized or replaced.
The original launcher remains compatible with the frozen 500-row SHA inventory.
"""
import argparse
import hashlib
import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen

_DEPENDENCY_SPEC = importlib.util.spec_from_file_location(
    "_draftkv_native_sources", Path(__file__).with_name("native_sources.py"))
native_sources = importlib.util.module_from_spec(_DEPENDENCY_SPEC)
_DEPENDENCY_SPEC.loader.exec_module(native_sources)

REPO = Path(__file__).resolve().parents[1]
VERSIONS = native_sources.VERSIONS


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return native_sources.digest(path)


def outside(repo, path):
    p = Path(path).expanduser().resolve()
    require(not p.is_relative_to(Path(repo).resolve()),
            "native-source and cache roots must be outside the source checkout")
    return p


def entries(repo, version, backbone=None, profile="inference"):
    return [(str(Path("versions") / version / item["path"]), item["sha256"])
            for item in native_sources.selected(repo, version, backbone, profile)]


def _source_for(repo, version, item, source):
    dest_name = str(Path("versions") / version / item["path"])
    dest = Path(repo) / dest_name
    local = source / Path(version) / item["path"] if source is not None else None
    expected = item["sha256"]
    if dest.exists() or dest.is_symlink():
        require(dest.is_file() and not dest.is_symlink() and
                dest.resolve().is_relative_to(Path(repo).resolve()) and
                digest(dest) == expected,
                "existing native source conflicts with SHA lock: " + dest_name)
        return "installed", dest
    if local is not None and local.exists():
        require(local.is_file() and local.resolve().is_relative_to(source) and
                digest(local) == expected,
                "authorized native source missing/incorrect: " + dest_name)
        return "user_source", local
    if item.get("online") is not None:
        return "official_online", None
    return "unavailable", None


def inspect(repo, version, source=None, backbone=None, profile="inference",
            cache_root=None):
    repo = Path(repo).resolve()
    source = outside(repo, source) if source is not None else None
    cache_root = outside(repo, cache_root or Path.home() / ".cache/draftkv/native-sources")
    selected = native_sources.selected(repo, version, backbone, profile)
    plan = []
    for item in selected:
        status, local = _source_for(repo, version, item, source)
        # A SHA checked source cache may be reused without touching the internet.
        if status == "official_online":
            cache_entry = cache_root / item["sha256"]
            if cache_entry.exists():
                require(cache_entry.is_file() and not cache_entry.is_symlink() and
                        digest(cache_entry) == item["sha256"] and
                        cache_entry.stat().st_size == item["online"]["bytes"],
                        "cached upstream source SHA-256/size mismatch: " + item["path"])
                status, local = "verified_cache", cache_entry
        plan.append((item, status, local))
    return plan


def _download(online, cache_root):
    # Exact public raw GitHub path was verified byte-for-byte while constructing
    # native_sources.json; verify the locked digest again after every download.
    url = online["url"]
    parsed = urlsplit(url)
    require(parsed.scheme == "https" and parsed.netloc == "raw.githubusercontent.com" and
            parsed.username is None and parsed.password is None and not parsed.query and
            not parsed.fragment,
            "untrusted native-source download host")
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / online["sha256"]
    require(not target.is_symlink(), "native-source cache entry is a symlink")
    if target.exists():
        require(target.is_file() and
                target.stat().st_size == online["bytes"] and
                digest(target) == online["sha256"],
                "cached source content differs from verified upstream SHA")
        return target
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=cache_root,
                                         prefix=".native-", suffix=".part",
                                         delete=False) as output:
            temporary = Path(output.name)
            checksum = hashlib.sha256()
            size = 0
            with urlopen(url, timeout=60) as network:
                while True:
                    chunk = network.read(1024 * 1024)
                    if not chunk:
                        break
                    checksum.update(chunk)
                    size += len(chunk)
                    require(size <= online["bytes"], "upstream source exceeds pinned size")
                    output.write(chunk)
        require(size == online["bytes"] and checksum.hexdigest() == online["sha256"],
                "upstream native-source download SHA-256/size mismatch")
        require(not target.exists(), "native-source cache entry appeared during download")
        temporary.replace(target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def provision(repo, version, source=None, dry_run=False, *, backbone=None,
              profile="inference", cache_root=None, require_available=False,
              allow_partial=False):
    repo = Path(repo).resolve()
    require((repo / ".git").exists(),
            "a Git clone with private .git/info/exclude is required")
    source = outside(repo, source) if source is not None else None
    cache = outside(repo, cache_root or Path.home() / ".cache/draftkv/native-sources")
    plan = inspect(repo, version, source=source, backbone=backbone,
                   profile=profile, cache_root=cache)
    states = {}
    for _, status, _ in plan:
        states[status] = states.get(status, 0) + 1
    missing = [str(Path("versions") / version / item["path"])
               for item, status, _ in plan if status == "unavailable"]
    summary = {
        "status": ("NEEDS_ORIGINAL_SOURCES" if missing else
                   "VERIFIED_ONLY_NO_MUTATION" if dry_run else
                   "READY_FOR_INSTALL"),
        "profile": profile, "backbone": backbone or version,
        "count": len(plan), "already_present": states.get("installed", 0),
        "verified_local": states.get("user_source", 0) + states.get("verified_cache", 0),
        "official_online": states.get("official_online", 0),
        "unavailable_count": len(missing),
        "unavailable_examples": missing[:10],
    }
    if dry_run:
        if require_available and missing:
            raise ValueError("unavailable native-source dependencies (" +
                             str(len(missing)) + "): " + ", ".join(missing[:6]))
        return summary
    if not allow_partial:
        require(not missing,
                "unavailable native-source dependencies (" + str(len(missing)) +
                "): " + ", ".join(missing[:6]) +
                "; inspect with --dry-run or provide original authorized --dependency-root")
    tracked = set(subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "-z"]).decode().split("\0"))
    actions = []
    for item, status, local in plan:
        name = str(Path("versions") / version / item["path"])
        dest = repo / name
        if status in ("installed", "unavailable"):
            continue
        require(name not in tracked and not dest.exists() and not dest.is_symlink(),
                "native source would overwrite a tracked/frozen file: " + name)
        if status == "official_online":
            local = _download(item["online"], cache)
        require(local is not None and local.is_file() and digest(local) == item["sha256"],
                "verified source unavailable or modified: " + name)
        actions.append((name, local, dest, item["sha256"]))
    excluded = Path(subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--git-path", "info/exclude"],
        text=True).strip())
    if not excluded.is_absolute():
        excluded = repo / excluded
    excluded.parent.mkdir(parents=True, exist_ok=True)
    existing = excluded.read_text() if excluded.is_file() else ""
    for name, src, dest, expected in actions:
        dest.parent.mkdir(parents=True, exist_ok=True)
        require(not dest.parent.is_symlink() and dest.parent.resolve().is_relative_to(repo),
                "native-source destination escapes checkout")
        tmp = dest.with_name(dest.name + ".draftkv-incomplete")
        require(not tmp.exists(), "incomplete native source from earlier attempt: " + name)
        shutil.copyfile(src, tmp)
        require(digest(tmp) == expected, "copied source SHA-256 mismatch: " + name)
        tmp.replace(dest)
    additions = sorted("/" + name for name, _, dest, _ in actions
                       if dest.exists() and name not in tracked and
                       "/" + name not in existing.splitlines())
    if additions:
        with excluded.open("a") as out:
            out.write("\n# Local verified DraftKV native dependencies (not distributed)\n" +
                      "\n".join(additions) + "\n")
    require(not subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True),
        "Git checkout became dirty after private dependency provisioning")
    return {**summary,
            "status": "PARTIAL_PUBLIC_NATIVE_SOURCES_DOWNLOADED"
                      if missing else "AUTHORIZED_NATIVE_SOURCES_READY",
            "source_hashes_verified": len(plan) - len(missing),
            "git_clean": True,
            "installed_now": len(actions)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=VERSIONS, required=True)
    parser.add_argument("--dependency-root", help="optional local original source tree")
    parser.add_argument("--cache-root", help="external verified public upstream source cache")
    parser.add_argument("--profile", choices=("inference", "full"), default="inference",
                        help="select native inference source lock or full archival closure")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-available", action="store_true",
                        help="preflight: fail early if source-locked files are unavailable")
    parser.add_argument("--download-available", action="store_true",
                        help="install exact publicly downloadable originals; report other originals as missing")
    args = parser.parse_args(argv)
    try:
        result = provision(REPO, VERSIONS[args.backbone], args.dependency_root,
                           dry_run=args.dry_run, backbone=args.backbone,
                           profile=args.profile, cache_root=args.cache_root,
                           require_available=args.require_available,
                           allow_partial=args.download_available)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, KeyError, TypeError, ValueError,
            subprocess.CalledProcessError) as error:
        parser.exit(2, "BLOCKED: " + str(error) + "\n")


if __name__ == "__main__":
    sys.exit(main())
