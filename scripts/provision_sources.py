#!/usr/bin/env python3
"""Install already-authorized original native sources with strict per-file SHAs.

Third-party access is the user's responsibility. Originals are never
redistributed or fetched by this command. Source-locked, untracked private
files are excluded from git release staging, but retained public files cannot
be modified and are always reverified by the release launchers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
VERSIONS = {"longlive1.3b": "longlive", "lingbot1.3b": "lingbot13",
            "lingbot14b": "h200_large", "solarwm33b": "h200_large"}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def entries(repo, version):
    data = json.loads((repo / "EXTERNAL_DEPENDENCY_SHA256.json").read_text())
    results = []
    for item in data:
        if item["version"] != version:
            continue
        rel = Path(item["path"])
        require(not rel.is_absolute() and ".." not in rel.parts and
                all(part not in ("", ".") for part in rel.parts),
                "invalid external path in source lock")
        name = str(Path("versions") / version / rel)
        require(len(item["sha256"]) == 64 and all(x in "0123456789abcdef"
                for x in item["sha256"]), "invalid native source SHA")
        results.append((name, item["sha256"]))
    require(results and len(set(x[0] for x in results)) == len(results),
            "selected backbone has no unique external source entries")
    return results


def provision(repo, version, source, dry_run=False):
    repo = Path(repo).resolve()
    source = Path(source).resolve()
    require(source.is_dir() and not source.is_relative_to(repo),
            "private native source tree must be external to checkout")
    require((repo / ".git").exists(), "a Git clone with private .git/info/exclude is required")
    manifest = entries(repo, version)
    tracked = set(subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "-z"]).decode().split("\0"))
    actions = []
    for name, expected in manifest:
        src = source / Path(*Path(name).parts[1:])
        dest = repo / name
        require(src.is_file() and src.resolve().is_relative_to(source) and
                digest(src) == expected, "authorized native source missing/incorrect: " + name)
        require(not dest.exists() or (dest.is_file() and
                dest.resolve().is_relative_to(repo) and digest(dest) == expected),
                "private source would overwrite or conflict: " + name)
        require(name not in tracked or dest.exists(),
                "cannot install an untracked blob over a frozen tracked source: " + name)
        actions.append((name, src, dest, expected))
    if dry_run:
        return dict(status="VERIFIED_ONLY_NO_MUTATION", count=len(actions),
                    already_present=sum(dest.exists() for _, _, dest, _ in actions))
    excluded = Path(subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--git-path", "info/exclude"],
        text=True).strip())
    if not excluded.is_absolute():
        excluded = repo / excluded
    excluded.parent.mkdir(parents=True, exist_ok=True)
    existing = excluded.read_text() if excluded.is_file() else ""
    try:
        for name, src, dest, expected in actions:
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".draftkv-incomplete")
            require(not tmp.exists(), "incomplete native installation; manual audit required: " + name)
            shutil.copyfile(src, tmp)
            require(digest(tmp) == expected, "copied native source SHA mismatch: " + name)
            tmp.replace(dest)
        extra = sorted("/" + name for name, _, dest, _ in actions
                       if dest.exists() and name not in tracked and
                       "/" + name not in existing.splitlines())
        with excluded.open("a") as output:
            if extra:
                output.write("\n# Authorized local-only DraftKV native dependencies\n" +
                             "\n".join(extra) + "\n")
    finally:
        # Do not remove evidence or attempt to conceal partial changes.
        pass
    require(not subprocess.check_output(["git", "-C", str(repo), "status",
                                         "--porcelain"], text=True),
            "public source Git checkout became dirty after authorized dependency provisioning")
    return dict(status="AUTHORIZED_NATIVE_SOURCES_READY",count=len(actions),
                source_hashes_verified=len(actions), git_clean=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", choices=VERSIONS, required=True)
    p.add_argument("--dependency-root", required=True,
                   help="user-provisioned tree: <root>/<version>/<package-relative path>")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    try:
        result = provision(REPO, VERSIONS[args.backbone], args.dependency_root,
                           dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        p.exit(2, "BLOCKED: " + str(error) + "\n")


if __name__ == "__main__":
    sys.exit(main())
