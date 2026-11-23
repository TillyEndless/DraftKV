"""Validate execution-plan source inventories against installed source files.

No historical whole-repository hash list is required. The plan still binds
its exact source bytes and the Native build/input receipts remain checked.
"""
from pathlib import Path
import hashlib


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_count(repo, version=None):
    base = Path(repo) / "versions"
    roots = [base / version] if version else list(base.iterdir()) if base.is_dir() else []
    files = [p for root in roots if root.is_dir()
             for p in root.rglob("*.py") if p.is_file() and not p.is_symlink()]
    if not files:
        raise ValueError("no installed model Python source files found")
    return len(files)


def check_inventory(repo, version, inventory):
    root = (Path(repo) / "versions" / version).resolve()
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("nonempty model source inventory required")
    for name, digest in inventory.items():
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("invalid model source inventory entry")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in ("", "."):
            raise ValueError("invalid model source inventory entry")
        file = root / relative
        if (not file.is_file() or file.is_symlink() or
                not file.resolve().is_relative_to(root) or sha(file) != digest):
            raise ValueError("execution plan source missing/changed: " + name)
    return len(inventory)
