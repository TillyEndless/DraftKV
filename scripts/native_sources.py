#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compact deployment view of original external source SHA locks.

The 500-row EXTERNAL_DEPENDENCY_SHA256.json is immutable for frozen launchers.
This module reconstructs its exact mapping from 100 path records / 101 SHA
variants, then selects only the source-locked files for the current backbone.
"""
import hashlib
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
NAME = "native_sources.json"
VERSIONS = {
    "longlive1.3b": "longlive", "lingbot1.3b": "lingbot13",
    "lingbot14b": "h200_large", "solarwm33b": "h200_large",
}
SHA = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_path(name):
    p = Path(name)
    require(not p.is_absolute() and ".." not in p.parts and
            all(v not in ("", ".") for v in p.parts),
            "unsafe external dependency path: " + str(name))
    return p


def _expected_inference_backbones(path):
    # Constrain the selector independently of editable manifest annotations.
    # Full original model-source locks are required even for helper/train files.
    if path.startswith("src/draftkv/vendor/backbones/lingbot14/"):
        return ["lingbot1.3b", "lingbot14b"]
    if path.startswith("src/draftkv/vendor/backbones/longlive13/"):
        return ["longlive1.3b"]
    if "sm120_cuda/" in path:
        return []
    if path == "src/draftkv/streaming/prepacked_cuda/prepacked_fwd_kernel.h":
        return list(VERSIONS)
    if path in {
        "src/draftkv/streaming/vendor/b01-original.tar.gz",
        "src/draftkv/vendor/b01/source_snapshot.tar.gz",
        "src/draftkv/vendor/b01/historical_evidence.tar.gz",
    }:
        return list(VERSIONS)
    raise ValueError("external source has no verified runtime category: " + path)


def manifest(repo=BASE):
    repo = Path(repo)
    compact = repo / "scripts" / NAME
    if not compact.is_file():
        return None  # Older isolated fixtures and legacy tools may still use the flat lock.
    data = json.loads(compact.read_text())
    require(data.get("schema_version") == 1 and isinstance(data.get("entries"), list),
            "invalid compact native-source manifest")
    legacy_path = repo / "EXTERNAL_DEPENDENCY_SHA256.json"
    require(data.get("legacy_sha256") == digest(legacy_path),
            "compact native-source manifest does not match frozen legacy SHA lock")
    expected = {(e["version"], e["path"]): e["sha256"]
                for e in json.loads(legacy_path.read_text())}
    require(len(expected) > 0, "legacy native SHA lock has no entries")
    actual = {}
    for entry in data["entries"]:
        name = str(validate_path(entry["path"]))
        require(isinstance(entry.get("inference_backbones"), list) and
                isinstance(entry.get("variants"), list),
                "invalid native-source group or variant")
        require(entry["inference_backbones"] == _expected_inference_backbones(name),
                "compact native-source inference group does not match verified selector")
        for variant in entry["variants"]:
            sha = variant["sha256"]
            require(isinstance(sha, str) and SHA.fullmatch(sha),
                    "invalid compact source SHA")
            for version in variant["versions"]:
                key = (version, name)
                require(key not in actual, "duplicate compact source version/path")
                actual[key] = sha
            online = variant.get("online")
            if online:
                require(online.get("sha256") == sha and
                        type(online.get("bytes")) is int and online["bytes"] >= 0 and
                        re.fullmatch(r"https://raw\.githubusercontent\.com/"
                                     r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[a-f0-9]{40}/"
                                     r"[A-Za-z0-9_.+/-]+", online.get("url", "")),
                        "unverified or unpinned external source URL")
    require(actual == expected, "compact source projection differs from frozen original source identities")
    return data


def selected(repo, version, backbone=None, profile="inference"):
    require(profile in ("inference", "full"), "invalid native dependency profile")
    if backbone is None:
        backbone = next((k for k, v in VERSIONS.items() if v == version), None)
    require((backbone is not None and VERSIONS.get(backbone) == version) or
            (profile == "full" and version.startswith("sp8_")),
            "backbone does not match native source version")
    data = manifest(repo)
    if data is None:
        # Legacy isolated test fixtures only: original 500-row manifest remains supported.
        legacy = json.loads((Path(repo) / "EXTERNAL_DEPENDENCY_SHA256.json").read_text())
        result = [{"path": x["path"], "sha256": x["sha256"], "online": None,
                   "role": "legacy", "group": "original_full_inventory"}
                  for x in legacy if x["version"] == version]
        return sorted(result, key=lambda x: x["path"])
    matches = []
    for row in data["entries"]:
        if profile == "inference" and backbone not in row["inference_backbones"]:
            continue
        candidate = [v for v in row["variants"] if version in v["versions"]]
        require(len(candidate) == 1, "ambiguous compact source variant")
        v = candidate[0]
        matches.append({"path": row["path"], "sha256": v["sha256"],
                        "online": v.get("online"), "role": row["role"],
                        "group": row["group"]})
    require(len({x["path"] for x in matches}) == len(matches) and matches,
            "selected native source set is empty or has duplicate paths")
    return sorted(matches, key=lambda x: x["path"])
