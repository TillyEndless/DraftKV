"""Offline proof of compact native-source manifests and source-safe provisioning."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


sources = module("_test_native_source_selection", ROOT / "scripts/native_sources.py")
provision = module("_test_native_source_provision", ROOT / "scripts/provision_sources.py")


class CompactSourceTests(unittest.TestCase):
    def test_compact_manifest_is_standalone(self):
        index = sources.manifest(ROOT)
        self.assertEqual(len(index["entries"]), 100)
        self.assertEqual(sum(len(e["variants"]) for e in index["entries"]), 101)
        self.assertFalse((ROOT / "EXTERNAL_DEPENDENCY_SHA256.json").exists())
        self.assertNotIn("legacy_sha256", index)

    def test_no_root_historical_lock_is_required(self):
        for name in ("EXTERNAL_DEPENDENCY_SHA256.json",
                     "RETAINED_SOURCE_SHA256.json",
                     "ORIGINAL_COMMAND_MAPPING.json"):
            self.assertFalse((ROOT / name).exists())
        self.assertTrue((ROOT / "scripts/model_execution_profiles.json").is_file())

    def test_per_backbone_inference_and_full_selection(self):
        lengths = {"longlive1.3b": 49, "lingbot1.3b": 53,
                   "lingbot14b": 53, "solarwm33b": 4}
        online = {"longlive1.3b": 28, "lingbot1.3b": 20,
                  "lingbot14b": 20, "solarwm33b": 0}
        for backbone, version in sources.VERSIONS.items():
            with self.subTest(backbone=backbone):
                short = sources.selected(ROOT, version, backbone, "inference")
                full = sources.selected(ROOT, version, backbone, "full")
                self.assertEqual(len(short), lengths[backbone])
                self.assertEqual(len(full), 100)
                self.assertEqual(sum(x.get("online") is not None for x in short),
                                 online[backbone])
                self.assertTrue(all("sm120" not in x["path"] for x in short))
                self.assertEqual(len([x for x in short if x["path"].endswith(".tar.gz")]), 3)
                self.assertTrue(set(x["path"] for x in short).issubset(
                    x["path"] for x in full))

    def test_frozen_model_source_lock_is_preserved(self):
        for b, ver, prefix, n in [
            ("longlive1.3b", "longlive", "/longlive13/", 45),
            ("lingbot1.3b", "lingbot13", "/lingbot14/", 49),
            ("lingbot14b", "h200_large", "/lingbot14/", 49),
        ]:
            actual = {x["path"] for x in sources.selected(ROOT, ver, b, "inference")}
            locked = {x["path"] for x in sources.selected(ROOT, ver, b, "full")
                      if prefix in x["path"]}
            self.assertEqual(len(locked), n)
            self.assertTrue(locked <= actual)

    def test_only_pinned_hash_compatible_upstream_urls_are_present(self):
        index = sources.manifest(ROOT)
        all_online = [v["online"] for x in index["entries"]
                      for v in x["variants"] if v.get("online")]
        self.assertEqual(len(all_online), 48)
        self.assertTrue(all(x["url"].startswith(
            "https://raw.githubusercontent.com/") and len(x["commit"]) == 40
            for x in all_online))

    def test_variant_does_not_erase_sp8_exception(self):
        filename = "src/draftkv/vendor/backbones/lingbot14/wan/image2video.py"
        old = next(x for x in sources.selected(ROOT, "h200_large", "lingbot14b", "full")
                   if x["path"] == filename)["sha256"]
        newer = next(x for x in sources.selected(ROOT, "sp8_solar_de790",
                                                  profile="full")
                     if x["path"] == filename)["sha256"]
        self.assertNotEqual(old, newer)


class VerifiedFetchTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "scripts").mkdir()
        self.cache = self.root / "cache"
        self.data = b"immutable upstream source contents"
        self.sha = hashlib.sha256(self.data).hexdigest()
        self.name = "src/draftkv/vendor/backbones/lingbot14/wan/modules/attention.py"
        self.url = ("https://raw.githubusercontent.com/Robbyant/lingbot-world-v2/" +
                    "1895d300d8ac936401689b26389f51cbd36530eb/" +
                    "wan/modules/attention.py")
        self.legacy = [{"version": "lingbot13", "path": self.name,
                        "sha256": self.sha}]
        self._commit_fixture()

    def _commit_fixture(self):
        entries = []
        for entry in self.legacy:
            item = {"path": entry["path"], "group": "lingbot_source_lock",
                    "role": "model_source_and_full_source_lock",
                    "inference_backbones": ["lingbot1.3b", "lingbot14b"],
                    "variants": [{"sha256": entry["sha256"],
                                  "versions": ["lingbot13"]}]}
            if entry["path"] == self.name:
                item["variants"][0]["online"] = {
                    "url": self.url, "sha256": self.sha,
                    "bytes": len(self.data)}
            entries.append(item)
        (self.repo / "scripts/native_sources.json").write_text(json.dumps({
            "schema_version": 1,
            "entries": entries,
        }))
        if not (self.repo / ".git").exists():
            subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
            subprocess.run(["git", "-C", str(self.repo), "config",
                            "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(self.repo), "config",
                            "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-qm", "fixture"],
                       check=True)

    def test_network_fetch_is_hash_checked_cached_and_git_clean(self):
        with patch.object(provision, "urlopen",
                          side_effect=lambda url, **kw: io.BytesIO(self.data)) as fetch:
            result = provision.provision(self.repo, "lingbot13",
                                         backbone="lingbot1.3b",
                                         cache_root=self.cache)
            self.assertEqual(result["source_hashes_verified"], 1)
            self.assertEqual(result["status"], "AUTHORIZED_NATIVE_SOURCES_READY")
            self.assertEqual(result["installed_now"], 1)
            self.assertEqual(fetch.call_count, 1)
            again = provision.provision(self.repo, "lingbot13",
                                        backbone="lingbot1.3b",
                                        cache_root=self.cache)
            self.assertEqual(again["installed_now"], 0)
            self.assertEqual(fetch.call_count, 1)
        installed = self.repo / "versions/lingbot13" / self.name
        self.assertEqual(installed.read_bytes(), self.data)
        self.assertFalse(subprocess.check_output(
            ["git", "-C", str(self.repo), "status", "--porcelain"]))

    def test_wrong_online_bytes_fail_without_publishing(self):
        with patch.object(provision, "urlopen",
                          side_effect=lambda url, **kw: io.BytesIO(b"wrong")):
            with self.assertRaisesRegex(ValueError, "SHA-256/size mismatch"):
                provision.provision(self.repo, "lingbot13",
                                    backbone="lingbot1.3b", cache_root=self.cache)
        self.assertFalse((self.repo / "versions/lingbot13" / self.name).exists())

    def test_download_available_uses_exact_online_sources_without_claiming_completion(self):
        extra = {"version": "lingbot13",
                 "path": "src/draftkv/vendor/backbones/lingbot14/custom.py",
                 "sha256": "1" * 64}
        self.legacy.append(extra)
        self._commit_fixture()
        with patch.object(provision, "urlopen",
                          side_effect=lambda url, **kw: io.BytesIO(self.data)) as fetch:
            result = provision.provision(self.repo, "lingbot13",
                                         backbone="lingbot1.3b",
                                         cache_root=self.cache, allow_partial=True)
            self.assertEqual(result["status"], "PARTIAL_PUBLIC_NATIVE_SOURCES_DOWNLOADED")
            self.assertEqual(result["unavailable_count"], 1)
            self.assertEqual(result["source_hashes_verified"], 1)
            self.assertEqual(result["installed_now"], 1)
            self.assertEqual(fetch.call_count, 1)
        self.assertTrue((self.repo / "versions/lingbot13" / self.name).is_file())
        self.assertFalse((self.repo / "versions/lingbot13" / extra["path"]).exists())
        self.assertFalse(subprocess.check_output(
            ["git", "-C", str(self.repo), "status", "--porcelain"]))

    def test_missing_original_is_reported_before_any_network(self):
        extra = {"version": "lingbot13",
                 "path": "src/draftkv/vendor/backbones/lingbot14/custom.py",
                 "sha256": "1" * 64}
        self.legacy.append(extra)
        self._commit_fixture()
        with patch.object(provision, "urlopen") as fetch:
            plan = provision.provision(self.repo, "lingbot13",
                                       backbone="lingbot1.3b",
                                       cache_root=self.cache, dry_run=True)
            self.assertEqual(plan["unavailable_count"], 1)
            with self.assertRaisesRegex(ValueError, "unavailable native-source"):
                provision.provision(self.repo, "lingbot13",
                                    backbone="lingbot1.3b", cache_root=self.cache)
            fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
