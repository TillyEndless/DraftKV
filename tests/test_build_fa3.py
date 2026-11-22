"""CPU-only tests for pinned Hopper FA3 source compilation workflow."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

FILE = Path(__file__).resolve().parents[1] / "scripts/build_fa3.py"
SPEC = importlib.util.spec_from_file_location("build_fa3", FILE)
fa3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fa3)


def tar_file(path, entries, prefix="official-abc"):
    with tarfile.open(path, "w:gz") as tar:
        for rel, body in entries.items():
            obj = tarfile.TarInfo(prefix + "/" + rel)
            obj.size = len(body)
            tar.addfile(obj, io.BytesIO(body))
    return path


class FA3BuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.artifact = self.root / "artifacts"

    def test_dry_run_has_no_side_effects_for_all_backbones(self):
        for model in fa3.BACKBONES:
            with self.subTest(model=model):
                with patch("builtins.print") as out:
                    rc = fa3.main(["--backbone", model, "--python",
                                   "/missing/python", "--cuda-home",
                                   "/usr/local/cuda-12.6", "--artifact-root",
                                   str(self.artifact), "--dry-run"])
                self.assertEqual(rc, 0)
                self.assertFalse(self.artifact.exists())
                self.assertIn("DRY_RUN_NO_INSTALL_OR_BUILD", out.call_args.args[0])

    def test_safe_extract_strips_only_one_root(self):
        source = tar_file(self.root / "source.tar.gz",
                          {"hopper/setup.py": b"from setuptools import setup\n",
                           "README.md": b"test"})
        out = self.root / "source"
        fa3.safe_extract(source, out)
        self.assertEqual((out / "README.md").read_bytes(), b"test")
        self.assertTrue((out / "hopper/setup.py").exists())

    def test_safe_extract_rejects_path_traversal(self):
        src = self.root / "bad.tar.gz"
        with tarfile.open(src, "w:gz") as tar:
            v = tarfile.TarInfo("safe/../../escape")
            v.size = 1
            tar.addfile(v, io.BytesIO(b"x"))
        with self.assertRaisesRegex(ValueError, "unsafe path"):
            fa3.safe_extract(src, self.root / "out")
        self.assertFalse((self.root / "escape").exists())

    def test_safe_extract_rejects_symlinks(self):
        src = self.root / "badlink.tar.gz"
        with tarfile.open(src, "w:gz") as tar:
            v = tarfile.TarInfo("safe/link")
            v.type = tarfile.SYMTYPE
            v.linkname = "../../escape"
            tar.addfile(v)
        with self.assertRaisesRegex(ValueError, "link or nonregular"):
            fa3.safe_extract(src, self.root / "out")

    def test_offline_pinned_archive_reused_and_not_mutated(self):
        path = self.root / "archive.tar.gz"
        path.write_bytes(b"fixture-bytes")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        info = dict(sha256=digest, url="https://example.invalid/fixture")
        with patch.object(fa3.request, "urlopen") as network:
            found = fa3.obtain(info, self.root / "unused.tar.gz", path)
            self.assertEqual(found, path.resolve())
            network.assert_not_called()
        self.assertEqual(path.read_bytes(), b"fixture-bytes")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            fa3.obtain(dict(sha256="0" * 64), self.root / "unused.tar.gz", path)

    def test_existing_wrong_cached_archive_refused(self):
        target = self.root / "source.tar.gz"
        target.write_bytes(b"tampered")
        with patch.object(fa3.request, "urlopen") as network:
            with self.assertRaisesRegex(ValueError, "cached archive SHA mismatch"):
                fa3.obtain(dict(sha256="1" * 64), target)
            network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
