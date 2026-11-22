"""Offline authorized native source provisioning: strict SHA, no Git pollution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "scripts/provision_sources.py"
SPEC = importlib.util.spec_from_file_location("provision_sources", MODULE)
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


class ProvisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.repo = self.base / "repo"
        self.repo.mkdir()
        self.source = self.base / "private"
        self.file = self.source / "lingbot13/src/draftkv/native/source.py"
        self.file.parent.mkdir(parents=True)
        self.file.write_text("frozen exact authorized source\n")
        self.expected = hashlib.sha256(self.file.read_bytes()).hexdigest()
        manifest = [dict(version="lingbot13",
                         path="src/draftkv/native/source.py", sha256=self.expected)]
        (self.repo/"EXTERNAL_DEPENDENCY_SHA256.json").write_text(json.dumps(manifest))
        subprocess.run(["git","init","-q",str(self.repo)],check=True)
        subprocess.run(["git","-C",str(self.repo),"config","user.name","Test"],check=True)
        subprocess.run(["git","-C",str(self.repo),"config","user.email","test@example.com"],check=True)
        subprocess.run(["git","-C",str(self.repo),"add","."],check=True)
        subprocess.run(["git","-C",str(self.repo),"commit","-qm","baseline"],check=True)

    def test_dry_run_unchanged_git(self):
        report = p.provision(self.repo,"lingbot13",self.source,dry_run=True)
        self.assertEqual(report["count"],1)
        self.assertEqual(report["status"],"VERIFIED_ONLY_NO_MUTATION")
        self.assertFalse((self.repo/"versions/lingbot13/src/draftkv/native/source.py").exists())

    def test_real_install_sha_and_git_clean_reuse(self):
        result = p.provision(self.repo,"lingbot13",self.source)
        self.assertEqual(result["source_hashes_verified"],1)
        dest = self.repo/"versions/lingbot13/src/draftkv/native/source.py"
        self.assertEqual(dest.read_bytes(),self.file.read_bytes())
        self.assertEqual(subprocess.check_output(
            ["git","-C",str(self.repo),"status","--porcelain"]),b"")
        second = p.provision(self.repo,"lingbot13",self.source)
        self.assertEqual(second["status"],"AUTHORIZED_NATIVE_SOURCES_READY")
        dest.write_text("tampered")
        with self.assertRaisesRegex(ValueError,"conflict"):
            p.provision(self.repo,"lingbot13",self.source)

    def test_wrong_authorized_source_sha_rejected_without_mutation(self):
        self.file.write_text("tampered source")
        with self.assertRaisesRegex(ValueError,"missing/incorrect"):
            p.provision(self.repo,"lingbot13",self.source)
        self.assertFalse((self.repo/"versions/lingbot13/src/draftkv/native/source.py").exists())


if __name__ == "__main__":
    unittest.main()
