"""Deployment binder regression: immutable source, input and native identity."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

FILE = Path(__file__).resolve().parents[1] / "tools/fresh_binding.py"
SPEC = importlib.util.spec_from_file_location("fresh_binding", FILE)
binding = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(binding)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FreshBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "checkout"
        version = self.repo / "versions/lingbot13"
        native = version / "native"
        native.mkdir(parents=True)
        source_file = version / "src/draftkv/backbones/worker.py"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("# verified immutable fixture\n")
        (version/"native/model.py").write_text("# native dependency\n")
        (self.repo/"RETAINED_SOURCE_SHA256.json").write_text(json.dumps([
            {"version":"lingbot13", "path":"src/draftkv/backbones/worker.py",
             "sha256":digest(source_file)}]))
        (self.repo/"EXTERNAL_DEPENDENCY_SHA256.json").write_text("[]")
        manifest = self.base/"manifest.json"
        manifest.write_text('{"cases":[]}\n')
        image = self.base/"image.png"
        image.write_bytes(b"fixture image")
        camera = self.base/"camera"
        camera.mkdir()
        (camera/"poses.npy").write_bytes(b"poses")
        (camera/"intrinsics.npy").write_bytes(b"intrinsics")
        checkpoint = self.base/"model-weights"
        checkpoint.mkdir()
        assets = self.base/"model-assets"
        assets.mkdir()
        interpreter = self.base/"python"
        interpreter.write_bytes(b"python placeholder")
        self.original = dict(backbone="LingBot1.3B", algorithm="draftkv", sp=1,
            repo_root=str(version), source_root=str(native), native_root=str(native),
            checkpoint=str(checkpoint), assets_dir=str(assets), python=str(interpreter),
            repo_commit="historical-src", gpu_profile=False,
            run_dir=str(self.base/"old"), output=str(self.base/"old/native.mp4"),
            adapter_source_files={"src/draftkv/backbones/worker.py":digest(source_file)},
            algorithm_plugin=dict(entry="src/draftkv/algorithms/draftkv/entry.py",
                root=str(version),
                files={"src/draftkv/backbones/worker.py":digest(source_file)},
                config={"b01":{"parameters":{"consumer":"original"},
                               "build_spec_path":str(self.base/"old-spec.json"),
                               "build_spec_sha256":"x"*64,
                               "cache_directory":str(self.base/"cache-old")},
                        "some_other_scientific_setting":"unchanged"}),
            request=dict(dataset_manifest=str(manifest),
                dataset_manifest_sha256=digest(manifest), image=str(image),
                camera_dir=str(camera),
                asset_sha256=dict(image=digest(image),
                    poses=digest(camera/"poses.npy"),
                    intrinsics=digest(camera/"intrinsics.npy"))))
        self.plan=self.base/"original-plan.json"
        self.plan.write_text(json.dumps(self.original))
        self.spec=self.base/"fresh-spec.json"
        self.spec.write_text('{"clean_commit":"abc"}')
        self.cache=self.base/"cache"
        self.cache.mkdir()
        self.run=self.base/"run"
        self.args=SimpleNamespace(backbone="lingbot1.3b",input_plan=str(self.plan),
            spec=str(self.spec),cache_directory=str(self.cache),
            run_dir=str(self.run),output=str(self.run/"native.mp4"),profile="summary")
        self.patch_root=patch.object(binding,"ROOT",self.repo)
        self.patch_identity=patch.object(binding,"source_identity",return_value="abc")
        self.patch_receipt=patch.object(binding,"native_receipt",return_value="0123")
        for p in (self.patch_root,self.patch_identity,self.patch_receipt):
            p.start()
            self.addCleanup(p.stop)

    def test_bind_verify_and_tamper_deployed_plan(self):
        binding.bind(self.args)
        self.assertTrue(binding.verify(self.run/"plan.json",self.run/"deployment-evidence.json"))
        payload=json.loads((self.run/"plan.json").read_text())
        self.assertEqual(payload["algorithm_plugin"]["config"]["some_other_scientific_setting"],
                         "unchanged")
        payload["algorithm_plugin"]["config"]["some_other_scientific_setting"]="tampered"
        (self.run/"plan.json").write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError,"changed"):
            binding.verify(self.run/"plan.json",self.run/"deployment-evidence.json")

    def test_reject_foreign_algorithm_and_camera_tampering(self):
        altered=json.loads(self.plan.read_text())
        altered["algorithm"]="native"
        self.plan.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError,"single-GPU DraftKV"):
            binding.bind(self.args)
        self.plan.write_text(json.dumps(self.original))
        (self.base/"camera/poses.npy").write_bytes(b"new poses")
        with self.assertRaisesRegex(ValueError,"camera SHA"):
            binding.bind(self.args)

    def test_no_overwrite_existing_run_or_input_mutation(self):
        binding.bind(self.args)
        with self.assertRaisesRegex(ValueError,"run directory already exists"):
            binding.bind(self.args)
        self.plan.write_text(json.dumps({**self.original,"secret":"modified input"}))
        with self.assertRaisesRegex(ValueError,"changed"):
            binding.verify(self.run/"plan.json",self.run/"deployment-evidence.json")


if __name__ == "__main__":
    unittest.main()
