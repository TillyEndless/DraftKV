"""CPU-only portable B01 provenance tests. No GPU, model imports or native build."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("portable_b01",
    Path(__file__).resolve().parents[1] / "tools/portable_b01.py")
port = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(port)


def spec(path, commit="f" * 40):
    return {
        "schema": 1, "arch": "9.0",
        "source_generation": "sg1", "compatibility_generation": "cg1",
        "cxx_flags": ["-O3"], "nvcc_flags": ["-lineinfo"],
        "modules": {"kernel": ["kernel.cu"]}, "clean_commit": commit,
        "cuda_home": "/usr/local/cuda",
        "abi": {"torch_version": "2.6.0+cu124", "torch_cuda": "12.4",
                "cxx11_abi": False, "python_version": "3.10",
                "python_executable": path, "soabi": "cpython-310-x86_64-linux-gnu",
                "ext_suffix": ".cpython-310-x86_64-linux-gnu.so"}}


class PortableB01Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.old = spec("/old/venv/bin/python")
        self.new = spec("/new/venv/bin/python")
        self.old_path = self.base / "original.json"
        self.new_path = self.base / "portable.json"
        self.old_path.write_text(json.dumps(self.old))
        self.new_path.write_text(json.dumps(self.new))
        self.native = self.base / "natives"
        self.native.mkdir()
        binary = self.native / "kernel.cpython-310-x86_64-linux-gnu.so"
        binary.write_bytes(b"fake native byte content for receipt identity only")
        self.receipt = dict(schema=1, validation="BUILD_EXPORTS_ONLY",
            spec=self.new, toolchain={"nvcc": "test"},
            binaries={"kernel": {"path": binary.name, "sha256": port.sha(binary)}})
        key = port.canonical(dict(spec=self.new, toolchain=self.receipt["toolchain"]))
        self.receipt["key"] = key
        self.native2 = self.base / key
        self.native.rename(self.native2)
        (self.native2 / "receipt.json").write_text(json.dumps(self.receipt))
        self.config = {"b01": {"cache_directory": "/old/cache",
                              "build_spec_path": str(self.old_path),
                              "build_spec_sha256": port.sha(self.old_path),
                              "parameters": {"consumer": "original"}},
                       "grouping": "fixed"}
        self.ref = {"algorithm": "draftkv", "backbone": "LingBot1.3B", "sp": 1,
                    "gpu_profile": True, "algorithm_plugin": {
                        "entry": "src/draftkv/algorithms/draftkv/entry.py",
                        "config": self.config}}
        self.portable = copy.deepcopy(self.ref)
        b = self.portable["algorithm_plugin"]["config"]["b01"]
        b.update(cache_directory=str(self.native2), build_spec_path=str(self.new_path),
                 build_spec_sha256=port.sha(self.new_path))
        self.mapping = {"scientific_keys": ["gpu_profile"],
                        "scientific_sections_sha256": port.scientific_identity(self.ref, ["gpu_profile"])}
        self.clean = patch.object(port, "clean_commit", return_value="f"*40)
        self.clean.start()
        self.addCleanup(self.clean.stop)

    def test_identity_changes_only_interpreter(self):
        port.compare_specs(self.old, self.new)

    def test_reject_scientific_spec_change(self):
        for section, key, replacement in [
            ("root", "arch", "8.9"),
            ("root", "modules", {"other": ["x.cu"]}),
            ("abi", "torch_cuda", "12.8"),
            ("abi", "soabi", "cpython-311-x86_64-linux-gnu")]:
            with self.subTest(field=key):
                value = copy.deepcopy(self.new)
                (value if section == "root" else value["abi"])[key] = replacement
                with self.assertRaisesRegex(ValueError, "changed"):
                    port.compare_specs(self.old, value)

    def test_cache_receipt_success_and_digest_rejection(self):
        self.assertTrue(port.verify_receipt(self.new, self.native2).exists())
        p = next(self.native2.glob("*.so"))
        p.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "native binary mismatch"):
            port.verify_receipt(self.new, self.native2)

    def test_approve_only_b01_path_and_sha_changes(self):
        result = port.approve(self.ref, self.portable, self.old_path,
                              self.new_path, self.native2, self.mapping)
        self.assertEqual(result["status"], "PORTABLE_B01_APPROVED_NOT_GPU_VALIDATED")

    def test_approval_rejects_algorithm_change(self):
        modified = copy.deepcopy(self.portable)
        modified["algorithm_plugin"]["config"]["grouping"] = "different"
        with self.assertRaisesRegex(ValueError, "non-B01 algorithm"):
            port.approve(self.ref, modified, self.old_path,
                         self.new_path, self.native2, self.mapping)

    def test_approval_rejects_old_unapproved_reference(self):
        altered = copy.deepcopy(self.ref)
        altered["gpu_profile"] = False
        with self.assertRaisesRegex(ValueError, "not the original approved"):
            port.approve(altered, self.portable, self.old_path,
                         self.new_path, self.native2, self.mapping)

    def test_approval_rejects_missing_receipt(self):
        (self.native2 / "receipt.json").unlink()
        with self.assertRaisesRegex(ValueError, "build receipt missing"):
            port.approve(self.ref, self.portable, self.old_path,
                         self.new_path, self.native2, self.mapping)

    def test_binding_fields_are_not_erased_from_non_b01_config(self):
        d = port.no_binding(self.portable["algorithm_plugin"]["config"])
        self.assertEqual(d["b01"], {"parameters": {"consumer": "original"}})
        self.assertEqual(d["grouping"], "fixed")

    def test_changed_compatibility_generation_requires_verified_bridge(self):
        altered = dict(self.new, compatibility_generation="new-python-contract")
        self.new_path.write_text(json.dumps(altered))
        self.portable["algorithm_plugin"]["config"]["b01"]["build_spec_sha256"] = port.sha(self.new_path)
        native = self.receipt.copy()
        native["spec"] = altered
        native["key"] = port.canonical(dict(spec=altered, toolchain=native["toolchain"]))
        new_native_dir = self.base / native["key"]
        self.native2.rename(new_native_dir)
        (new_native_dir / "receipt.json").write_text(json.dumps(native))
        self.portable["algorithm_plugin"]["config"]["b01"]["cache_directory"] = str(new_native_dir)
        with self.assertRaisesRegex(ValueError, "compatibility bridge attestation missing"):
            port.approve(self.ref, self.portable, self.old_path,
                         self.new_path, new_native_dir, self.mapping)

        version = self.base / "versions/lingbot13/src/draftkv/streaming"
        version.mkdir(parents=True)
        binding = version / "binding.py"
        binding.write_text("# fixture of immutable B01 bridge\n")
        attestation = {
            "schema": 1, "native_bridge_accepted": True,
            "original_spec_sha256": port.sha(self.old_path),
            "portable_spec_sha256": port.sha(self.new_path),
            "old_compatibility_generation": self.old["compatibility_generation"],
            "new_compatibility_generation": altered["compatibility_generation"],
            "binding_source_sha256": port.sha(binding),
        }
        (self.base / "compatibility-bridge.json").write_text(json.dumps(attestation))
        with patch.object(port, "REPO", self.base):
            result = port.approve(self.ref, self.portable, self.old_path,
                                  self.new_path, new_native_dir, self.mapping)
            self.assertEqual(result["status"], "PORTABLE_B01_APPROVED_NOT_GPU_VALIDATED")
            binding.write_text("# unauthorized mutation\n")
            with self.assertRaisesRegex(ValueError, "bridge evidence/source mismatch"):
                port.approve(self.ref, self.portable, self.old_path,
                             self.new_path, new_native_dir, self.mapping)

    def test_original_build_schema_and_architecture_rejected_if_altered(self):
        self.new["arch"] = "8.9"
        with self.assertRaisesRegex(ValueError, "arch"):
            port.compare_specs(self.old, self.new)

    def test_cli_fresh_spec_dispatch_actual_function(self):
        with patch.object(port, "prepare_fresh") as handler:
            self.assertEqual(port.main(["prepare-fresh", "--backbone", "lingbot1.3b",
                                        "--python", "/test/python",
                                        "--cuda-home", "/test/cuda",
                                        "--artifact-root", "/test/output"]), 0)
            handler.assert_called_once()

    def test_cli_build_and_bind_dispatch_functions(self):
        with patch.object(port, "build") as do_build:
            self.assertEqual(port.main(["build", "--backbone", "lingbot1.3b",
                                        "--python", "/test/python", "--spec", "/test/spec",
                                        "--cache-root", "/test/cache"]), 0)
            self.assertEqual(do_build.call_count, 1)
        with patch.object(port, "bind") as do_bind:
            self.assertEqual(port.main(["bind", "--backbone", "lingbot1.3b",
                                        "--reference-plan", "/test/reference",
                                        "--original-spec", "/test/old", "--spec", "/test/new",
                                        "--cache-directory", "/test/cache", "--run-dir", "/test/run",
                                        "--output", "/test/output", "--output-plan", "/test/plan",
                                        "--evidence", "/test/evidence"]), 0)
            self.assertEqual(do_bind.call_count, 1)

    def test_cli_does_not_accept_unexpected_mode(self):
        with self.assertRaises(SystemExit):
            port.main(["not-a-command"])


if __name__ == "__main__":
    unittest.main()
