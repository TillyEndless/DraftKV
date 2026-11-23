"""Offline coverage for bootstrap's automatic FA2/FA3 installation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("draftkv_bootstrap_attention_test", ROOT / "scripts/draftkv.py")
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class AutomaticAttentionBootstrapTests(unittest.TestCase):
    def setUp(self):
        ctx = tempfile.TemporaryDirectory(prefix="draftkv-attention-test-")
        self.addCleanup(ctx.cleanup)
        self.root = Path(ctx.name)
        self.environment_root = self.root / "envs"
        self.backbone = "lingbot1.3b"
        self.base = self.environment_root / self.backbone
        python = self.base / "venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text("mock Python for offline unit tests")
        self.cuda_home = self.root / "cuda"
        (self.cuda_home / "bin").mkdir(parents=True)
        (self.cuda_home / "bin/nvcc").write_text("mock nvcc executable")
        self.wheel = self.root / "flash_attn.whl"
        self.wheel.write_bytes(b"test-only fake wheel")
        self.args = cli.make_parser().parse_args(
            ["bootstrap", "--backbone", self.backbone,
             "--env-root", str(self.environment_root)])
        self.lock = cli.pinned_official_fa2_wheel(self.backbone)

    def _gpu(self, backend):
        return dict(backend=backend,
                    gpu_index=0, detected=True,
                    name="NVIDIA H200" if backend == "fa3" else "NVIDIA GeForce RTX 5090",
                    compute_capability="9.0" if backend == "fa3" else "12.0",
                    driver_version="570.144")

    def _commands(self, command, env=None, capture=False):
        args = [str(x) for x in command]
        if capture:
            return SimpleNamespace(stdout=json.dumps({"version": self.lock["version"]}))
        if any("build_fa3.py" in x for x in args):
            receipt = self.base / "native/fa3/build-receipt.json"
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps(dict(status="BUILT_AND_INSTALLED",
                                               backbone=self.backbone)))

    def test_5090_bootstrap_installs_fa2_not_fa3_and_reuses_receipt(self):
        with patch.object(cli.gpu_backend, "detect", return_value=self._gpu("fa2")), \
             patch.object(cli, "obtain_fa2_wheel",
                          return_value=(self.wheel, self.lock["sha256"])) as obtain, \
             patch.object(cli, "commands", side_effect=self._commands) as execute:
            first = cli.bootstrap_attention(self.args)
            self.assertEqual(first["selected_backend"], "fa2")
            self.assertEqual(first["fa2"], "INSTALLED")
            self.assertEqual(first["fa3"], "NOT_APPLICABLE")
            self.assertEqual(first["runtime_validation"], "NOT_EXECUTED")
            self.assertEqual(obtain.call_count, 1)
            self.assertFalse(any("build_fa3.py" in str(v)
                                 for c in execute.call_args_list for v in c.args[0]))
            again = cli.bootstrap_attention(self.args)
            self.assertEqual(again["fa2"], "REUSED_VERIFIED_INSTALL")
            self.assertEqual(obtain.call_count, 1)
        receipt = json.loads((self.base / "native/attention-install.json").read_text())
        self.assertEqual(receipt["selected_backend"], "fa2")

    def test_h200_bootstrap_builds_fa3_in_addition_to_model_fa2(self):
        with patch.object(cli.gpu_backend, "detect", return_value=self._gpu("fa3")), \
             patch.object(cli, "_cuda_toolkit_root",
                          return_value=str(self.cuda_home)), \
             patch.object(cli, "obtain_fa2_wheel",
                          return_value=(self.wheel, self.lock["sha256"])) as obtain, \
             patch.object(cli, "commands", side_effect=self._commands) as execute:
            result = cli.bootstrap_attention(self.args)
        self.assertEqual(result["selected_backend"], "fa3")
        self.assertEqual(result["fa2"], "INSTALLED")
        self.assertEqual(result["fa3"], "BUILT_AND_INSTALLED")
        self.assertEqual(obtain.call_count, 1)
        self.assertTrue(any("build_fa3.py" in str(v)
                            for c in execute.call_args_list for v in c.args[0]))
        self.assertTrue((self.base / "native/fa3/build-receipt.json").is_file())

    def test_dry_run_never_downloads_or_builds(self):
        self.args.dry_run = True
        for backend in ["fa2", "fa3"]:
            with self.subTest(backend=backend):
                with patch.object(cli.gpu_backend, "detect", return_value=self._gpu(backend)), \
                     patch.object(cli, "commands") as exec_command, \
                     patch.object(cli, "obtain_fa2_wheel") as fetch:
                    result = cli.bootstrap_attention(self.args)
                    self.assertEqual(result["selected_backend"], backend)
                    exec_command.assert_not_called()
                    fetch.assert_not_called()

    def test_existing_receipt_mismatched_gpu_fails(self):
        (self.base / "native").mkdir()
        receipt = self.base / "native/attention-install.json"
        receipt.write_text(json.dumps(dict(backbone=self.backbone,
                                           selected_backend="fa3",
                                           fa2_sha256=self.lock["sha256"])))
        with patch.object(cli.gpu_backend, "detect", return_value=self._gpu("fa2")), \
             patch.object(cli, "commands") as action:
            with self.assertRaisesRegex(ValueError, "receipt disagrees"):
                cli.bootstrap_attention(self.args)
            action.assert_not_called()


if __name__ == "__main__":
    unittest.main()
