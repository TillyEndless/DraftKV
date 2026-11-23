"""Pure CPU tests for hardware-selected FlashAttention installation."""
import importlib.util
import io
from pathlib import Path
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def read_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = read_module("draftkv_gpu_backend_tests", "gpu_backend.py")
cli = read_module("draftkv_gpu_cli_tests", "draftkv.py")
installer = read_module("draftkv_gpu_installer_tests", "_install.py")


class DeviceBackendTests(unittest.TestCase):
    def test_hopper_builds_fa3_other_gpus_use_fa2(self):
        fixtures = [
            ("NVIDIA H100 80GB HBM3, 9.0, 535.129", "fa3"),
            ("NVIDIA H200, 9.0, 570.144", "fa3"),
            ("NVIDIA GeForce RTX 5090, 12.0, 570.144", "fa2"),
            ("NVIDIA GeForce RTX 4090, 8.9, 550.42", "fa2"),
            ("NVIDIA A100-SXM4-80GB, 8.0, 535.1", "fa2"),
        ]
        for output, expected in fixtures:
            with self.subTest(device=output):
                with patch.object(policy.subprocess, "check_output",
                                  return_value=output + "\n") as run:
                    actual = policy.detect(gpu_index=2)
                    self.assertEqual(actual["backend"], expected)
                    self.assertTrue(actual["detected"])
                    self.assertEqual(run.call_args.args[0][1], "--id=2")

    def test_missing_nvidia_smi_is_not_a_hopper_requirement(self):
        with patch.object(policy.subprocess, "check_output",
                          side_effect=FileNotFoundError("nvidia-smi")):
            actual = policy.detect(0)
        self.assertFalse(actual["detected"])
        self.assertEqual(actual["backend"], "fa2")

    def test_invalid_gpu_identifier_refused(self):
        for number in (-1, True, "0"):
            with self.assertRaises(ValueError):
                policy.detect(number)

    def test_hopper_only_fa3_compilation_policy(self):
        args = SimpleNamespace(build_fa3=True)
        self.assertTrue(cli.should_build_fa3(args, {"backend": "fa3"}))
        self.assertFalse(cli.should_build_fa3(args, {"backend": "fa2"}))
        args.build_fa3 = False
        self.assertFalse(cli.should_build_fa3(args, {"backend": "fa2"}))
        with self.assertRaisesRegex(ValueError, "Hopper requires FA3"):
            cli.should_build_fa3(args, {"backend": "fa3"})

    def test_install_dry_run_shows_5090_fa2_and_h200_fa3(self):
        for gpu,expected in [("NVIDIA GeForce RTX 5090, 12.0, 570.144","fa2"),
                             ("NVIDIA H200, 9.0, 570.144","fa3")]:
            with self.subTest(gpu=gpu):
                args=cli.make_parser().parse_args(["install","--backbone","lingbot1.3b",
                                                    "--cuda-home","/usr/local/cuda-12.8",
                                                    "--gpu","1","--dry-run"])
                with patch.object(cli.gpu_backend.subprocess,"check_output",
                                  return_value=gpu + "\n"):
                    with patch("sys.stdout",new_callable=io.StringIO) as result:
                        cli.install(args)
                    import json
                    out=json.loads(result.getvalue())
                self.assertEqual(out["selected_attention_backend"],expected)
                self.assertEqual(out["gpu"]["gpu_index"],1)

    def test_non_hopper_fa3_optout_does_not_block_before_native_source_preflight(self):
        args=cli.make_parser().parse_args(["install","--backbone","lingbot1.3b",
                                           "--asset-root","/tmp/models","--cuda-home",
                                           "/usr/local/cuda-12.8","--no-build-fa3"])
        with patch.object(cli.gpu_backend, "detect",
                          return_value=dict(backend="fa2",detected=True,
                                            compute_capability="12.0")):
            with patch.object(cli, "commands") as run:
                with self.assertRaisesRegex(ValueError, "CUDA/C\+\+ extension and model sources are not all downloadable"):
                    cli.install(args)
                run.assert_not_called()

    def test_5090_reporting_no_longer_requires_h200_device_identity(self):
        environment={"platform":"Darwin arm64","compiler_cuda":"12.6",
                     "compiler_gcc":"12.0","recorded_compute_capability":"9.0"}
        # Check actual status independent of the historical SM90 profile.
        class Recorder:
            def __init__(self):
                self.rows=[]
            def item(self,status,label,description):
                self.rows.append((status,label,description))
        output=Recorder()
        driver=SimpleNamespace(cuDriverGetVersion=lambda ptr:0)
        with patch.object(installer,"_gpu_backend",
                          return_value=SimpleNamespace(detect=lambda index=0:
                              dict(detected=True,name="NVIDIA GeForce RTX 5090",
                                   compute_capability="12.0",driver_version="570.144",
                                   backend="fa2"))), \
             patch.object(installer.subprocess,"check_output",
                          side_effect=["Cuda compilation tools, release 12.8",
                                       "12.2.0"]), \
             patch.object(installer.ctypes,"CDLL",return_value=driver), \
             patch.object(installer.shutil,"which",return_value=None):
            selected=installer.hardware_checks(environment,output,portable=True)
        self.assertEqual(selected,"fa2")
        device=[row for row in output.rows if row[1]=="GPU architecture/driver"]
        self.assertEqual(len(device),1)
        self.assertEqual(device[0][0],"PASS")
        self.assertNotIn("original profile SM90",device[0][2])
        self.assertEqual([row[0] for row in output.rows if row[1]=="nvcc"], ["PASS"])
        self.assertEqual([row[0] for row in output.rows if row[1]=="c++"], ["PASS"])


if __name__=="__main__":
    unittest.main()
