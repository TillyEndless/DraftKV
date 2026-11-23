"""Public --model spellings keep the frozen --backbone implementation compatible."""
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location(path.replace("/", "_"), ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelCLITest(unittest.TestCase):
    def test_primary_cli_accepts_model_and_legacy_backbone(self):
        draftkv = load("scripts/draftkv.py")
        parser = draftkv.make_parser()
        for action in ("bootstrap", "environment", "assets", "install", "run", "prepare", "native-deps"):
            for flag in ("--model", "--backbone"):
                with self.subTest(action=action, flag=flag):
                    options = [action, flag, "lingbot1.3b"]
                    # Parser-only check: other required args are provided where necessary.
                    if action == "assets":
                        options += ["--asset-root", "/tmp/assets"]
                    elif action == "install":
                        options += ["--cuda-home", "/usr/local/cuda"]
                    elif action == "run":
                        options += ["--input", "/tmp/request.json", "--output", "/tmp/out.mp4"]
                    elif action == "prepare":
                        options += ["--dataset-root", "/tmp/dataset", "--case-id", "case_0001",
                                    "--output", "/tmp/prepared"]
                    args = parser.parse_args(options)
                    self.assertEqual(args.backbone, "lingbot1.3b")

    def test_low_level_profile_on_maps_to_frozen_summary(self):
        draftkv = load("scripts/draftkv.py")
        for value, expected in (("on", "summary"), ("summary", "summary"), ("off", "off")):
            with self.subTest(profile=value):
                with mock.patch.object(draftkv, "run") as invoke:
                    status = draftkv.main(["run", "--model", "lingbot1.3b",
                                           "--input", "/tmp/request.json",
                                           "--output", "/tmp/video.mp4", "--profile", value])
                self.assertEqual(status, 0)
                self.assertEqual(invoke.call_args.args[0].profile, expected)

    def test_public_reproduction_accepts_model_and_legacy_alias(self):
        entry = load("scripts/reproduce.py")
        for flag in ("--model", "--backbone"):
            with self.subTest(flag=flag):
                with mock.patch.object(entry, "run_command") as runner:
                    with mock.patch("builtins.print"):
                        code = entry.main(["infer", flag, "lingbot1.3b", "--gpu", "2",
                                           "--profile", "on", "--input", "/tmp/request.json",
                                           "--dry-run"])
                self.assertEqual(code, 0)
                args = runner.call_args.args[0]
                self.assertIn("--backbone", args)  # Frozen original worker's private protocol
                self.assertEqual(args[args.index("--backbone") + 1], "lingbot1.3b")
                self.assertEqual(args[args.index("--gpu") + 1], "2")
                self.assertEqual(args[args.index("--profile") + 1], "on")

    def test_source_provisioning_and_evaluator_setup_accept_model(self):
        setup = load("scripts/setup_benchmarks.py")
        for flag in ("--model", "--backbone"):
            with self.subTest(flag=flag):
                with mock.patch.object(setup, "setup") as mocked:
                    with mock.patch("builtins.print"):
                        code = setup.main([flag, "lingbot1.3b",
                                           "--asset-root", "/tmp/external-assets", "--dry-run"])
                self.assertEqual(code, 0)
                self.assertEqual(mocked.call_args.args[0].backbone, "lingbot1.3b")

    def test_auxiliary_script_help_mentions_model(self):
        for path, args in (
            ("scripts/_install.py", ["--help"]),
            ("scripts/provision_sources.py", ["--help"]),
            ("scripts/build_fa3.py", ["--help"]),
        ):
            with self.subTest(path=path):
                result = subprocess.run(
                    [sys.executable, str(ROOT / path), *args],
                    cwd=ROOT, text=True, capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--model", result.stdout)
                self.assertIn("--backbone", result.stdout)

    def test_readme_uses_base_model_terminology_and_model_flag(self):
        for name in ("README.md", "docs/README_CN.md"):
            text = (ROOT / name).read_text()
            self.assertIn("--model", text)
            self.assertNotIn("--backbone", text)
            self.assertNotIn("Backbone", text)
            self.assertIn("基础模型" if name.endswith("_CN.md") else "base model", text.lower() if not name.endswith("_CN.md") else text)


if __name__ == "__main__":
    unittest.main()
