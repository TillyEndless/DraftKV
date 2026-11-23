"""CPU-only tests for native Conda package installation and interpreter binding."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backend = load("test_env_backend", ROOT / "scripts/env_backend.py")
cli = load("test_conda_cli", ROOT / "scripts/draftkv.py")
installer = load("test_conda_installer", ROOT / "scripts/_install.py")


class NativeCondaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.prefix = self.root / "miniconda/envs/draftkv"
        (self.prefix / "bin").mkdir(parents=True)
        (self.prefix / "conda-meta").mkdir()
        (self.prefix / "bin/python").write_text("fake python fixture")
        self.python = self.prefix / "bin/python"
        self.python.chmod(0o755)
        self.cache = self.root / "cache/lingbot1.3b"
        self.env = patch.dict(os.environ, {
            "CONDA_PREFIX": str(self.prefix),
            "CONDA_DEFAULT_ENV": "draftkv",
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def conda_identity(self, command, **kwargs):
        return json.dumps({"version": [3, 10], "prefix": str(self.prefix),
                           "executable": str(self.python)})

    def test_torch_mirror_keeps_cuda_channel_locked(self):
        self.assertEqual(backend.torch_index_url(None, "12.4"),
                         "https://download.pytorch.org/whl/cu124")
        mirror = "https://mirrors.aliyun.com/pytorch-wheels/cu124"
        self.assertEqual(backend.torch_index_url(mirror, "12.4"), mirror)
        for bad in ("http://mirrors.aliyun.com/pytorch-wheels/cu124",
                    "https://mirrors.aliyun.com/pytorch-wheels/cu128",
                    "https://evil.example/pytorch-wheels/cu124"):
            with self.subTest(url=bad), self.assertRaisesRegex(ValueError, "PyTorch index"):
                backend.torch_index_url(bad, "12.4")
        args = cli.make_parser().parse_args(["bootstrap", "--backbone", "lingbot1.3b",
                                             "--torch-index-url", mirror])
        self.assertEqual(args.torch_index_url, mirror)

    def test_mode_selection_and_explicit_base_refusal(self):
        self.assertEqual(backend.resolve_mode("auto"), "conda")
        self.assertEqual(backend.resolve_mode("venv"), "venv")
        with patch.dict(os.environ, {"CONDA_DEFAULT_ENV": "base"}):
            self.assertEqual(backend.resolve_mode("auto"), "venv")
            with self.assertRaisesRegex(ValueError, "Conda base"):
                backend.conda_python()

    def test_validates_conda_interpreter_and_explicit_python(self):
        with patch.object(backend.subprocess, "check_output", side_effect=self.conda_identity):
            self.assertEqual(backend.conda_python(), self.python)
            self.assertEqual(backend.conda_python(str(self.python)), self.python)
            with self.assertRaisesRegex(ValueError, "--python"):
                backend.conda_python("/usr/bin/python3")
        with patch.object(backend.subprocess, "check_output", return_value=json.dumps({
                "version": [3, 9], "prefix": str(self.prefix)})):
            with self.assertRaisesRegex(ValueError, "CPython 3.10"):
                backend.conda_python()
        with patch.dict(os.environ, {"CONDA_PREFIX": ""}):
            with self.assertRaisesRegex(ValueError, "activated"):
                backend.conda_python()

    def test_binding_is_idempotent_and_backbones_are_mutually_exclusive(self):
        self.assertFalse(self.cache.exists())
        backend.binding(self.cache, "conda", self.python, "lingbot1.3b", claim=True)
        backend.binding(self.cache, "conda", self.python, "lingbot1.3b", claim=True)
        self.assertEqual(json.loads((self.prefix / ".draftkv-backbone.json").read_text())["backbone"],
                         "lingbot1.3b")
        with self.assertRaisesRegex(ValueError, "different DraftKV Backbone"):
            backend.binding(self.root / "cache/longlive1.3b", "conda", self.python,
                            "longlive1.3b", claim=True)
        with self.assertRaisesRegex(ValueError, "another Python/environment"):
            backend.binding(self.cache, "venv", self.cache / "venv/bin/python",
                            "lingbot1.3b")
        self.assertFalse((self.cache / "venv").exists())

    def test_legacy_cache_requires_separate_env_root(self):
        self.cache.mkdir(parents=True)
        (self.cache / "portable-dependency-closure.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "legacy venv"):
            backend.binding(self.cache, "conda", self.python, "lingbot1.3b")

    def test_paths_and_runtime_caches_stay_outside_conda(self):
        args = cli.make_parser().parse_args(["environment", "--backbone", "lingbot1.3b",
                                             "--env-root", str(self.root / "cache"),
                                             "--env-mode", "conda"])
        with patch.object(cli.env_backend, "conda_python", return_value=self.python):
            base, python, b01, _ = cli.paths(args)
        self.assertEqual(python, self.python)
        self.assertEqual(base, self.cache)
        self.assertEqual(b01, self.cache / "native/b01")
        actual = cli.env_for(python, base=base)
        self.assertEqual(actual["TRITON_CACHE_DIR"], str(self.cache / "cache/triton"))
        self.assertEqual(actual["TORCH_EXTENSIONS_DIR"], str(self.cache / "cache/torch_extensions"))
        self.assertNotIn("/venv/", actual["PATH"].split(os.pathsep)[0])

    def test_installer_does_not_create_venv_or_use_virtualenv_pip_flag(self):
        data = {"environments": {"lingbot1.3b": {
            "packages": {"torch": "2.6.0+cu124"}, "manual_distributions": [],
            "python": "3.10", "torch_cuda": "12.4", "compiler_cuda": "12.6"
        }}}
        args = SimpleNamespace(backbone="lingbot1.3b", python=None, env_mode="conda",
                               env_root=str(self.root / "cache"), dry_run=False,
                               portable=False, gpu=0,
                               torch_index_url="https://mirrors.aliyun.com/pytorch-wheels/cu124")
        report = installer.Report()
        def probe(command, **kwargs):
            flattened = [str(x) for x in command]
            if len(flattened) >= 2 and flattened[1] == "-I" and "import json,sys" in " ".join(flattened):
                return self.conda_identity(command)
            if flattened[-1].startswith("import sys;print("):
                return "3.10"
            if "pip" in flattened and "--version" in flattened:
                return "pip 25.0"
            if any("import importlib.metadata" in item for item in flattened):
                return "{}"
            raise AssertionError("unexpected probe: " + repr(flattened))
        with patch.object(installer, "source_checks"), \
             patch.object(installer, "hardware_checks"), \
             patch.object(installer, "probe_python"), \
             patch.object(installer.subprocess, "check_output", side_effect=probe), \
             patch.object(installer, "run") as pip_run:
            installer.setup(args, data, report, self.cache)
        self.assertFalse((self.cache / "venv").exists())
        self.assertEqual(pip_run.call_count, 1)
        call = [str(x) for x in pip_run.call_args.args[0]]
        self.assertEqual(call[0], str(self.python))
        self.assertNotIn("--require-virtualenv", call)
        self.assertIn("torch==2.6.0+cu124", call)
        self.assertIn("https://mirrors.aliyun.com/pytorch-wheels/cu124", call)
        self.assertEqual(json.loads((self.cache / "runtime-environment.json").read_text())["mode"],
                         "conda")


if __name__ == "__main__":
    unittest.main()
