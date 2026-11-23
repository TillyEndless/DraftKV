"""Offline tests: end-user facade must fail closed and avoid implicit method/config rewrites."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("quickstart", ROOT / "scripts/draftkv.py")
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class UserCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.venv = self.home / "env/lingbot1.3b/venv/bin/python"
        self.venv.parent.mkdir(parents=True)
        self.venv.write_bytes(b"test interpreter for offline CLI only")
        self.art = self.home / "art"
        self.art.mkdir()
        self.prepared = self.home / "prepared-plan.json"
        self.prepared.write_text(json.dumps(dict(backbone="LingBot1.3B", request={}, algorithm_plugin={})))
        self.output = self.home / "freshrun/native.mp4"

    def args_run(self, *extra):
        return cli.make_parser().parse_args(["run", "--backbone", "lingbot1.3b",
                                             "--env-root", str(self.home / "env"),
                                             "--input", str(self.prepared),
                                             "--output", str(self.output), *extra])

    def args_install(self, *extra):
        return cli.make_parser().parse_args(["install", "--backbone", "lingbot1.3b",
                                             "--env-root", str(self.home / "env"),
                                             "--asset-root", str(self.art),
                                             "--cuda-home", "/usr/local/cuda-12.6",
                                             *extra])

    def test_model_environment_path_keeps_venv_bin_not_python_target(self):
        target = self.home / "system-python"
        target.write_bytes(b"system python fixture")
        self.venv.unlink()
        self.venv.symlink_to(target)
        env = cli.env_for(self.venv)
        self.assertEqual(env["PATH"].split(":")[0], str(self.venv.parent))
        self.assertNotEqual(env["PATH"].split(":")[0], str(target.parent))

    def test_install_and_run_share_exact_portable_cache_contract(self):
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("installer_fixture", ROOT / "scripts/_install.py")
        installer = module_from_spec(spec)
        spec.loader.exec_module(installer)
        base = self.venv.parent.parent.parent
        # On macOS /var is an alias of /private/var; compare canonical targets.
        expected = installer.install_env_vars(base.resolve())
        actual = cli.env_for(self.venv)
        for name in cli.PORTABLE_CACHE_LAYOUT:
            with self.subTest(variable=name):
                self.assertEqual(Path(actual[name]).resolve(), Path(expected[name]).resolve())
        self.assertEqual(actual["PYTHONNOUSERSITE"], "1")
        self.assertEqual(actual["PIP_CONFIG_FILE"], expected["PIP_CONFIG_FILE"])
        self.assertEqual(actual["PIP_EXTRA_INDEX_URL"], "")
        self.assertFalse((base / "cache").exists(), "read-only environment lookup must not create cache")

    def test_runtime_cache_symlink_escape_refused_before_gpu_work(self):
        base = self.venv.parent.parent.parent
        (base / "cache").mkdir()
        outside = self.home / "outside_cache"
        outside.mkdir()
        (base / "cache/triton").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "runtime cache escapes"):
            cli.env_for(self.venv)

    def test_runtime_receipt_preserves_headers_cuda_and_cache_paths(self):
        base = self.venv.parent.parent.parent
        receipt = base / "native/build-env.json"
        receipt.parent.mkdir()
        include = self.home / "python-dev/include"
        include.mkdir(parents=True)
        toolkit = self.home / "cuda"
        (toolkit / "bin").mkdir(parents=True)
        (toolkit / "bin/nvcc").write_text("fixture")
        caches = cli.cache_paths_for(self.venv)
        receipt.write_text(json.dumps(dict(schema_version=1, python=str(self.venv),
            python_includes=[str(include)],cuda_home=str(toolkit),
            torch_extensions_dir=caches["TORCH_EXTENSIONS_DIR"],
            runtime_cache_paths=caches)))
        env = cli.env_for(self.venv)
        self.assertEqual(env["DRAFTKV_PREPACK_BUILD"], caches["DRAFTKV_PREPACK_BUILD"])
        self.assertEqual(env["TRITON_CACHE_DIR"], caches["TRITON_CACHE_DIR"])
        self.assertEqual(env["CPATH"].split(":" )[0], str(include))
        self.assertEqual(env["CUDA_HOME"], str(toolkit))
        self.assertEqual(env["PATH"].split(":")[0], str(toolkit / "bin"))

    def test_runtime_receipt_rejects_mismatched_cache_contract(self):
        base = self.venv.parent.parent.parent
        receipt = base / "native/build-env.json"
        receipt.parent.mkdir()
        caches = cli.cache_paths_for(self.venv)
        forged = dict(caches, TRITON_CACHE_DIR=str(self.home / "different"))
        receipt.write_text(json.dumps(dict(schema_version=1, python=str(self.venv),
            python_includes=[],cuda_home="/fake/toolkit",
            torch_extensions_dir=caches["TORCH_EXTENSIONS_DIR"],
            runtime_cache_paths=forged)))
        with self.assertRaisesRegex(ValueError, "cache identity disagree"):
            cli.env_for(self.venv)

    def test_four_selectable_backbones(self):
        self.assertEqual(len(cli.BACKBONES), 4)
        for name in cli.BACKBONES:
            self.assertEqual(cli.make_parser().parse_args(
                ["run", "--backbone", name, "--input", str(self.prepared),
                 "--output", str(self.output)]).backbone, name)

    def test_each_model_install_dry_run(self):
        for name in cli.BACKBONES:
            with self.subTest(backbone=name),patch.object(cli, "commands") as calls:
                args = cli.make_parser().parse_args([
                    "install", "--backbone", name, "--python", "python3.10",
                    "--env-root", str(self.home / "env"),
                    "--asset-root", str(self.art),
                    "--cuda-home", "/usr/local/cuda-12.6", "--dry-run"])
                cli.install(args)
                calls.assert_not_called()

    def test_install_requests_audited_portable_dependency_resolution(self):
        code=(cli.ROOT/"scripts/draftkv.py").read_text()
        self.assertIn('"--python", args.python, "--portable"]', code)
        helper=(cli.ROOT/"scripts/_install.py").read_text()
        self.assertIn('"--upgrade-strategy", "only-if-needed"', helper)
        self.assertIn('"pip", "check"', helper)
        self.assertIn('"CURRENT_PORTABLE_ENVIRONMENT_NOT_HISTORICAL"', helper)

    def test_install_cannot_succeed_without_required_native_sources(self):
        # Refuse before creating a venv, downloading weights, or touching the GPU.
        with patch.object(cli, "commands") as launch:
            with self.assertRaisesRegex(ValueError, "dependency-root"):
                cli.install(self.args_install())
            launch.assert_not_called()

    def test_install_missing_model_assets_is_a_failure(self):
        with patch.object(cli, "commands") as launch:
            args = cli.make_parser().parse_args(["install","--backbone","lingbot1.3b",
                "--cuda-home","/usr/local/cuda-12.6",
                "--dependency-root",str(self.home/"private")])
            with self.assertRaisesRegex(ValueError, "asset-root"):
                cli.install(args)
            launch.assert_not_called()

    def test_install_cannot_claim_success_if_fa3_compilation_disabled(self):
        with patch.object(cli, "commands") as launch:
            with self.assertRaisesRegex(ValueError, "requires FA3 compilation"):
                cli.install(self.args_install("--dependency-root", str(self.home/"private"),
                                              "--no-build-fa3"))
            launch.assert_not_called()

    def test_each_model_prepare_option_is_source_scoped(self):
        for name in cli.BACKBONES:
            with self.subTest(backbone=name):
                args = cli.make_parser().parse_args([
                    "prepare", "--backbone", name, "--env-root", str(self.home / "env"),
                    "--dataset-root", str(self.art), "--case-id", "case_0001",
                    "--output", str(self.home / "cases")])
                self.assertEqual(args.backbone,name)
                self.assertEqual(args.case_id,"case_0001")

    def test_prepare_with_missing_orbia_lock_fails_before_running(self):
        fake_python=self.home/"env/lingbot14b/venv/bin/python"
        fake_python.parent.mkdir(parents=True,exist_ok=True)
        fake_python.write_bytes(b"fake worker interpreter")
        args = cli.make_parser().parse_args(["prepare","--backbone","lingbot14b",
            "--env-root",str(self.home/"env"),"--dataset-root",str(self.art),
            "--index",str(self.prepared),"--index-sha256","a"*64,
            "--case-id","case_0002","--output",str(self.home/"ready")])
        with patch.object(cli,"commands") as execution:
            with self.assertRaisesRegex(ValueError, "original ORBIA source lock absent"):
                cli.prepare_inputs(args)
            execution.assert_not_called()

    def test_prepare_invokes_frozen_main_explicitly(self):
        code = (cli.ROOT / "scripts/draftkv.py").read_text()
        self.assertIn("from draftkv.benchmarks.orbia_requests import main;", code)
        self.assertIn("from draftkv.backbones.bench1_inputs import main;", code)
        self.assertNotIn('"-m", "draftkv.benchmarks.orbia_requests"', code)
        self.assertNotIn('"-m", "draftkv.backbones.bench1_inputs"', code)

    def test_install_dry_run_has_zero_side_effects(self):
        with patch.object(cli, "commands") as run:
            cli.install(self.args_install("--dry-run"))
        run.assert_not_called()

    def test_run_dry_run_does_not_bind_or_touch_gpu(self):
        cache = self.home / "env/lingbot1.3b/native/b01/cache/one"
        cache.mkdir(parents=True)
        with patch.object(cli, "find_receipt",
                          return_value=(cache.parent.parent / "build-spec.json",cache)):
            with patch.object(cli,"gpu_uuid") as gpu, patch.object(cli,"commands") as run:
                cli.run(self.args_run("--dry-run"))
                gpu.assert_not_called()
                run.assert_not_called()
        self.assertFalse(self.output.parent.exists())

    def test_prepared_plan_required(self):
        self.prepared.unlink()
        with self.assertRaisesRegex(ValueError, "source-locked request/plan missing"):
            cli.run(self.args_run())

    def test_run_binds_and_launches_builtin_once(self):
        out = self.output
        def simulate(argv, env=None, capture=False):
            if "fresh_binding.py" in str(argv[1]):
                out.parent.mkdir(parents=True)
                (out.parent / "plan.json").write_text("{}")
            elif "portable_run.sh" in str(argv[0]):
                out.write_bytes(b"video")
            else:
                raise AssertionError("unexpected command")
        with patch.object(cli,"find_receipt",
                          return_value=(self.home/"native/build-spec.json",
                                        self.home/"native/cache/key")), \
             patch.object(cli,"gpu_uuid", return_value="GPU-valid-uuid"), \
             patch.object(cli,"commands", side_effect=simulate) as run:
            cli.run(self.args_run("--profile","summary"))
            self.assertEqual(run.call_count, 2)
            actual = run.call_args_list[-1].args[0]
            self.assertIn("--builtin-worker", actual)
            self.assertIn("--portable-evidence",actual)
            self.assertEqual(actual[actual.index("--profile")+1],"portable-summary")

    def test_startup_only_does_not_demand_video_or_profile_admission(self):
        out = self.output
        def simulate(argv, env=None, capture=False):
            if "fresh_binding.py" in str(argv[1]):
                out.parent.mkdir(parents=True)
                (out.parent / "plan.json").write_text("{}")
            elif "portable_run.sh" in str(argv[0]):
                self.assertIn("--native-startup-only", argv)
                receipt = out.parent / "traces/portable_native_startup_rank0.json"
                receipt.parent.mkdir()
                receipt.write_text(json.dumps({"status":"NATIVE_EXTENSION_READY",
                                                "backbone":"LingBot1.3B"}))
            else:
                raise AssertionError("unexpected command")
        with patch.object(cli, "find_receipt",
                          return_value=(self.home/"native/build-spec.json",
                                        self.home/"native/cache/key")), \
             patch.object(cli,"gpu_uuid", return_value="GPU-valid-uuid"), \
             patch.object(cli,"commands", side_effect=simulate) as called:
            cli.run(self.args_run("--startup-only", "--profile", "off"))
            self.assertEqual(called.call_count, 2)
        self.assertFalse(out.exists())

    def test_startup_only_rejects_formal_profile_before_gpu(self):
        with patch.object(cli, "gpu_uuid") as gpu:
            with self.assertRaisesRegex(ValueError, "does not produce an admitted"):
                cli.run(self.args_run("--startup-only", "--profile", "summary"))
            gpu.assert_not_called()

    def test_run_refuses_overwrite(self):
        self.output.parent.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "already exists"):
            cli.run(self.args_run())

    def test_quickstart_disallows_unknown_algorithm(self):
        with self.assertRaises(SystemExit):
            cli.make_parser().parse_args(["run","--backbone","native",
                                          "--input",str(self.prepared),
                                          "--output",str(self.output)])

    def test_install_rejects_incomplete_offline_fa2_override_before_side_effects(self):
        args = self.args_install("--dependency-root", str(self.home / "original"),
                                 "--fa2-wheel", str(self.home / "missing.whl"))
        with patch.object(cli, "commands") as execution:
            with self.assertRaisesRegex(ValueError, "fa2-wheel"):
                cli.install(args)
            execution.assert_not_called()

    def test_official_fa2_release_locks_cover_all_backbones(self):
        for backbone in cli.BACKBONES:
            with self.subTest(backbone=backbone):
                lock = cli.pinned_official_fa2_wheel(backbone)
                self.assertEqual(len(lock["sha256"]), 64)
                self.assertIn("Dao-AILab/flash-attention/releases/download", lock["url"])
                suffix = "cxx11abiTRUE" if backbone == "longlive1.3b" else "cxx11abiFALSE"
                self.assertIn(suffix + "-cp310-cp310-linux_x86_64.whl", lock["filename"])
                self.assertGreater(lock["bytes"], 100_000_000)
        self.assertEqual(cli.pinned_official_fa2_wheel("longlive1.3b")["version"],
                         "2.8.3.post1")
        self.assertEqual(cli.pinned_official_fa2_wheel("lingbot1.3b")["version"],
                         "2.8.3")

    def test_default_install_dry_run_does_not_require_local_fa2_wheel(self):
        args = self.args_install("--dry-run")
        with patch.object(cli, "commands") as commands_called, \
             patch.object(cli.urllib.request, "urlopen") as download:
            cli.install(args)
        commands_called.assert_not_called()
        download.assert_not_called()

    def test_official_wheel_download_verified_and_cached_without_network_reuse(self):
        import hashlib
        import io
        content = b"verified sample fa2 wheel contents"
        filename = "flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
        asset = dict(version="2.8.3", filename=filename,
                     url="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
                         + filename.replace("+", "%2B"),
                     sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
        args = self.args_install()
        base = self.home / "env/lingbot1.3b"
        with patch.object(cli, "pinned_official_fa2_wheel", return_value=asset), \
             patch.object(cli, "verify_official_fa2_abi") as abi, \
             patch.object(cli.urllib.request, "urlopen", side_effect=lambda *a, **kw: io.BytesIO(content)) as dl:
            wheel, digest = cli.obtain_fa2_wheel(args, self.venv, base)
            self.assertEqual(wheel.read_bytes(), content)
            self.assertEqual(digest, asset["sha256"])
            wheel2, digest2 = cli.obtain_fa2_wheel(args, self.venv, base)
            self.assertEqual((wheel2, digest2), (wheel, digest))
            self.assertEqual(dl.call_count, 1)
            self.assertEqual(abi.call_count, 2)
        self.assertFalse(list(wheel.parent.glob(".fa2-*")))

    def test_official_wheel_rejects_bad_hash_without_caching(self):
        import io
        filename = "flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
        asset = dict(version="2.8.3", filename=filename,
                     url="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
                         + filename.replace("+", "%2B"),
                     sha256="0" * 64, bytes=len(b"corrupt wheel"))
        base = self.home / "env/lingbot1.3b"
        with patch.object(cli, "pinned_official_fa2_wheel", return_value=asset), \
             patch.object(cli, "verify_official_fa2_abi"), \
             patch.object(cli.urllib.request, "urlopen",
                          side_effect=lambda *a, **kw: io.BytesIO(b"corrupt wheel")):
            with self.assertRaisesRegex(ValueError, "SHA-256/size mismatch"):
                cli.obtain_fa2_wheel(self.args_install(), self.venv, base)
        self.assertEqual(list((base/"cache/verified_wheels").iterdir()), [])

    def test_official_fa2_binary_abi_is_checked_before_download(self):
        facts = dict(python=[3, 10], torch="2.6.0+cu124", cuda="12.4", cxx11abi=True)
        with patch.object(cli.platform, "system", return_value="Linux"), \
             patch.object(cli.platform, "machine", return_value="x86_64"), \
             patch.object(cli, "commands", return_value=SimpleNamespace(stdout=json.dumps(facts))):
            with self.assertRaisesRegex(ValueError, "cxx11abiFALSE"):
                cli.verify_official_fa2_abi(self.venv, "lingbot1.3b")
        facts["cxx11abi"] = False
        with patch.object(cli.platform, "system", return_value="Linux"), \
             patch.object(cli.platform, "machine", return_value="x86_64"), \
             patch.object(cli, "commands", return_value=SimpleNamespace(stdout=json.dumps(facts))):
            cli.verify_official_fa2_abi(self.venv, "lingbot1.3b")
        facts.update(torch="2.8.0+cu128", cuda="12.8", cxx11abi=True)
        with patch.object(cli.platform, "system", return_value="Linux"), \
             patch.object(cli.platform, "machine", return_value="x86_64"), \
             patch.object(cli, "commands", return_value=SimpleNamespace(stdout=json.dumps(facts))):
            cli.verify_official_fa2_abi(self.venv, "longlive1.3b")

    def test_fa2_wheel_hash_and_name_are_mandatory(self):
        wheel = self.home / "flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl"
        wheel.write_bytes(b"test-wheel-with-audited-name")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            cli.verify_user_fa2_wheel(str(wheel), "0" * 64)
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            cli.verify_user_fa2_wheel(str(wheel), "bad")
        import hashlib
        self.assertEqual(cli.verify_user_fa2_wheel(str(wheel),
            hashlib.sha256(wheel.read_bytes()).hexdigest()), wheel.resolve())

    def test_runtime_receipt_propagates_cuda_and_python_headers_into_jit(self):
        # The worker JIT has a separate C++ compilation step from B01 install.
        py = self.venv
        base = py.parent.parent.parent
        native = base / "native"
        native.mkdir(exist_ok=True)
        hdr = self.home / "python3.10"
        hdr.mkdir()
        cuda = self.home / "cuda"
        (cuda / "bin").mkdir(parents=True)
        (cuda / "bin/nvcc").write_text("fake compiler for offline test")
        receipt = {"schema_version": 1, "python": str(py),
            "python_includes": [str(hdr)], "cuda_home": str(cuda),
            "torch_extensions_dir": str(base / "cache/torch_extensions")}
        (native / "build-env.json").write_text(json.dumps(receipt))
        with patch.dict(cli.os.environ, {"CPATH": "/other/include"}):
            env = cli.env_for(py)
        self.assertEqual(env["CUDA_HOME"], str(cuda))
        self.assertEqual(env["CPATH"].split(":"), [str(hdr), "/other/include"])
        self.assertEqual(env["TORCH_EXTENSIONS_DIR"], receipt["torch_extensions_dir"])
        self.assertTrue(env["PATH"].startswith(str(cuda / "bin") + ":" + str(py.parent)))

    def test_python_headers_require_pyconfig_h_as_well_as_python_h(self):
        root = self.home / "headers"
        pyh = root / "python3.10"
        pyh.mkdir(parents=True)
        (pyh / "Python.h").write_text("fixture")
        def python_probe(parts, capture=False, env=None):
            return SimpleNamespace(stdout=json.dumps({"python": [3, 10], "include": "/not-present"}))
        with patch.object(cli, "commands", side_effect=python_probe):
            with self.assertRaisesRegex(ValueError, "pyconfig.h"):
                cli.resolve_python_headers(self.venv, str(root))
            config = root / "x86_64-linux-gnu/python3.10"
            config.mkdir(parents=True)
            (config / "pyconfig.h").write_text("fixture")
            self.assertEqual(cli.resolve_python_headers(self.venv, str(root)),
                             [str(pyh.resolve()), str(config.resolve())])

    def test_portable_extras_include_qwen_video_and_lingbot_native_imports(self):
        self.assertEqual(cli.PORTABLE_EXTRAS["solarwm33b"]["Pillow"], "9.5.0")
        self.assertEqual(cli.PORTABLE_EXTRAS["solarwm33b"]["imageio-ffmpeg"], "0.6.0")
        self.assertIn("solarwm33b", cli.TORCHVISION_CU124)
        self.assertIn("lingbot14b", cli.TORCHVISION_CU124)
        self.assertEqual(cli.PORTABLE_EXTRAS["lingbot14b"]["ftfy"], "6.3.1")
        self.assertEqual(cli.PORTABLE_EXTRAS["longlive1.3b"]["scipy"], "1.15.3")
        self.assertEqual(cli.PORTABLE_EXTRAS["longlive1.3b"]["imageio-ffmpeg"], "0.6.0")

    def test_authorized_native_input_materials_are_shipped_byte_identical(self):
        # Compare to the precise independently verified authorized SHA values.
        import hashlib
        expected = {
           "versions/h200_large/configs/datasets/orbia_native_policy.json":
               "a98ae3a5dfd86c90c7d504f0588e736af1368c896be5a8aa49ebf3ed7bfe7953",
           "versions/lingbot13/configs/datasets/orbia_native_policy.json":
               "a98ae3a5dfd86c90c7d504f0588e736af1368c896be5a8aa49ebf3ed7bfe7953",
           "versions/longlive/configs/backbones/bench1_native_inputs.receipt.json":
               "608d43b6cd059b991bc097aff41b763d652abc4b1762ff74e8c4af6cdead4b72",
           "versions/longlive/scripts/bench1_prepare_inputs.py":
               "f12661f8a35cc4fff7a8d9dd22c9d6130d6a78e1bee0068d708721a6041ab0d7",
        }
        for filename, expected_sha in expected.items():
            with self.subTest(filename=filename):
                self.assertEqual(hashlib.sha256((ROOT / filename).read_bytes()).hexdigest(), expected_sha)


if __name__ == "__main__":
    unittest.main()
