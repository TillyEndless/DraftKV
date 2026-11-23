"""CPU-only tests of installation/provisioning, with tiny isolated fixtures."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("install_helpers", ROOT / "scripts/_install.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
DATA = json.loads(helper.MANIFEST.read_text())


class InstallScripts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="draftkv-install-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.asset = {"id": "tiny", "path": "models/model.bin", "bytes": 7,
                      "sha256": hashlib.sha256(b"fixture").hexdigest(),
                      "url": "https://example.invalid/pinned/model.bin", "license": "test fixture"}
        self.report = helper.Report()

    def cli(self, script, *args):
        return subprocess.run([str(ROOT / "scripts" / script), *args], capture_output=True, text=True)

    def provision(self, **options):
        with contextlib.redirect_stdout(io.StringIO()):
            helper.provision_file(self.asset, self.root, self.report, **options)

    def test_three_shell_entrypoints(self):
        for script in ("setup_env.sh", "prepare_assets.sh", "verify_install.sh"):
            r = self.cli(script, "--help")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("--backbone", r.stdout)

    def test_four_distinct_environment_selections_without_writes(self):
        for backbone, env in DATA["environments"].items():
            dest = self.root / backbone
            r = self.cli("setup_env.sh", "--backbone", backbone, "--env-root", str(dest), "--dry-run")
            self.assertEqual(r.returncode, 0, r.stderr)
            for name, version in helper.selected_packages(env).items():
                self.assertIn(f"{name}: {version}", r.stdout)
            self.assertIn(str(dest / backbone / "venv"), r.stdout)
            self.assertIn("MANUAL_REQUIRED", r.stdout)
            self.assertFalse(dest.exists())

    def test_all_and_component_asset_selection(self):
        for backbone in DATA["environments"]:
            for category in ("all", "models", "datasets", "evaluator"):
                r = self.cli("prepare_assets.sh", "--backbone", backbone, "--" + category,
                             "--asset-root", str(self.root / "assets"), "--dry-run")
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertIn("OVERALL", r.stdout)
        self.assertFalse((self.root / "assets").exists())

    def test_asset_root_and_selection_required(self):
        for args in (("--models",), ("--asset-root", str(self.root))):
            r = self.cli("prepare_assets.sh", "--backbone", "lingbot14b", *args)
            self.assertNotEqual(r.returncode, 0)

    def test_unknown_backbone_rejected(self):
        r = self.cli("setup_env.sh", "--backbone", "other", "--dry-run")
        self.assertNotEqual(r.returncode, 0)

    def test_existing_valid_file_skips_network(self):
        p = self.root / self.asset["path"]
        p.parent.mkdir()
        p.write_bytes(b"fixture")
        with patch.object(helper, "run") as download:
            self.provision(download=True)
        download.assert_not_called()
        self.assertEqual(self.report.statuses, ["PASS"])

    def test_existing_wrong_sha_preserved(self):
        p = self.root / self.asset["path"]
        p.parent.mkdir()
        p.write_bytes(b"corrupt")
        with patch.object(helper, "run") as download:
            self.provision(download=True)
        download.assert_not_called()
        self.assertEqual(p.read_bytes(), b"corrupt")
        self.assertEqual(self.report.statuses, ["FAIL"])

    def test_correct_download_published_and_reused(self):
        p = self.root / self.asset["path"]
        def download(cmd, **kwargs):
            self.assertIn("--continue-at", cmd)
            self.assertEqual(cmd[cmd.index("--proto") + 1], "=https")
            Path(cmd[cmd.index("--output") + 1]).write_bytes(b"fixture")
        with patch.object(helper, "run", side_effect=download) as call:
            self.provision(download=True)
            self.provision(download=True)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(p.read_bytes(), b"fixture")
        self.assertEqual(self.report.statuses, ["PASS", "PASS"])

    def test_bad_download_sha_not_published(self):
        def download(cmd, **kwargs):
            Path(cmd[cmd.index("--output") + 1]).write_bytes(b"badbody")
        with patch.object(helper, "run", side_effect=download):
            self.provision(download=True)
        self.assertFalse((self.root / self.asset["path"]).exists())
        self.assertEqual(self.report.statuses, ["FAIL"])

    def test_interrupted_download_can_resume_without_retry_loop(self):
        part = self.root / (self.asset["path"] + ".part")
        def interrupted(cmd, **kwargs):
            part.write_bytes(b"fix")
            raise subprocess.CalledProcessError(18, cmd)
        with patch.object(helper, "run", side_effect=interrupted) as call:
            self.provision(download=True)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(part.read_bytes(), b"fix")
        self.assertFalse((self.root / self.asset["path"]).exists())
        def resume(cmd, **kwargs):
            self.assertEqual(part.read_bytes(), b"fix")
            part.write_bytes(b"fixture")
        with patch.object(helper, "run", side_effect=resume):
            self.provision(download=True)
        self.assertEqual((self.root / self.asset["path"]).read_bytes(), b"fixture")

    def test_concurrent_valid_target_never_overwritten(self):
        target = self.root / self.asset["path"]
        def raced(cmd, **kwargs):
            target.write_bytes(b"fixture")
            Path(cmd[cmd.index("--output") + 1]).write_bytes(b"fixture")
        with patch.object(helper, "run", side_effect=raced):
            self.provision(download=True)
        self.assertEqual(target.read_bytes(), b"fixture")
        self.assertEqual(self.report.statuses, ["FAIL"])  # no-clobber link refuses race

    def test_gated_assets_require_authorization_and_standard_login(self):
        self.asset.update(gated=True, url="https://huggingface.co/org/gated/resolve/rev/file")
        with patch.object(helper, "run") as download, patch.object(helper, "hf_token", return_value=None):
            self.provision(download=True)
            self.provision(download=True, gated=True)
        download.assert_not_called()
        self.assertEqual(self.report.statuses, ["MANUAL_REQUIRED", "MANUAL_REQUIRED"])

    def test_hf_token_not_exposed_in_command_or_output(self):
        token = "installation-test-placeholder"
        self.asset.update(gated=True, url="https://huggingface.co/org/gated/resolve/rev/file")
        def download(cmd, **kwargs):
            self.assertNotIn(token, " ".join(cmd))
            self.assertIn(token, kwargs["input"])
            Path(cmd[cmd.index("--output") + 1]).write_bytes(b"fixture")
        with patch.object(helper, "hf_token", return_value=token), patch.object(helper, "run", side_effect=download):
            self.provision(download=True, gated=True)
        self.assertEqual(self.report.statuses, ["PASS"])

    def test_missing_public_source_is_manual(self):
        self.asset["url"] = None
        with patch.object(helper, "run") as download:
            self.provision(download=True)
        download.assert_not_called()
        self.assertEqual(self.report.statuses, ["MANUAL_REQUIRED"])

    def test_source_root_and_symlink_into_source_rejected(self):
        r = self.cli("prepare_assets.sh", "--backbone", "longlive1.3b", "--models",
                     "--asset-root", str(ROOT / "uncreated-assets"), "--dry-run")
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((ROOT / "uncreated-assets").exists())
        (self.root / "models").symlink_to(ROOT / "versions", target_is_directory=True)
        with self.assertRaises(ValueError):
            self.provision(download=True)

    def test_partial_symlink_escape_rejected_before_write(self):
        other = self.root / "other"
        other.mkdir()
        assets = self.root / "assets"
        (assets / "models").mkdir(parents=True)
        (assets / "models/model.bin.part").symlink_to(other / "original")
        with self.assertRaises(ValueError):
            helper.provision_file(self.asset, assets, self.report, download=True)
        self.assertFalse((other / "original").exists())

    def test_traversal_and_absolute_asset_path_rejected(self):
        for path in ("../escape", str(self.root / "absolute")):
            self.asset["path"] = path
            with self.assertRaises(ValueError):
                self.provision(download=True)

    def test_verify_missing_dependencies_exits_nonzero_without_generation(self):
        r = self.cli("verify_install.sh", "--backbone", "lingbot14b", "--python",
                     str(self.root / "missing-python"), "--env-root", str(self.root / "envs"),
                     "--asset-root", str(self.root / "assets"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("FAIL environment Python", r.stdout)
        self.assertIn("GPU generation/performance NOT_CHECKED", r.stdout)
        self.assertFalse((self.root / "envs").exists())

    def test_verification_scopes_separate_inference_from_scoring(self):
        for scope, needs_native, needs_evaluator, assets in (
            ("inference", False, False, ["models"]),
            ("draftkv", True, False, ["models"]),
            ("benchmark", True, True, ["models", "datasets", "evaluator"])):
            with self.subTest(scope=scope):
                with patch.object(helper, "source_checks") as source, \
                     patch.object(helper, "hardware_checks"), \
                     patch.object(helper, "probe_python"), \
                     patch.object(helper, "native_checks") as native, \
                     patch.object(helper, "evaluator_checks") as evaluator, \
                     patch.object(helper, "asset_checks") as assets_called, \
                     contextlib.redirect_stdout(io.StringIO()):
                    result=helper.main(["verify", "--backbone", "lingbot1.3b",
                                        "--scope",scope,"--env-root",str(self.root / "env"),
                                        "--asset-root",str(self.root / "assets")])
                    self.assertEqual(result,0)
                    self.assertEqual(source.call_count,1)
                    self.assertEqual(native.called,needs_native)
                    self.assertEqual(evaluator.called,needs_evaluator)
                    self.assertEqual(assets_called.call_args.args[3],assets)

    def test_direct_asset_download_uses_hf_auth_automatically(self):
        with patch.object(helper, "asset_checks") as called, \
             contextlib.redirect_stdout(io.StringIO()):
            result = helper.main([
                "assets", "--backbone", "solarwm33b",
                "--asset-root", str(self.root / "assets"), "--models"])
        self.assertEqual(result, 0)
        self.assertTrue(called.call_args.kwargs["gated"])
        self.assertTrue(called.call_args.kwargs["download"])

    def test_hf_gated_asset_dry_run_requires_no_authorization_flag(self):
        with patch.object(helper, "asset_checks") as called, \
             contextlib.redirect_stdout(io.StringIO()):
            result = helper.main([
                "assets", "--backbone", "solarwm33b",
                "--asset-root", str(self.root / "assets"), "--models", "--dry-run"])
        self.assertEqual(result, 0)
        self.assertTrue(called.call_args.kwargs["gated"])
        self.assertTrue(called.call_args.kwargs["dry"])

    def test_setup_rejects_verification_only_scope(self):
        r=self.cli("setup_env.sh", "--backbone","lingbot1.3b",
                   "--scope","inference", "--dry-run")
        self.assertNotEqual(r.returncode,0)

    def test_original_source_and_configuration_identity(self):
        with contextlib.redirect_stdout(io.StringIO()):
            helper.source_checks(DATA, self.report)
        self.assertEqual(self.report.statuses, ["PASS"])

    def test_source_corruption_is_detected(self):
        (self.root / "versions/v").mkdir(parents=True)
        (self.root / "versions/v/config.json").write_bytes(b"changed")
        (self.root / "RETAINED_SOURCE_SHA256.json").write_text(json.dumps([
            {"version": "v", "path": "config.json", "sha256": hashlib.sha256(b"original").hexdigest()}]))
        with patch.object(helper, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            helper.source_checks({"protected_files": {}}, self.report)
        self.assertEqual(self.report.statuses, ["FAIL"])

    def test_setup_uses_only_pinned_binary_packages_and_original_indexes(self):
        for backbone, env in DATA["environments"].items():
            with self.subTest(backbone=backbone):
                args = SimpleNamespace(backbone=backbone, dry_run=False, python=sys.executable,
                                       env_root=str(self.root))
                base = self.root / backbone
                (base / "venv/bin").mkdir(parents=True)
                (base / "venv/pyvenv.cfg").write_text("include-system-site-packages = false\n")
                outputs = [env["python"] + "\n", env["python"] + "\n", "pip fixture\n", "{}"]
                with patch.object(helper, "source_checks"), patch.object(helper, "hardware_checks"), \
                     patch.object(helper, "probe_python"), patch.object(helper.subprocess, "check_output", side_effect=outputs), \
                     patch.object(helper, "run") as install, contextlib.redirect_stdout(io.StringIO()):
                    helper.setup(args, DATA, self.report, base)
                calls = [call.args[0] for call in install.call_args_list]
                selected = helper.selected_packages(env)
                expected = {f"{name}=={version}" for name, version in selected.items()
                            if name not in env["manual_distributions"]}
                requested = [next(str(arg) for arg in command if "==" in str(arg)) for command in calls]
                self.assertEqual(len(calls), len(expected))
                self.assertEqual(set(requested), expected)
                for command, requirement in zip(calls, requested):
                    self.assertIn("--no-deps", command)
                    self.assertIn("--only-binary=:all:", command)
                    package = requirement.split("==", 1)[0]
                    self.assertNotIn(package, env["manual_distributions"])
                    index = ("https://download.pytorch.org/whl/cu" + env["torch_cuda"].replace(".", "")
                             if helper.package_key(package) in ("torch", "torchvision") else "https://pypi.org/simple")
                    self.assertIn(index, command)

    def test_asset_manifest_has_no_mutable_model_revisions(self):
        for asset in DATA["assets"]:
            self.assertRegex(asset["sha256"], r"^[a-f0-9]{64}$")
            self.assertNotIn("/resolve/main/", asset.get("url") or "")
            self.assertNotIn("/main/", asset.get("url") or "")
            helper.child(self.root, asset["path"])

    def test_only_required_model_files_for_each_backbone(self):
        counts = {"longlive1.3b": 9, "lingbot1.3b": 13, "lingbot14b": 16, "solarwm33b": 75}
        for backbone, count in counts.items():
            assets = [a for a in DATA["assets"] if backbone in a["backbones"] and a["category"] == "models"]
            self.assertEqual(len(assets), count)
            self.assertTrue(all(len(a["backbones"]) == len(set(a["backbones"])) for a in assets))
            self.assertFalse(any("teaser" in a["path"] for a in assets))
            if backbone == "lingbot1.3b":
                self.assertFalse(any("models/lingbot14/transformers/" in a["path"] for a in assets))


    def test_shell_entrypoints_work_from_foreign_cwd_and_paths_with_spaces(self):
        scripts = self.root / "checkout with spaces/scripts"
        scripts.mkdir(parents=True)
        for name in ("setup_env.sh", "prepare_assets.sh", "verify_install.sh", "_install.py", "install_manifest.json"):
            shutil.copy2(ROOT / "scripts" / name, scripts / name)
        bootstrap = self.root / "bootstrap python"
        bootstrap.symlink_to(sys.executable)
        foreign = self.root / "unrelated cwd"
        foreign.mkdir()
        env = dict(os.environ, DRAFTKV_INSTALL_PYTHON=str(bootstrap))
        for name in ("setup_env.sh", "prepare_assets.sh", "verify_install.sh"):
            with self.subTest(script=name):
                result = subprocess.run([str(scripts / name), "--help"], cwd=foreign,
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--backbone", result.stdout)
        assets = self.root / "asset destination with spaces"
        result = subprocess.run([str(scripts / "prepare_assets.sh"), "--backbone", "lingbot14b",
                                 "--models", "--asset-root", str(assets), "--dry-run"],
                                cwd=foreign, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(assets.exists())

    def test_invalid_cli_arguments_return_usage_error_without_writes(self):
        cases = [
            ("setup_env.sh", ("--dry-run",)),
            ("setup_env.sh", ("--backbone", "other", "--dry-run")),
            ("setup_env.sh", ("--backbone", "lingbot14b", "--unknown")),
            ("prepare_assets.sh", ("--backbone", "lingbot14b", "--models", "--all")),
            ("prepare_assets.sh", ("--backbone", "lingbot14b", "--models")),
        ]
        for script, args in cases:
            with self.subTest(script=script, args=args):
                result = self.cli(script, *args, "--env-root", str(self.root / "envs"))
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((self.root / "envs").exists())

    def test_complete_partial_is_published_without_network(self):
        target = self.root / self.asset["path"]
        target.parent.mkdir()
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(b"fixture")
        with patch.object(helper, "run") as download:
            self.provision(download=True)
            self.provision(download=True)
        download.assert_not_called()
        self.assertEqual(target.read_bytes(), b"fixture")
        self.assertEqual(target.stat().st_ino, partial.stat().st_ino)
        self.assertEqual(self.report.statuses, ["PASS", "PASS"])

    def test_corrupt_partial_range_failure_is_retained_without_retry(self):
        target = self.root / self.asset["path"]
        target.parent.mkdir()
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(b"wrong!!")
        with patch.object(helper, "run", side_effect=subprocess.CalledProcessError(33, ["curl"])) as download:
            self.provision(download=True)
        download.assert_called_once()
        self.assertFalse(target.exists())
        self.assertEqual(partial.read_bytes(), b"wrong!!")
        self.assertEqual(self.report.statuses, ["FAIL"])

    def test_offline_http_and_auth_download_failures_are_bounded_and_redacted(self):
        self.asset.update(gated=True, url="https://huggingface.co/org/gated/resolve/rev/file")
        token = "hf_test_secret_never_logged"
        for code in (6, 22, 28):  # DNS/offline, HTTP authentication, timeout
            with self.subTest(curl_code=code):
                report = helper.Report()
                output = io.StringIO()
                with patch.object(helper, "hf_token", return_value=token), \
                     patch.object(helper, "run", side_effect=subprocess.CalledProcessError(code, ["curl"])) as download, \
                     contextlib.redirect_stdout(output):
                    helper.provision_file(self.asset, self.root, report, download=True, gated=True)
                download.assert_called_once()
                self.assertEqual(report.statuses, ["FAIL"])
                self.assertNotIn(token, output.getvalue())
                self.assertFalse((self.root / self.asset["path"]).exists())

    def test_hf_token_can_be_read_from_standard_cache_without_output(self):
        home = self.root / "HF cache with spaces"
        home.mkdir()
        (home / "token").write_text(" hf_fixture_token\n")
        with patch.dict(os.environ, {"HF_HOME": str(home)}, clear=True):
            self.assertEqual(helper.hf_token(), "hf_fixture_token")
        explicit = self.root / "explicit token"
        explicit.write_text("hf_explicit_token\n")
        with patch.dict(os.environ, {"HF_TOKEN_PATH": str(explicit)}, clear=True):
            self.assertEqual(helper.hf_token(), "hf_explicit_token")

    def test_missing_toolkit_and_driver_are_reported_without_building(self):
        output = io.StringIO()
        with patch.object(helper.platform, "system", return_value="Linux"), \
             patch.object(helper.platform, "machine", return_value="x86_64"), \
             patch.object(helper.subprocess, "check_output", side_effect=FileNotFoundError("missing tool")), \
             patch.object(helper.ctypes, "CDLL", side_effect=OSError("missing driver")), \
             patch.object(helper.shutil, "which", return_value=None), \
             patch.object(helper, "run") as build, contextlib.redirect_stdout(output):
            helper.hardware_checks(DATA["environments"]["lingbot14b"], self.report)
        build.assert_not_called()
        self.assertIn("MANUAL_REQUIRED", self.report.statuses)
        self.assertIn("NOT_CHECKED", self.report.statuses)
        self.assertIn("install original toolchain", output.getvalue())

    def test_native_abi_failure_is_fail_closed_without_compilation(self):
        native = self.root / "native libraries"
        native.mkdir()
        library = native / "draftkv_fixture_route.so"
        library.write_bytes(b"fixture")
        env = {"original_native_sha256": {"route": hashlib.sha256(b"fixture").hexdigest()}}
        with patch.object(helper, "run", side_effect=subprocess.CalledProcessError(1, ["python"])) as load, \
             contextlib.redirect_stdout(io.StringIO()):
            helper.native_checks(env, native, Path(sys.executable), self.report)
        load.assert_called_once()
        command = [str(arg) for arg in load.call_args.args[0]]
        self.assertIn(str(library), command)
        self.assertIn("import torch,importlib.util,sys", command[4])
        self.assertEqual(self.report.statuses, ["FAIL"])

    def test_setup_reuses_matching_packages_across_repeated_runs(self):
        base = self.root / "envs/lingbot14b"
        (base / "venv/bin").mkdir(parents=True)
        (base / "venv/pyvenv.cfg").write_text("include-system-site-packages = false\n")
        args = SimpleNamespace(backbone="lingbot14b", dry_run=False, python=sys.executable,
                               env_root=str(self.root / "envs"))
        observed = {name.lower(): value for name, value in DATA["environments"][args.backbone]["packages"].items()}
        outputs = ["3.10\n", "3.10\n", "pip fixture\n", json.dumps(observed)] * 2
        with patch.object(helper, "source_checks"), patch.object(helper, "hardware_checks"), \
             patch.object(helper, "probe_python"), \
             patch.object(helper.subprocess, "check_output", side_effect=outputs), \
             patch.object(helper, "run") as install, contextlib.redirect_stdout(io.StringIO()):
            helper.setup(args, DATA, self.report, base)
            helper.setup(args, DATA, self.report, base)
        install.assert_not_called()
        self.assertNotIn("FAIL", self.report.statuses)

    def test_setup_rejects_nested_cache_or_venv_symlink_escape_before_writes(self):
        args = SimpleNamespace(backbone="lingbot14b", dry_run=False, python=sys.executable,
                               env_root=str(self.root / "envs"))
        base = self.root / "envs/lingbot14b"
        outside = self.root / "outside chosen env root"
        outside.mkdir()
        for relative in ("venv", "tmp", "cache/pip", "cache/torch_extensions"):
            with self.subTest(path=relative):
                if base.exists():
                    shutil.rmtree(base)
                link = base / relative
                link.parent.mkdir(parents=True)
                link.symlink_to(outside, target_is_directory=True)
                with patch.object(helper, "source_checks"), patch.object(helper, "hardware_checks"), \
                     patch.object(helper.subprocess, "check_output", return_value="3.10\n"), \
                     patch.object(helper, "run") as install, contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, "escapes --env-root"):
                        helper.setup(args, DATA, self.report, base)
                install.assert_not_called()
                self.assertEqual(list(outside.iterdir()), [])
                self.assertFalse((base / "tmp").exists() and relative != "tmp")

    def test_empty_interpreter_probe_output_reports_failure(self):
        with patch.object(helper.subprocess, "check_output", return_value=""), \
             patch.object(helper.subprocess, "run") as check, contextlib.redirect_stdout(io.StringIO()):
            helper.probe_python(Path(sys.executable), DATA["environments"]["lingbot14b"], self.report,
                                self.root / "envs/lingbot14b")
        check.assert_not_called()
        self.assertEqual(self.report.statuses, ["FAIL"])
        self.report = helper.Report()
        args = SimpleNamespace(backbone="longlive1.3b", evaluator_python=sys.executable)
        with patch.object(helper.subprocess, "check_output", return_value=""), \
             contextlib.redirect_stdout(io.StringIO()):
            helper.evaluator_checks(args, DATA, self.report, self.root / "envs/longlive1.3b")
        self.assertEqual(self.report.statuses, ["FAIL", "MANUAL_REQUIRED"])


    def test_missing_ensurepip_is_manual_before_new_venv_or_cache_creation(self):
        base = self.root / "envs/lingbot14b"
        args = SimpleNamespace(backbone="lingbot14b", dry_run=False, python=sys.executable,
                               env_root=str(self.root / "envs"))
        output = io.StringIO()
        with patch.object(helper, "source_checks"), patch.object(helper, "hardware_checks"), \
             patch.object(helper.subprocess, "check_output", side_effect=[
                 "3.10\n", subprocess.CalledProcessError(1, ["python", "-m", "ensurepip"]),
                 subprocess.CalledProcessError(1, ["python", "-m", "pip"])]) as probe, \
             patch.object(helper, "run") as install, contextlib.redirect_stdout(output):
            helper.setup(args, DATA, self.report, base)
        install.assert_not_called()
        self.assertFalse(base.exists())
        self.assertEqual(self.report.statuses, ["MANUAL_REQUIRED"])
        self.assertIn("MANUAL_REQUIRED Python ensurepip", output.getvalue())
        self.assertEqual(probe.call_args_list[1].args[0][-3:], ["-m", "ensurepip", "--version"])

    def test_system_pip_bootstrap_populates_only_new_isolated_venv(self):
        base = self.root / "envs/lingbot14b"
        args = SimpleNamespace(backbone="lingbot14b", dry_run=False, python=sys.executable,
                               env_root=str(self.root / "envs"))
        calls = []
        def run_stub(cmd, env=None):
            calls.append((list(map(str,cmd)), dict(env or {})))
            if "-m" in cmd and "venv" in cmd:
                venv = base / "venv"
                (venv / "bin").mkdir(parents=True)
                (venv / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
                (venv / "bin/python").write_bytes(b"mock interpreter")
        side = [
            "3.10\n", subprocess.CalledProcessError(1, ["python", "-m", "ensurepip"]),
            "pip 26.2.1 from /user/.local/lib/python3.10/site-packages/pip",
            "3.10\n", "pip 26.2.1 from /new/venv/pip", "{}",
        ]
        with patch.object(helper, "source_checks"), patch.object(helper, "hardware_checks"),              patch.object(helper.subprocess, "check_output", side_effect=side),              patch.object(helper, "run", side_effect=run_stub),              patch.object(helper, "probe_python"):
            helper.setup(args, DATA, self.report, base)
        self.assertGreaterEqual(len(calls), 2)
        self.assertIn("--without-pip", calls[0][0])
        self.assertIn("--python", calls[1][0])
        self.assertIn(str(base / "venv/bin/python"), calls[1][0])
        self.assertIn("pip==26.2.1", calls[1][0])
        self.assertNotIn("--system-site-packages", calls[0][0])
        self.assertEqual(calls[1][1].get("PIP_EXTRA_INDEX_URL"), "")
        self.assertNotIn("PYTHONNOUSERSITE", calls[1][1])
        self.assertIn("PASS", self.report.statuses)

    def test_existing_incomplete_venv_missing_pip_is_preserved_without_mutation(self):
        base = self.root / "envs/lingbot14b"
        venv = base / "venv"
        (venv / "bin").mkdir(parents=True)
        config = venv / "pyvenv.cfg"
        config.write_text("include-system-site-packages = false\n")
        marker = venv / "retained fixture"
        marker.write_bytes(b"preserve")
        before = {p.relative_to(venv): p.read_bytes() for p in venv.rglob("*") if p.is_file()}
        args = SimpleNamespace(backbone="lingbot14b", dry_run=False, python=sys.executable,
                               env_root=str(self.root / "envs"))
        output = io.StringIO()
        with patch.object(helper, "source_checks"), patch.object(helper, "hardware_checks"), \
             patch.object(helper.subprocess, "check_output", side_effect=[
                 "3.10\n", "3.10\n", subprocess.CalledProcessError(1, ["python", "-m", "pip"])]) as probe, \
             patch.object(helper, "probe_python") as imports, patch.object(helper, "run") as install, \
             contextlib.redirect_stdout(output):
            helper.setup(args, DATA, self.report, base)
        install.assert_not_called()
        imports.assert_not_called()
        self.assertEqual(self.report.statuses, ["MANUAL_REQUIRED"])
        self.assertIn("MANUAL_REQUIRED venv pip", output.getvalue())
        self.assertEqual(probe.call_args.args[0][-3:], ["-m", "pip", "--version"])
        self.assertEqual(before, {p.relative_to(venv): p.read_bytes() for p in venv.rglob("*") if p.is_file()})

    def test_complete_existing_venv_does_not_require_base_ensurepip(self):
        base = self.root / "envs/lingbot14b"
        (base / "venv/bin").mkdir(parents=True)
        (base / "venv/pyvenv.cfg").write_text("include-system-site-packages = false\n")
        args = SimpleNamespace(backbone="lingbot14b", dry_run=False, python=sys.executable,
                               env_root=str(self.root / "envs"))
        observed = {name.lower(): value for name, value in DATA["environments"][args.backbone]["packages"].items()}
        def probe(command, **kwargs):
            if "ensurepip" in command:
                raise subprocess.CalledProcessError(1, command)
            if command[-3:] == ["-m", "pip", "--version"]:
                return "pip fixture\n"
            return json.dumps(observed) if "import importlib.metadata" in command[-1] else "3.10\n"
        with patch.object(helper, "source_checks"), patch.object(helper, "hardware_checks"), \
             patch.object(helper.subprocess, "check_output", side_effect=probe) as check, \
             patch.object(helper, "probe_python") as imports, patch.object(helper, "run") as install, \
             contextlib.redirect_stdout(io.StringIO()):
            helper.setup(args, DATA, self.report, base)
        self.assertFalse(any("ensurepip" in call.args[0] for call in check.call_args_list))
        install.assert_not_called()
        imports.assert_called_once()
        self.assertNotIn("FAIL", self.report.statuses)

    def test_new_venv_checks_bundled_ensurepip_without_bootstrap_fallback(self):
        base = self.root / "envs/lingbot14b"
        args = SimpleNamespace(backbone="lingbot14b", dry_run=False, python=sys.executable,
                               env_root=str(self.root / "envs"))
        observed = {name.lower(): value for name, value in DATA["environments"][args.backbone]["packages"].items()}
        outputs = ["3.10\n", "pip bundled fixture\n", "3.10\n", "pip fixture\n", json.dumps(observed)]
        with patch.object(helper, "source_checks"), patch.object(helper, "hardware_checks"), \
             patch.object(helper.subprocess, "check_output", side_effect=outputs) as probe, \
             patch.object(helper, "probe_python"), patch.object(helper, "run") as install, \
             contextlib.redirect_stdout(io.StringIO()):
            helper.setup(args, DATA, self.report, base)
        self.assertEqual(probe.call_args_list[1].args[0][-3:], ["-m", "ensurepip", "--version"])
        self.assertEqual(probe.call_args_list[3].args[0][-3:], ["-m", "pip", "--version"])
        install.assert_called_once()
        self.assertEqual(install.call_args.args[0][-3:], ["-m", "venv", base / "venv"])


    def test_python_probe_reports_missing_module_and_bounded_redacted_native_error(self):
        secret = "hf_fixture_secret_should_never_appear"
        native_error = "/tmp/fixture.so: undefined symbol: fixture_symbol " + secret + " " + "x" * 300
        env = {"python": "%s.%s" % sys.version_info[:2],
               "packages": {"fixture_missing": "fixture", "fixture_native": "fixture"},
               "torch_cuda": "12.4", "site_files": {}}
        observed = {}
        def import_fixture(name):
            if name == "fixture_missing":
                raise ModuleNotFoundError("unnecessary message " + secret, name="fixture_dependency")
            if name == "fixture_native":
                raise ImportError(native_error + "\nprivate second line")
            raise RuntimeError("unnecessary message " + secret)
        def execute_probe(command, **kwargs):
            result = io.StringIO()
            with patch("importlib.metadata.version", return_value="fixture"), \
                 patch("importlib.import_module", side_effect=import_fixture), \
                 patch.object(sys, "argv", ["-c", *command[-2:]]), \
                 patch.dict(os.environ, kwargs["env"], clear=True), contextlib.redirect_stdout(result):
                exec(command[4], {})
            observed.update(json.loads(result.getvalue()))
            return result.getvalue()
        output = io.StringIO()
        with patch.dict(os.environ, {"HF_TOKEN": secret}), \
             patch.object(helper.subprocess, "check_output", side_effect=execute_probe), \
             patch.object(helper.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), \
             contextlib.redirect_stdout(output):
            helper.probe_python(Path(sys.executable), env, self.report, self.root / "envs/fixture")
        self.assertEqual(observed["imports"]["fixture_missing"], "ModuleNotFoundError: fixture_dependency")
        native = observed["imports"]["fixture_native"]
        self.assertIn("ImportError: /tmp/fixture.so: undefined symbol: fixture_symbol", native)
        self.assertIn("<redacted>", native)
        self.assertLessEqual(len(native), len("ImportError: ") + 240)
        self.assertNotIn("second line", native)
        self.assertEqual(observed["imports"]["flash_attn_interface"], "RuntimeError")
        self.assertNotIn(secret, output.getvalue())
        self.assertIn("FAIL import fixture_missing: ModuleNotFoundError: fixture_dependency", output.getvalue())


    def test_portable_compatibility_overlays_do_not_rewrite_original_pins(self):
        for backbone in ("lingbot14b", "solarwm33b"):
            with self.subTest(backbone=backbone):
                env = DATA["environments"][backbone]
                original = helper.selected_packages(env)
                portable = helper.portable_packages(env)
                self.assertEqual(original["Pillow"], "9.0.1")
                self.assertEqual(portable["Pillow"], "9.5.0")
                self.assertEqual(portable["ftfy"], "6.3.1")
                self.assertEqual(portable["torchvision"], "0.21.0+cu124")
                self.assertEqual(portable.get("imageio-ffmpeg"),
                                 "0.6.0" if backbone == "solarwm33b" else None)
                self.assertEqual(original["torch"], portable["torch"])
                self.assertNotIn("ftfy", original)
                self.assertNotIn("torchvision", original)
                self.assertTrue(helper.receipt_is_override("Pillow", env, portable))
                self.assertFalse(helper.receipt_is_override("torch", env, portable))

    def test_portable_dry_run_reports_new_pins_without_claiming_historical_identity(self):
        env = DATA["environments"]["solarwm33b"]
        args = SimpleNamespace(backbone="solarwm33b", dry_run=True, portable=True)
        buffer = io.StringIO()
        with patch.object(helper, "source_checks"), patch.object(helper, "run") as install, \
             patch.object(helper.subprocess, "check_output") as probe, contextlib.redirect_stdout(buffer):
            helper.setup(args, {"environments": {"solarwm33b": env}}, self.report,
                         self.root / "envs/solarwm33b")
        install.assert_not_called()
        probe.assert_not_called()
        self.assertIn("NOT_CHECKED Pillow: 9.5.0", buffer.getvalue())
        self.assertIn("NOT_CHECKED torchvision: 0.21.0+cu124", buffer.getvalue())
        self.assertIn("new portable compatibility pins", buffer.getvalue())
        self.assertIn("not paper environment", buffer.getvalue())
        self.assertEqual(env["packages"]["Pillow"], "9.0.1")

    def test_stale_portable_receipt_fails_before_interpreter_import(self):
        env = DATA["environments"]["lingbot14b"]
        base = self.root / "envs/lingbot14b"
        base.mkdir(parents=True)
        (base / "portable-dependency-closure.json").write_text(json.dumps(dict(
            schema=1, scope="CURRENT_PORTABLE_ENVIRONMENT_NOT_HISTORICAL",
            top_level_pins=helper.selected_packages(env),
            portable_top_level_pins={"Pillow": "9.0.1"})))
        report = helper.Report()
        with patch.object(helper.subprocess, "check_output") as probe:
            helper.probe_python(Path(sys.executable), env, report, base)
        probe.assert_not_called()
        self.assertIn("FAIL", report.statuses)

    def test_portable_receipt_verifies_runtime_pins_and_qwen_import(self):
        env = DATA["environments"]["solarwm33b"]
        original = helper.selected_packages(env)
        effective = helper.portable_packages(env)
        base = self.root / "envs/solarwm33b"
        base.mkdir(parents=True)
        (base / "portable-dependency-closure.json").write_text(json.dumps(dict(
            schema=1, scope="CURRENT_PORTABLE_ENVIRONMENT_NOT_HISTORICAL",
            top_level_pins=original, portable_top_level_pins=effective,
            portable_compatibility_overrides=env["portable_overrides"])))
        report = helper.Report()
        buffer = io.StringIO()
        def probe(cmd, **kwargs):
            self.assertEqual(json.loads(cmd[-2]), list(effective))
            imports = json.loads(cmd[-1])
            self.assertIn("torchvision", imports)
            self.assertIn("ftfy", imports)
            self.assertIn("transformers.models.qwen3_vl.modeling_qwen3_vl", imports)
            self.assertIn("imageio_ffmpeg", imports)
            return json.dumps(dict(python="3.10", python_full="3.10.12", python_header=True,
                include=str(self.root), site=str(self.root), torch=effective["torch"],
                cuda="12.4", abi=False, versions=effective,
                imports={name: "OK" for name in imports}))
        with patch.object(helper.subprocess, "check_output", side_effect=probe), \
             patch.object(helper.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), \
             contextlib.redirect_stdout(buffer):
            helper.probe_python(Path(sys.executable), env, report, base)
        self.assertNotIn("FAIL", report.statuses, buffer.getvalue())
        self.assertIn("portable override (NOT historical)=9.5.0", buffer.getvalue())
        self.assertIn("PASS import transformers.models.qwen3_vl.modeling_qwen3_vl", buffer.getvalue())
        self.assertIn("MANUAL_REQUIRED portable dependency provenance", buffer.getvalue())

    def supplemental_environment(self):
        return {"python": "3.10", "torch_cuda": "12.4", "compiler_cuda": "12.6",
                "packages": {"torch": "2.6.0+cu124", "Pillow": "9.0.1"},
                "manual_distributions": [], "site_files": {},
                "installation_packages": {"torch": "99.0", "pillow": "99.0",
                    "torchvision": "0.21.0+cu124", "ftfy": "6.3.1", "scipy": "1.15.3",
                    "imageio": "2.37.0", "nvidia-cublas-cu12": "12.4.5.8"},
                "installation_packages_provenance": "fixture: current H200 reference closure/upstream Torch requirements"}

    def test_supplemental_setup_preserves_historical_pins_and_official_indexes(self):
        env = self.supplemental_environment()
        base = self.root / "envs/lingbot1.3b"
        (base / "venv/bin").mkdir(parents=True)
        (base / "venv/pyvenv.cfg").write_text("include-system-site-packages = false\n")
        args = SimpleNamespace(backbone="lingbot1.3b", dry_run=False, python=sys.executable,
                               env_root=str(self.root / "envs"))
        output = io.StringIO()
        def install(command, **kwargs):
            self.assertIn(env["installation_packages_provenance"], output.getvalue())
            self.assertIn("NOT recovered historical lock", output.getvalue())
        with patch.object(helper, "source_checks"), patch.object(helper, "hardware_checks"), \
             patch.object(helper, "probe_python"), \
             patch.object(helper.subprocess, "check_output", side_effect=["3.10\n", "3.10\n", "pip fixture\n", "{}"]), \
             patch.object(helper, "run", side_effect=install) as installed, contextlib.redirect_stdout(output):
            helper.setup(args, {"environments": {args.backbone: env}}, self.report, base)
            self.assertEqual(self.report.finish(), 3)
        commands = [call.args[0] for call in installed.call_args_list]
        requested = [next(str(arg) for arg in command if "==" in str(arg)) for command in commands]
        self.assertEqual(set(requested), {name + "==" + version for name, version in helper.selected_packages(env).items()})
        self.assertIn("torch==2.6.0+cu124", requested)
        self.assertIn("Pillow==9.0.1", requested)
        self.assertFalse(any("99.0" in requirement for requirement in requested))
        for command in commands:
            self.assertIn("--no-deps", command)
            self.assertIn("--only-binary=:all:", command)
            cuda = any(str(arg).startswith(("torch==", "torchvision==")) for arg in command)
            self.assertIn("https://download.pytorch.org/whl/cu124" if cuda else "https://pypi.org/simple", command)
        self.assertIn("GPU generation/performance NOT_CHECKED", output.getvalue())
        self.assertEqual(env["packages"], {"torch": "2.6.0+cu124", "Pillow": "9.0.1"})

    def test_supplemental_dry_run_lists_exact_selection_without_readiness_claim(self):
        env = self.supplemental_environment()
        base = self.root / "envs/lingbot1.3b"
        args = SimpleNamespace(backbone="lingbot1.3b", dry_run=True)
        output = io.StringIO()
        with patch.object(helper, "source_checks"), patch.object(helper, "run") as install, \
             patch.object(helper.subprocess, "check_output") as probe, contextlib.redirect_stdout(output):
            helper.setup(args, {"environments": {args.backbone: env}}, self.report, base)
            self.assertEqual(self.report.finish(dry=True), 0)
        install.assert_not_called()
        probe.assert_not_called()
        self.assertFalse(base.exists())
        self.assertIn("NOT_CHECKED torchvision: 0.21.0+cu124", output.getvalue())
        self.assertIn("NOT_CHECKED nvidia-cublas-cu12: 12.4.5.8", output.getvalue())
        self.assertNotIn("99.0", output.getvalue())
        self.assertIn("NOT recovered historical lock", output.getvalue())
        self.assertIn("GPU generation/performance NOT_CHECKED", output.getvalue())

    def test_supplemental_probe_checks_metadata_and_exact_cuda_suffixes_without_guessing_imports(self):
        env = self.supplemental_environment()
        expected = helper.selected_packages(env)
        for changed in (None, "torch", "torchvision", "nvidia-cublas-cu12"):
            with self.subTest(changed=changed):
                report = helper.Report()
                versions = dict(expected)
                torch = expected["torch"]
                if changed == "torch":
                    torch = torch.split("+")[0]
                elif changed == "torchvision":
                    versions["torchvision"] = versions["torchvision"].split("+")[0]
                elif changed:
                    versions[changed] = "0"
                def probe(command, **kwargs):
                    self.assertEqual(json.loads(command[-2]), list(expected))
                    imports = json.loads(command[-1])
                    self.assertEqual(set(imports), {"torch", "PIL", "torchvision", "ftfy", "scipy", "imageio",
                                                    "flash_attn_interface", "flash_attn_3._C"})
                    self.assertFalse(any(name.startswith("nvidia") for name in imports))
                    return json.dumps({"python": "3.10", "versions": versions, "torch": torch,
                                       "cuda": "12.4", "abi": False, "site": str(self.root),
                                       "imports": {name: "OK" for name in imports}})
                output = io.StringIO()
                with patch.object(helper.subprocess, "check_output", side_effect=probe), \
                     patch.object(helper.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), \
                     contextlib.redirect_stdout(output):
                    helper.probe_python(Path(sys.executable), env, report, self.root / "envs/lingbot1.3b")
                self.assertEqual("FAIL" in report.statuses, changed is not None)
                self.assertIn("supplemental (NOT recovered historical lock)=0.21.0+cu124", output.getvalue())
                self.assertNotIn("original=0.21.0+cu124", output.getvalue())


    def test_actual_manifest_supplemental_package_missing_or_mismatched_is_fail(self):
        env = dict(DATA["environments"]["lingbot1.3b"])
        env["site_files"] = {}  # This test isolates metadata from external frozen site files.
        target = "nvidia-cublas-cu12"
        self.assertIn(target, env["installation_packages"])
        expected = helper.selected_packages(env)
        for state in ("exact", "missing", "mismatched"):
            with self.subTest(state=state):
                report = helper.Report()
                versions = dict(expected)
                if state == "missing":
                    versions.pop(target)
                elif state == "mismatched":
                    versions[target] = "0.0.0-fixture"
                def probe(command, **kwargs):
                    self.assertIn(target, json.loads(command[-2]))
                    imports = json.loads(command[-1])
                    self.assertFalse(any(name.startswith("nvidia") for name in imports))
                    return json.dumps({"python": env["python"], "versions": versions,
                                       "torch": env["packages"]["torch"], "cuda": env["torch_cuda"],
                                       "abi": env["torch_cuda"] == "12.8", "site": str(self.root),
                                       "imports": {name: "OK" for name in imports}})
                output = io.StringIO()
                with patch.object(helper.subprocess, "check_output", side_effect=probe), \
                     patch.object(helper.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), \
                     contextlib.redirect_stdout(output):
                    helper.probe_python(Path(sys.executable), env, report, self.root / "envs/lingbot1.3b")
                self.assertEqual("FAIL" in report.statuses, state != "exact")
                self.assertIn(("PASS" if state == "exact" else "FAIL") + " " + target + ":", output.getvalue())
                self.assertIn("supplemental (NOT recovered historical lock)=" + expected[target], output.getvalue())


    def test_probe_reports_active_python_header_present_or_missing_without_build(self):
        include = self.root / "Python include with spaces"
        include.mkdir()
        env = {"python": "%s.%s" % sys.version_info[:2], "torch_cuda": "12.4",
               "packages": {"torch": "fixture"}, "site_files": {}}
        torch = SimpleNamespace(__version__="fixture", version=SimpleNamespace(cuda="12.4"),
                                _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=False))
        def execute_probe(command, **kwargs):
            result = io.StringIO()
            # Resolve sysconfig.get_paths before mocking importlib.import_module.
            # Linux Python 3.11 otherwise resolves sysconfig through the mocked
            # importer and receives a SimpleNamespace instead of the module.
            with patch("importlib.metadata.version", return_value="fixture"), \
                 patch("sysconfig.get_paths", return_value={"purelib": str(self.root / "site"), "include": str(include)}), \
                 patch("importlib.import_module", side_effect=lambda name: torch if name == "torch" else SimpleNamespace()), \
                 patch.object(sys, "argv", ["-c", *command[-2:]]), contextlib.redirect_stdout(result):
                exec(command[4], {})
            return result.getvalue()
        for present in (False, True):
            with self.subTest(present=present):
                if present:
                    (include / "Python.h").write_text("/* availability fixture */\n")
                report = helper.Report()
                output = io.StringIO()
                with patch.object(helper.subprocess, "check_output", side_effect=execute_probe), \
                     patch.object(helper.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as check, \
                     patch.object(helper, "run") as build, contextlib.redirect_stdout(output):
                    helper.probe_python(Path(sys.executable), env, report, self.root / "envs/fixture")
                build.assert_not_called()
                self.assertEqual(check.call_count, 1)  # read-only pip check only
                expected = ("PASS" if present else "MANUAL_REQUIRED") + " Python.h availability: " + str(include / "Python.h")
                self.assertIn(expected, output.getvalue())
                self.assertIn("active Python " + sys.version.split()[0], output.getvalue())
                if not present:
                    self.assertIn("provide matching Python " + sys.version.split()[0] + " development headers externally", output.getvalue())
                self.assertNotIn("FAIL", report.statuses)


if __name__ == "__main__":
    unittest.main()
