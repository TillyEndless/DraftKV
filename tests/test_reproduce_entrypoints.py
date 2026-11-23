"""Offline tests for the public reproduction CLI and benchmark provenance."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = module_from_path("draftkv_reproduce_cli", "scripts/reproduce.py")
bench = module_from_path("draftkv_benchmark_setup", "scripts/setup_benchmarks.py")
evaluator = module_from_path("draftkv_evaluation_facade", "scripts/evaluate.py")


class ReproduceCLITest(unittest.TestCase):
    def test_setup_forwards_gpu_env_mode_and_downloader(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            args = SimpleNamespace(
                backbone="lingbot1.3b", gpu=3, env_mode="conda",
                asset_root=tmp, env_root=tmp + "/env",
                with_dataset=True, with_evaluators=True,
                python=None, dry_run=True,
            )
            with mock.patch.object(cli, "run_command", side_effect=lambda c, dry_run=False: calls.append((list(c), dry_run))):
                cli.setup(args)
            self.assertEqual(len(calls), 2)
            command, dry_run = calls[0]
            self.assertIn("bootstrap", command)
            self.assertEqual(command[command.index("--gpu") + 1], "3")
            self.assertEqual(command[command.index("--env-mode") + 1], "conda")
            self.assertIn("--with-dataset", command)
            self.assertIn("--dry-run", command)
            evaluator_command = calls[1][0]
            self.assertTrue(str(evaluator_command[1]).endswith("/scripts/setup_benchmarks.py"))

    def test_setup_can_skip_optional_evaluators(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(backbone="longlive1.3b", gpu=0, env_mode="auto",
                                   asset_root=tmp, env_root=tmp + "/env",
                                   with_dataset=False, with_evaluators=False,
                                   python=None, dry_run=True)
            with mock.patch.object(cli, "run_command") as runner:
                cli.setup(args)
            runner.assert_called_once()
            command = runner.call_args.args[0]
            self.assertNotIn("--with-dataset", command)

    def test_inference_selects_one_verified_prepared_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "prepared" / "lingbot1.3b" / "cases"
            for case in ("case_0002", "case_0001"):
                path = folder / case
                path.mkdir(parents=True)
                (path / "request.json").write_text(json.dumps({"case_id": case}))
            args = SimpleNamespace(backbone="lingbot1.3b", asset_root=tmp,
                                   env_root=tmp + "/cache", gpu=2, env_mode="venv",
                                   case_id=None, input=None, output=None, dry_run=True,
                                   profile="on")
            with mock.patch.object(cli, "run_command") as runner:
                cli.infer(args)
            command = list(runner.call_args.args[0])
            self.assertEqual(command[command.index("--gpu") + 1], "2")
            self.assertEqual(command[command.index("--profile") + 1], "on")
            self.assertIn("case_0001", command[command.index("--input") + 1].as_posix())
            self.assertIn("--dry-run", command)

    def test_missing_prepared_request_fails_honestly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "No source-verified prepared request"):
                cli.infer(SimpleNamespace(backbone="lingbot1.3b", asset_root=tmp,
                                          env_root=tmp, gpu=0, env_mode="auto",
                                          case_id=None, input=None, output=None,
                                          dry_run=True, profile="off"))

    def test_bad_case_id_never_selects_other_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "prepared" / "lingbot1.3b" / "cases/case_0001"
            folder.mkdir(parents=True)
            (folder / "request.json").write_text(json.dumps({"case_id": "case_0001"}))
            with self.assertRaisesRegex(ValueError, "No source-verified prepared request"):
                cli.infer(SimpleNamespace(backbone="lingbot1.3b", asset_root=tmp,
                                          env_root=tmp, gpu=0, env_mode="auto",
                                          case_id="case_0099", input=None, output=None,
                                          dry_run=True, profile="off"))

    def test_gpu_must_be_nonnegative(self):
        with self.assertRaises(SystemExit) as exc:
            cli.main(["infer", "--backbone", "lingbot1.3b", "--gpu", "-1"])
        self.assertEqual(exc.exception.code, 2)


class BenchmarkSetupTest(unittest.TestCase):
    def test_original_orbia_lock_is_unchanged(self):
        lock = bench.official_dataset_lock()
        self.assertEqual(lock["selection"]["short_cases"], 480)
        self.assertEqual(lock["official_commit"],
                         "f7d8ddde3069a6d5ce75a87fa6b4fc896cc78a59")
        self.assertEqual(lock["distribution"]["public"]["size"], 1038430258)

    def test_offline_plan_is_network_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "data"
            args = SimpleNamespace(backbone="lingbot1.3b", asset_root=str(folder),
                                   gpu=0, dry_run=True)
            with mock.patch.object(bench.subprocess, "run",
                                   side_effect=AssertionError("network/installation invoked")):
                bench.setup(args)
            self.assertFalse(folder.exists())

    def test_evaluator_source_must_be_exactly_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "not installed"):
                evaluator.verified_checkout(Path(tmp), evaluator.ORBIA_REV)

    def test_empty_gpu_scoring_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = evaluator.main(["table3", "--asset-root", tmp,
                                     "--video", tmp + "/missing.mp4",
                                     "--da3-cache", tmp,
                                     "--output", tmp + "/metrics.json"])
            self.assertEqual(status, 1)
            self.assertFalse((Path(tmp) / "metrics.json").exists())

    def test_bilingual_readme_links_resolve_and_locale_state_matches(self):
        import re
        from urllib.parse import urlparse

        for name in ("README.md", "docs/README_CN.md"):
            with self.subTest(readme=name):
                readme = ROOT / name
                content = readme.read_text()
                references = re.findall(
                    r'\]\(([^)]+)\)|<img[^>]+src="([^"]+)"|<a[^>]+href="([^"]+)"',
                    content,
                )
                for occurrence in references:
                    path = next(value for value in occurrence if value)
                    if urlparse(path).scheme or path.startswith("#"):
                        continue
                    self.assertTrue((readme.parent / path.split("#", 1)[0]).is_file(),
                                    f"Broken link in {name}: {path}")
                if name == "README.md":
                    self.assertIn('<strong>English</strong>', content)
                    self.assertIn('<a href="docs/README_CN.md">简体中文</a>', content)
                else:
                    self.assertIn('<a href="../README.md">English</a>', content)
                    self.assertIn('<strong>简体中文</strong>', content)
                self.assertIn("prepared" if name == "README.md" else "准备好的", content)

    def test_consolidated_bilingual_readmes_are_self_contained(self):
        for name in ("README.md", "docs/README_CN.md"):
            content = (ROOT / name).read_text()
            for command in ("./scripts/reproduce.sh setup",
                            "./scripts/reproduce.sh infer",
                            "./scripts/draftkv.sh environment",
                            "./scripts/draftkv.sh bootstrap",
                            "./scripts/draftkv.sh run",
                            "python3 scripts/evaluate.py table2",
                            "python3 scripts/evaluate.py table3"):
                self.assertIn(command, content)
            self.assertIn("--profile on", content)
            self.assertIn("--all --resume", content)
            self.assertIn("--model", content)
            # Public README must provide a concrete end-to-end 1.3B example
            # and explain its generic placeholders without exposing scoring internals.
            self.assertIn("./scripts/reproduce.sh setup", content)
            self.assertIn("--model lingbot1.3b --gpu 0", content)
            self.assertIn('export DEMO_RUN="$DRAFTKV_ASSETS/demo/lingbot1.3b"', content)
            self.assertIn('--output "$DEMO_RUN/video.mp4"', content)
            self.assertIn('--video "$DEMO_RUN/video.mp4"', content)
            self.assertIn('--output "$DEMO_RUN/table3.json"', content)
            self.assertIn("<input_path>", content)
            self.assertIn("<gpu_id>", content)
            self.assertNotIn("Overall = ", content)
            self.assertNotIn("480", content)
            self.assertNotIn("--backbone", content)
            self.assertNotIn("native-deps", content)
            for retired in ("INSTALL", "QUICKSTART", "PROFILING", "REPRODUCIBILITY"):
                self.assertNotIn(f"docs/{retired}", content)
                self.assertFalse((ROOT / f"docs/{retired}.md").exists())
                self.assertFalse((ROOT / f"docs/{retired}_CN.md").exists())
        self.assertTrue((ROOT / "docs/THIRD_PARTY_NOTICES.md").is_file())


if __name__ == "__main__":
    unittest.main()
