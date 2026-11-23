"""CPU-only invariants for --all: 480-case index, integrity and resume."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("batch_reproduce", ROOT / "scripts/reproduce.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root, n=480):
    base = root / "prepared" / "lingbot1.3b"
    items = {}
    case_ids = [f"case_{i:04d}" for i in range(1, n + 1)]
    for case_id in case_ids:
        p = base / "cases" / case_id / "request.json"
        digest = write_json(p, {
            "case_id": case_id, "backbone": "LingBot1.3B",
            "input_mode": "orbia",
            "dataset_manifest_sha256": "a" * 64,
            "dataset_index_sha256": "b" * 64,
        })
        items[case_id] = {"request": str(p), "request_sha256": digest}
    receipt = {"status": "READY", "backbone": "LingBot1.3B",
               "case_ids": case_ids, "dataset_manifest_sha256": "a" * 64,
               "dataset_index_sha256": "b" * 64}
    receipt_sha = write_json(base / "preparation_receipt.json", receipt)
    write_json(base / "requests.json", {
        "schema_version": 1, "backbone": "LingBot1.3B",
        "input_mode": "orbia", "dataset_manifest_sha256": "a" * 64,
        "dataset_index_sha256": "b" * 64, "requests": items,
        "preparation_receipt_sha256": receipt_sha,
    })
    return base


def args(root, **overrides):
    data = dict(backbone="lingbot1.3b", gpu=1, profile="on",
                env_mode="conda", asset_root=str(root), env_root=str(root / "cache"),
                dry_run=False, all=True, case_id=None, input=None,
                output=None, output_dir=str(root / "batch"), resume=False,
                keep_going=False)
    data.update(overrides)
    return SimpleNamespace(**data)


class BatchTest(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.root = Path(self.t.name)

    def tearDown(self):
        self.t.cleanup()

    def test_complete_index_and_order(self):
        fixture(self.root)
        index, digest, cases = cli.verified_batch_requests(self.root, "lingbot1.3b")
        self.assertEqual(len(cases), 480)
        self.assertEqual(cases[0][0], "case_0001")
        self.assertEqual(cases[-1][0], "case_0480")
        self.assertEqual(index.name, "requests.json")
        self.assertEqual(digest, cli.sha256_file(index))

    def test_partial_index_fails_before_running(self):
        fixture(self.root, n=479)
        with self.assertRaisesRegex(ValueError, "complete 480-case"):
            cli.verified_batch_requests(self.root, "lingbot1.3b")

    def test_corrupted_request_fails(self):
        base = fixture(self.root)
        p = base / "cases" / "case_0243" / "request.json"
        p.write_text(p.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            cli.verified_batch_requests(self.root, "lingbot1.3b")

    def test_corrupted_receipt_fails(self):
        base = fixture(self.root)
        with (base / "preparation_receipt.json").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "receipt SHA-256 mismatch"):
            cli.verified_batch_requests(self.root, "lingbot1.3b")

    def test_batch_runs_each_case_in_distinct_directories(self):
        fixture(self.root)
        commands = []
        def fake_run(command, dry_run=False):
            commands.append(command)
            target = Path(command[command.index("--output") + 1])
            target.parent.mkdir(parents=True)
            target.write_bytes(b"mp4")
        with mock.patch.object(cli, "run_command", side_effect=fake_run):
            with redirect_stdout(StringIO()):
                cli.infer(args(self.root))
        self.assertEqual(len(commands), 480)
        self.assertIn("--profile", commands[0])
        self.assertEqual(commands[0][commands[0].index("--profile") + 1], "on")
        self.assertEqual(commands[0][commands[0].index("--gpu") + 1], "1")
        manifest = json.loads((self.root / "batch/batch.json").read_text())
        self.assertEqual(len(manifest["cases"]), 480)
        self.assertEqual({v["status"] for v in manifest["cases"].values()}, {"completed"})
        self.assertEqual(len({v["output"] for v in manifest["cases"].values()}), 480)

    def test_resume_after_failure_uses_next_attempt_without_overwrite(self):
        fixture(self.root)
        commands = []
        def fake_run(command, dry_run=False):
            output = Path(command[command.index("--output") + 1])
            commands.append(output)
            output.parent.mkdir(parents=True)
            if len(commands) == 3:
                raise RuntimeError("simulated crash")
            output.write_bytes(b"mp4")
        # Use CalledProcessError to model worker failure.
        def error_run(command, dry_run=False):
            output = Path(command[command.index("--output") + 1])
            commands.append(output)
            output.parent.mkdir(parents=True)
            if len(commands) == 3:
                raise __import__("subprocess").CalledProcessError(12, command)
            output.write_bytes(b"mp4")
        with mock.patch.object(cli, "run_command", side_effect=error_run):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                with self.assertRaisesRegex(ValueError, "Batch incomplete"):
                    cli.infer(args(self.root))
        state = json.loads((self.root / "batch/batch.json").read_text())
        self.assertEqual(state["cases"]["case_0001"]["status"], "completed")
        self.assertEqual(state["cases"]["case_0003"]["status"], "failed")
        resumed_commands = []
        def resumed(command, dry_run=False):
            output = Path(command[command.index("--output") + 1])
            resumed_commands.append(output)
            output.parent.mkdir(parents=True)
            output.write_bytes(b"mp4")
        with mock.patch.object(cli, "run_command", side_effect=resumed):
            with redirect_stdout(StringIO()):
                cli.infer(args(self.root, resume=True))
        self.assertEqual(len(resumed_commands), 478)
        self.assertIn("attempt-0002", str(resumed_commands[0]))
        self.assertEqual(state["cases"]["case_0003"]["attempt"], 1)
        new_state = json.loads((self.root / "batch/batch.json").read_text())
        self.assertEqual(new_state["cases"]["case_0003"]["attempt"], 2)
        self.assertEqual(len(new_state["cases"]), 480)

    def test_resume_rejects_changed_gpu(self):
        fixture(self.root)
        with mock.patch.object(cli, "run_command", side_effect=__import__("subprocess").CalledProcessError(1, ["x"])):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                with self.assertRaises(ValueError):
                    cli.infer(args(self.root))
        with self.assertRaisesRegex(ValueError, "identity/request index changed"):
            cli.infer(args(self.root, gpu=3, resume=True))

    def test_dry_run_has_no_side_effects(self):
        fixture(self.root)
        with mock.patch.object(cli, "run_command") as runner:
            with redirect_stdout(StringIO()):
                cli.infer(args(self.root, dry_run=True))
        self.assertEqual(runner.call_count, 480)
        self.assertFalse((self.root / "batch").exists())

    def test_incompatible_mode_flags_rejected(self):
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                cli.main(["infer", "--backbone", "lingbot1.3b",
                          "--all", "--case-id", "case_0001"])
            with self.assertRaises(SystemExit):
                cli.main(["infer", "--backbone", "lingbot1.3b", "--resume"])

    def test_longlive_fails_without_verified_full_manifest(self):
        with self.assertRaisesRegex(ValueError, "LongLive batch inputs"):
            cli.verified_batch_requests(self.root, "longlive1.3b")


if __name__ == "__main__":
    unittest.main()
