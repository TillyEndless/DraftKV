"""Source-owned input planner wrapper: CLI forwarding and fail-closed scope tests."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

FILE = Path(__file__).resolve().parents[1] / "tools/prepare_input_plan.py"
SPEC = importlib.util.spec_from_file_location("prepare_input_plan", FILE)
plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan)


class InputPlanTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        base = Path(self.t.name)
        self.public = base / "pub"
        self.version = self.public / "versions/lingbot13"
        (self.version/"src/draftkv/vendor/backbones/lingbot14").mkdir(parents=True)
        (self.version/"configs/backbones").mkdir(parents=True)
        (self.version/"configs/backbones/source_locks.json").write_text("{}")
        self.private = base / "private"
        self.private.mkdir()
        self.req = self.private / "request.json"
        self.req.write_text(json.dumps(dict(execution_scope="diagnostic",
            backbone="LingBot1.3B", case_id="sample", dataset_manifest_sha256="1"*64)))
        for folder in ("model","assets","native"):
            (self.private/folder).mkdir()
        for name in ("python", "spec.json"):
            (self.private/name).write_bytes(b"fixture")
        self.args=SimpleNamespace(backbone="lingbot1.3b",request=str(self.req),
            checkpoint=str(self.private/"model"),assets_dir=str(self.private/"assets"),
            python=str(self.private/"python"),b01_spec=str(self.private/"spec.json"),
            b01_cache=str(self.private/"native"),
            output_plan=str(self.private/"source-plan.json"),
            native_source=None,lora=None,silence=None,profile="summary")
        self.patch = patch.object(plan, "PUBLIC", self.public)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_source_planner_delegation_no_implicit_budget(self):
        output = dict(backbone="LingBot1.3B",algorithm="draftkv",sp=1,
                      request=dict(case_id="sample"))
        fake=SimpleNamespace(returncode=0,stdout=json.dumps(output),stderr="")
        with patch.object(plan.subprocess,"run",return_value=fake) as runner:
            plan.plan(self.args)
        expected=json.loads(Path(self.args.output_plan).read_text())
        self.assertEqual(expected,output)
        cmd=runner.call_args.args[0]
        self.assertEqual(Path(cmd[0]).resolve(),Path(self.args.python).resolve())
        self.assertEqual(cmd[2],"-c")
        self.assertIn(str(Path(self.args.request).resolve()),cmd)
        self.assertIn("summary",cmd)
        self.assertTrue("from draftkv.backbones.planning import make_plan" in cmd[3])

    def test_reject_unapproved_request_without_running_python(self):
        data=json.loads(self.req.read_text());data["execution_scope"]="arbitrary"
        self.req.write_text(json.dumps(data))
        with patch.object(plan.subprocess,"run") as run:
            with self.assertRaisesRegex(ValueError,"approved source-bound"):
                plan.plan(self.args)
            run.assert_not_called()

    def test_python_venv_symlink_remains_invoked_path(self):
        symlink = self.private / "venv/bin/python"
        symlink.parent.mkdir(parents=True)
        symlink.symlink_to(self.private / "python")
        self.args.python = str(symlink)
        fake = SimpleNamespace(returncode=0,
            stdout=json.dumps(dict(backbone="LingBot1.3B",algorithm="draftkv",
                                   request=dict(case_id="sample"))),stderr="")
        with patch.object(plan.subprocess, "run", return_value=fake) as runner:
            plan.plan(self.args)
        actual = runner.call_args.args[0]
        self.assertEqual(actual[0], str(symlink))
        self.assertNotEqual(actual[0], str(symlink.resolve()))

    def test_wrong_backbone_request_rejected(self):
        data=json.loads(self.req.read_text());data["backbone"]="SolarWM33B"
        self.req.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,"belongs to another"):
            plan.plan(self.args)

class DeviceArchivePlannerTests(unittest.TestCase):
    def test_original_direct_source_mode_selected_not_legacy_cpu_consumer(self):
        code = FILE.read_text()
        self.assertIn("host_pages_policy='quality'", code)
        self.assertIn("device_archive_plan=True", code)
        self.assertIn("gpu_krope_8x16=True", code)
        self.assertNotIn("host_pages_policy='systems'", code)

if __name__=="__main__":
    unittest.main()
