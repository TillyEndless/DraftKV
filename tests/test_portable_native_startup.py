"""Offline contracts for native extension compiler/startup handoff."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('portable_native_startup', ROOT / 'tools/portable_native_startup.py')
startup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(startup)


class NativeStartupTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.binary_root = self.root / 'native-build'
        self.binary_root.mkdir()
        self.binary = self.binary_root / 'device_route.so'
        self.binary.write_bytes(b'isolated deterministic fixture')
        self.plan_path = self.root / 'plan.json'
        self.plan = dict(backbone='LingBot14B', algorithm='draftkv', sp=1,
            run_dir=str(self.root / 'new-run'),
            algorithm_plugin=dict(config=dict(device_archive_plan=True,
                gpu_krope_8x16=True, retrieval_ablation='full',
                b01=dict(cache_directory='/external/b01'))))
        self.plan_path.write_text(json.dumps(self.plan))
        self.plan['_plan_path'] = str(self.plan_path)
        self.caches = {k: str(self.binary_root if k == 'DRAFTKV_PREPACK_BUILD'
                           else self.root / k.lower()) for k in startup.CACHE_NAMES}
        self.module = SimpleNamespace(__name__='draftkv_device_route_fixture',
                                      __file__=str(self.binary), source_identity='sha-fixture')
        self.fake_torch = SimpleNamespace(__version__='2.6.0+cu124',version=SimpleNamespace(cuda='12.4'),
            cuda=SimpleNamespace(set_device=lambda rank: None,
                                 get_device_capability=lambda rank: (9, 0)))

    def test_four_original_backbones_and_full_plan_guard(self):
        for key, name in startup.BACKBONES.items():
            with self.subTest(key=key):
                self.plan['backbone'] = name
                self.assertIs(startup.validate_plan(self.plan, key), self.plan)
        self.plan['algorithm_plugin']['config']['retrieval_ablation'] = 'camera-only'
        with self.assertRaisesRegex(RuntimeError, 'approved device-archive'):
            startup.validate_plan(self.plan, 'solarwm33b')

    def test_requires_isolated_compiler_cache_before_importing_torch(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'explicitly scoped'):
                startup.precompile(self.plan, 'lingbot14b')
        self.assertFalse((self.root / 'new-run').exists())

    def test_precompile_records_actual_binary_hash_without_model_or_timer(self):
        original_import = startup.importlib.import_module
        def fake_import(name):
            if name == 'draftkv.streaming.device_route':
                return SimpleNamespace(native=lambda: self.module)
            return original_import(name)
        with patch.dict(os.environ, {**self.caches, 'RANK':'0', 'LOCAL_RANK':'0'}):
            with patch.dict(sys.modules, {'torch': self.fake_torch}):
                with patch.object(startup.importlib, 'import_module', side_effect=fake_import):
                    receipt = startup.precompile(self.plan, 'lingbot14b')
        self.assertEqual(receipt['status'], 'NATIVE_EXTENSION_READY')
        self.assertEqual(receipt['extension']['sha256'], startup.digest(self.binary))
        self.assertEqual(receipt['cache_bindings'], self.caches)
        self.assertEqual(receipt['plan_sha256'], startup.digest(self.plan_path))
        self.assertEqual(receipt['capability'], [9, 0])
        self.assertNotIn('e2e_latency_s', receipt)
        self.assertTrue((self.root / 'new-run/traces/portable_native_startup_rank0.json').is_file())
        with patch.dict(os.environ, {**self.caches, 'RANK':'0', 'LOCAL_RANK':'0'}):
            with patch.dict(sys.modules, {'torch': self.fake_torch}):
                with patch.object(startup.importlib, 'import_module', return_value=SimpleNamespace(native=lambda: self.module)):
                    with self.assertRaises(FileExistsError):
                        startup.precompile(self.plan, 'lingbot14b')

    def test_preload_native_extension_must_remain_inside_private_cache(self):
        self.module.__file__ = str(self.root / 'outside.so')
        Path(self.module.__file__).write_bytes(b'x')
        with patch.dict(os.environ, {**self.caches, 'RANK':'0', 'LOCAL_RANK':'0'}):
            with patch.dict(sys.modules, {'torch': self.fake_torch}):
                with patch.object(startup.importlib, 'import_module', return_value=SimpleNamespace(native=lambda:self.module)):
                    with self.assertRaisesRegex(RuntimeError, 'inside the selected model cache'):
                        startup.precompile(self.plan, 'lingbot14b')

    def test_low_level_launcher_rejects_unverified_startup_only(self):
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location('launcher_fixture', ROOT / 'tools/launch_portable.py')
        launcher = module_from_spec(spec)
        spec.loader.exec_module(launcher)
        with self.assertRaisesRegex(ValueError, 'source-attested portable'):
            launcher.main(['run', '--native-startup-only', '--backbone', 'lingbot14b',
                           '--profile', 'off'])
        source = (ROOT/'tools/launch_portable.py').read_text()
        self.assertIn('"tools/portable_native_startup.py"', source)
        self.assertIn('"--startup-only"', source)

    def test_startup_only_never_calls_original_worker(self):
        with patch.object(startup, 'precompile', return_value=dict(status='NATIVE_EXTENSION_READY',
                       scope='pre_generation_build_only_not_formal_profile_or_video',backbone='LingBot14B')) as prep:
            with patch.dict(sys.modules, {'draftkv.backbones.worker': SimpleNamespace(main=lambda argv: (_ for _ in ()).throw(AssertionError('worker should not run')))}):
                self.assertEqual(startup.main(['--plan', str(self.plan_path), '--backbone','lingbot14b','--startup-only']),0)
                prep.assert_called_once()

    def test_normal_path_reaches_frozen_worker_main_after_precompile(self):
        calls = []
        def simulated_worker(args):
            calls.append(args)
            return 0
        with patch.object(startup, 'precompile', return_value=dict(status='NATIVE_EXTENSION_READY',
                      scope='pre_generation_build_only_not_formal_profile_or_video',backbone='LingBot14B')):
            with patch.dict(sys.modules, {'draftkv.backbones.worker': SimpleNamespace(main=simulated_worker)}):
                self.assertEqual(startup.main(['--plan',str(self.plan_path),'--backbone','lingbot14b']),0)
        self.assertEqual(calls, [['--plan', str(self.plan_path.resolve())]])


if __name__ == '__main__':
    unittest.main()
