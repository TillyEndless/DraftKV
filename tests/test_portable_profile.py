"""CPU-only portable profile diagnostic tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

P = Path(__file__).resolve().parents[1] / "tools/portable_profile.py"
SPEC = importlib.util.spec_from_file_location("draftkv_portable_profile", P)
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


class PortableProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        (self.run / "native.mp4").write_bytes(b"video fixture")
        self.d = dict(route_count=900, device_route_calls=900, slot_calls=750,
            actual_native_backend=3, forward_transactions=30,
            completed_transactions=30, transfer={"h2d_bytes": 10240},
            receive_capacity_max_bytes=50331648)
        self.save()

    def save(self):
        (self.run / "runtime_profile.json").write_text(json.dumps(self.d))

    def test_valid_worker_profile_summary(self):
        result = profile.summarize(self.run, 14.2, "a" * 64)
        self.assertEqual(result["status"], "PORTABLE_WORKER_PROFILE_COMPLETE")
        self.assertEqual(result["actual_attention_backend"], "FA3")
        self.assertEqual(result["worker_wall_s"], 14.2)
        self.assertEqual(result["h2d_bytes"], 10240)
        self.assertIn("NOT original", result["scope"])
        self.assertTrue((self.run / "portable_profile_summary.json").exists())
        with self.assertRaisesRegex(RuntimeError, "output exists"):
            profile.summarize(self.run, 14.3, "a" * 64)

    def test_built_in_worker_fa3_trace_summary(self):
        (self.run/"runtime_profile.json").unlink()
        traces=self.run/"traces"
        traces.mkdir()
        events=[dict(event="native_attention_backend",resolved=3),
            dict(event="algorithm_forward_begin",forward_id=1),
            dict(event="attention"),dict(event="draftkv_b01_attention",
                 consumer_backend="FA3",final_join=True,
                 h2d_bytes=4096,kv_slots=2,receive_capacity_bytes=50331648),
            dict(event="algorithm_forward_complete",forward_id=1),
            dict(event="worker_complete")]
        (traces/"backbone_rank0.jsonl").write_text("".join(
            json.dumps(e)+"\n" for e in events))
        (traces/"gpu_profile_rank0.json").write_text(json.dumps(
            dict(status="COMPLETE",scope="generation_through_vae_decode_excluding_video_encoding",
                 e2e_latency_s=8.1,diffusion_latency_s=4.2,
                 peak_pytorch_allocated_bytes=123456)))
        result=profile.summarize(self.run, 12.6,"f"*64)
        self.assertEqual(result["source"],"built_in_worker_trace")
        self.assertEqual(result["b01_fa3_attention_calls"],1)
        self.assertEqual(result["kv_slot_calls"],1)
        self.assertEqual(result["h2d_bytes"],4096)
        self.assertEqual(result["native_gpu_profile"]["generation_vae_decode_s"],8.1)

    def prepare_model_profile(self, gpu):
        (self.run / "runtime_profile.json").unlink()
        traces = self.run / "traces"
        traces.mkdir()
        events = [dict(event="native_attention_backend", resolved=3),
                  dict(event="algorithm_forward_begin", forward_id=99),
                  dict(event="draftkv_b01_attention", consumer_backend="FA3",
                       final_join=True, kv_slots=3, h2d_bytes=8192,
                       receive_capacity_bytes=33554432),
                  dict(event="algorithm_forward_complete", forward_id=99),
                  dict(event="worker_complete")]
        (traces / "backbone_rank0.jsonl").write_text("".join(
            json.dumps(e) + "\n" for e in events))
        (traces / "gpu_profile_rank0.json").write_text(json.dumps(gpu))

    def test_solar_worker_native_write_scope_kept_separate(self):
        self.prepare_model_profile(dict(
            status="COMPLETE", backbone="SolarWM33B",
            scope="generation_through_vae_decode_and_native_video_write",
            e2e_latency_s=17.8, diffusion_latency_s=8.4,
            peak_allocated_bytes=424242))
        result = profile.summarize(self.run, 31.7, "e" * 64)
        native = result["native_gpu_profile"]
        self.assertEqual(native["generation_measured_s"], 17.8)
        self.assertEqual(native["generation_vae_decode_and_write_s"], 17.8)
        self.assertNotIn("generation_vae_decode_s", native)
        self.assertTrue(native["includes_native_video_write"])
        self.assertEqual(native["peak_pytorch_allocated_bytes"], 424242)

    def test_mismatched_solar_scope_and_compile_contamination_fail_closed(self):
        gpu = dict(status="COMPLETE", backbone="SolarWM33B",
                   scope="generation_through_vae_decode_excluding_video_encoding")
        self.prepare_model_profile(gpu)
        with self.assertRaisesRegex(RuntimeError, "model-specific measurement scope"):
            profile.summarize(self.run, 20.0, "e" * 64)
        gpu["scope"] = "generation_through_vae_decode_and_native_video_write"
        gpu["status"] = "COMPILE_CONTAMINATED"
        gpu["latency_admission"] = {"reasons": ["UNVERIFIED_BUILD_DRIVER"]}
        (self.run / "traces/gpu_profile_rank0.json").write_text(json.dumps(gpu))
        with self.assertRaisesRegex(RuntimeError, "UNVERIFIED_BUILD_DRIVER"):
            profile.summarize(self.run, 20.0, "e" * 64)
        self.assertFalse((self.run / "portable_profile_summary.json").exists())

    def test_built_in_rejects_no_fa3_history(self):
        (self.run/"runtime_profile.json").unlink()
        traces=self.run/"traces"
        traces.mkdir()
        (traces/"backbone_rank0.jsonl").write_text(
            json.dumps(dict(event="worker_complete"))+"\n")
        (traces/"gpu_profile_rank0.json").write_text(json.dumps(
            dict(status="COMPLETE",scope="generation_through_vae_decode_excluding_video_encoding")))
        with self.assertRaisesRegex(RuntimeError,"incomplete"):
            profile.summarize(self.run, 2,"f"*64)

    def test_missing_or_empty_video_rejected(self):
        (self.run / "native.mp4").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing"):
            profile.summarize(self.run, 1, "b" * 64)

    def test_missing_historical_retrieval_rejected(self):
        self.d["route_count"] = 0
        self.save()
        with self.assertRaisesRegex(RuntimeError, "no recorded historical"):
            profile.summarize(self.run, 2, "c" * 64)

    def test_non_fa3_or_incomplete_transactions_rejected(self):
        self.d["actual_native_backend"] = 2
        self.save()
        with self.assertRaisesRegex(RuntimeError, "FA3"):
            profile.summarize(self.run, 2, "d" * 64)
        self.d["actual_native_backend"] = 3
        self.d["completed_transactions"] = 29
        self.save()
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            profile.summarize(self.run, 2, "d" * 64)

if __name__ == "__main__":
    unittest.main()
