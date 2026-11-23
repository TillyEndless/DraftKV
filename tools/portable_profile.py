"""Deployment-only performance summary for the preserved DraftKV model worker.

Supports both an original external campaign observer's runtime_profile.json and
the retained built-in worker's native trace/gpu_profile files. No paper E2E
admission is claimed; never inserts CUDA synchronizations or changes kernels.
"""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def built_in_summary(directory):
    trace = directory / "traces/backbone_rank0.jsonl"
    gpu = directory / "traces/gpu_profile_rank0.json"
    if not trace.is_file() or not gpu.is_file():
        raise RuntimeError("built-in worker trace and GPU profile are both required")
    gp = json.loads(gpu.read_text())
    # The frozen SolarWM worker includes native video writing in its measured
    # interval; the other workers exclude it. Never relabel these as the same
    # metric or equate either with the paper's campaign-admitted E2E numbers.
    expected_scope = ("generation_through_vae_decode_and_native_video_write"
                      if gp.get("backbone") == "SolarWM33B" else
                      "generation_through_vae_decode_excluding_video_encoding")
    if gp.get("scope") != expected_scope:
        raise RuntimeError("native GPU profiler reported a mismatched model-specific measurement scope")
    if gp.get("status") != "COMPLETE":
        reasons = gp.get("latency_admission", {}).get("reasons", [])
        raise RuntimeError("native GPU timing not admitted (" + str(gp.get("status")) +
                           "): " + ", ".join(map(str, reasons[:5])))
    counts = {}
    began, completed = set(), set()
    h2d_bytes, slots, max_receive = 0, 0, 0
    native_fa3, completion = False, False
    with trace.open(encoding="utf-8") as stream:
        events = (json.loads(line) for line in stream)
        for event in events:
            name = event.get("event")
            counts[name] = counts.get(name, 0) + 1
            if name == "algorithm_forward_begin":
                began.add(event["forward_id"])
            if name == "algorithm_forward_complete":
                completed.add(event["forward_id"])
            if name == "native_attention_backend":
                if event.get("resolved") != 3:
                    raise RuntimeError("native attention backend was not FA3")
                native_fa3 = True
            if name == "draftkv_b01_attention":
                if event.get("consumer_backend") != "FA3" or event.get("final_join") is not True:
                    raise RuntimeError("B01 did not execute the expected FA3 final join")
                h2d_bytes += int(event.get("h2d_bytes", 0))
                slots += int(event.get("kv_slots", 0) > 0)
                max_receive = max(max_receive, int(event.get("receive_capacity_bytes", 0)))
            if name == "worker_complete":
                completion = True
    if not native_fa3 or not completion or not began or began != completed:
        raise RuntimeError("incomplete built-in worker transaction or FA3 trace")
    if not counts.get("draftkv_b01_attention") or not slots or not max_receive:
        raise RuntimeError("no recorded B01/FA3 historical streaming activity")
    native_measurement = dict(
        scope=gp["scope"],
        generation_measured_s=gp.get("e2e_latency_s"),
        diffusion_s=gp.get("diffusion_latency_s"),
        peak_pytorch_allocated_bytes=gp.get("peak_pytorch_allocated_bytes", gp.get("peak_allocated_bytes")),
        includes_native_video_write=gp["scope"] == "generation_through_vae_decode_and_native_video_write")
    native_measurement["generation_vae_decode_and_write_s" if native_measurement["includes_native_video_write"]
                       else "generation_vae_decode_s"] = gp.get("e2e_latency_s")
    return dict(source="built_in_worker_trace", original_profile_sha256=sha(trace),
                gpu_profile_sha256=sha(gpu),
                attention_calls=counts.get("attention", 0),
                b01_fa3_attention_calls=counts["draftkv_b01_attention"],
                completed_forward_transactions=len(completed),
                actual_attention_backend="FA3",
                kv_slot_calls=slots, receive_capacity_max_bytes=max_receive,
                h2d_bytes=h2d_bytes,
                native_gpu_profile=native_measurement)


def summarized_campaign_observer(directory):
    path = directory / "runtime_profile.json"
    data = json.loads(path.read_text())
    required = ("route_count", "device_route_calls", "slot_calls", "actual_native_backend",
                "forward_transactions", "completed_transactions", "transfer")
    if any(k not in data for k in required):
        raise RuntimeError("campaign worker profile lacks required completion fields")
    if not isinstance(data["transfer"], dict) or data["actual_native_backend"] != 3:
        raise RuntimeError("production FA3 profile incomplete")
    if data["forward_transactions"] <= 0 or data["forward_transactions"] != data["completed_transactions"]:
        raise RuntimeError("campaign worker forward transactions incomplete")
    if data["route_count"] <= 0 or data["slot_calls"] <= 0:
        raise RuntimeError("no recorded historical retrieval/slot streaming")
    return dict(source="original_campaign_observer", original_profile_sha256=sha(path),
                completed_forward_transactions=data["completed_transactions"],
                draftkv_route_count=data["route_count"],
                device_route_count=data["device_route_calls"],
                kv_slot_calls=data["slot_calls"],
                actual_attention_backend="FA3",
                h2d_bytes=data["transfer"].get("h2d_bytes"),
                receive_capacity_max_bytes=data.get("receive_capacity_max_bytes"))


def summarize(run_dir, worker_wall_s, plan_hash, output=None):
    run_dir = Path(run_dir)
    video = Path(output) if output else run_dir / "native.mp4"
    if not video.is_file() or video.stat().st_size == 0:
        raise RuntimeError("video output missing after successful worker exit")
    if (run_dir / "runtime_profile.json").is_file():
        summary = summarized_campaign_observer(run_dir)
    else:
        summary = built_in_summary(run_dir)
    result = dict(schema=1, status="PORTABLE_WORKER_PROFILE_COMPLETE",
        scope="deployment runtime diagnostic, NOT original Table 1/2 admitted E2E",
        plan_sha256=plan_hash,
        worker_wall_s=worker_wall_s,
        worker_wall_scope="full worker process including model loading; not paper E2E",
        video_sha256=sha(video), video_bytes=video.stat().st_size,
        **summary)
    target = run_dir / "portable_profile_summary.json"
    if target.exists():
        raise RuntimeError("portable profile output exists; no silent overwrite")
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
