# Profiling options

**English** | [简体中文](PROFILING_CN.md)

DraftKV provides two modes through the portable command-line interface:

| Mode | Purpose |
|---|---|
| `--profile off` (default) | Run the native worker without requesting an extra outer profiling summary. Suitable for functional checks and generation. |
| `--profile summary` | Collect portable execution diagnostics into `portable_profile_summary.json` in the external run directory. This is **not** a standardized benchmark or paper latency measurement. |

## Native extension startup check

To verify that the original native extension can be compiled and imported without generating a video:

```bash
./scripts/draftkv.sh run \
  --backbone lingbot1.3b \
  --env-root /external/draftkv-envs \
  --asset-root /external/draftkv-assets \
  --input /external/prepared/lingbot13/cases/YOUR_ORBIA_CASE_ID/request.json \
  --output /external/runs/native-startup/native.mp4 \
  --gpu 0 --profile off --startup-only
```

`PORTABLE_NATIVE_STARTUP_READY` means that the native extension import succeeded. It does **not** indicate a model forward pass or generated video. Each run requires a new external output directory.

## Generation with optional diagnostics

```bash
./scripts/draftkv.sh run \
  --backbone lingbot1.3b \
  --env-root /external/draftkv-envs \
  --asset-root /external/draftkv-assets \
  --input /external/prepared/lingbot13/cases/YOUR_ORBIA_CASE_ID/request.json \
  --output /external/runs/generation/native.mp4 \
  --gpu 0 --profile summary
```

`--profile summary` records deployment diagnostics when the required native instrumentation is available. For normal generation, use `--profile off`. A GPU shared with other tasks can be used for functionality checks if sufficient resources remain; reliable latency benchmarking requires an appropriately isolated GPU and consistent workloads. Do not interpret the diagnostic summary as a benchmark result.

For commands and input preparation, see [Quickstart](QUICKSTART.md). For prerequisites and dependencies, see [Installation](INSTALL.md). The original lower-level command interface is available via `./run.sh --help` and requires separately prepared original-schema plans and external dependencies.
