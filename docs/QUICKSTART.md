# Quickstart

The CLI supports four backbones: `longlive1.3b`, `lingbot1.3b`, `lingbot14b`, and `solarwm33b`. The examples below use **Linux x86_64, NVIDIA H200 (SM90), Python 3.10**, and LingBot-World 1.3B. Supply authorized original model checkpoints, datasets and native dependencies separately.

## 1. Clone and list models

```bash
git clone https://github.com/TillyEndless/DraftKV-public.git
cd DraftKV-public
./scripts/draftkv.sh list
```

## 2. Install

```bash
./scripts/draftkv.sh install \
  --backbone lingbot1.3b \
  --python python3.10 \
  --cuda-home /usr/local/cuda-12.6 \
  --env-root /external/draftkv-envs \
  --asset-root /external/draftkv-assets \
  --dependency-root /external/authorized-native-sources \
  --fa2-wheel /external/wheels/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl \
  --fa2-sha256 REPLACE_WITH_ACTUAL_SHA256 \
  --python-dev-root /external/python-dev/usr/include \
  --authorized-gated
```

Use the actual paths, the actual SHA-256 of a compatible FlashAttention-2 wheel and your own upstream access authorization. The installer provisions isolated per-backbone environments, verifies external sources and builds the required native components. For prerequisites and optional flags, see [Installation](INSTALL.md).

## 3. Prepare an input case

```bash
./scripts/draftkv.sh prepare \
  --backbone lingbot1.3b \
  --env-root /external/draftkv-envs \
  --dataset-root /external/orbia \
  --index /external/orbia/installed-files.json \
  --index-sha256 REPLACE_WITH_ACTUAL_INDEX_SHA256 \
  --case-id YOUR_ORBIA_CASE_ID \
  --output /external/prepared/lingbot13
```

The ORBIA dataset, case and index must originate from an authorized original distribution. If its distribution lock is not shipped, add `--lock /external/orbia/orbia480.lock.json`. For LongLive, use the original text-to-video input protocol and its authorized WorldAttention manifest via `--manifest`.

## 4. Optional native readiness check

```bash
./scripts/draftkv.sh run \
  --backbone lingbot1.3b \
  --env-root /external/draftkv-envs \
  --asset-root /external/draftkv-assets \
  --input /external/prepared/lingbot13/cases/YOUR_ORBIA_CASE_ID/request.json \
  --output /external/runs/startup/native.mp4 \
  --gpu 0 --profile off --startup-only
```

This compiles/imports the native extension only; it does not generate video. A completely idle GPU is not required, but sufficient CUDA memory is necessary.

## 5. Generate video

```bash
./scripts/draftkv.sh run \
  --backbone lingbot1.3b \
  --env-root /external/draftkv-envs \
  --asset-root /external/draftkv-assets \
  --input /external/prepared/lingbot13/cases/YOUR_ORBIA_CASE_ID/request.json \
  --output /external/runs/generation/native.mp4 \
  --gpu 0 --profile off
```

Use a **fresh external output directory** for every run. Input plans must be source-verified original-schema requests; a raw photo or arbitrary prompt is not a substitute. `--profile summary` is an optional deployment diagnostic, explained in [Profiling](PROFILING.md).

Use `./scripts/draftkv.sh install --help`, `prepare --help` or `run --help` for every supported option. External assets, environments, compiled binaries, caches and output videos should remain outside this repository.
