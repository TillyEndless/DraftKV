# DraftKV

**English** | [简体中文](README_CN.md)

**DraftKV: Protecting Persistent Worlds through Hierarchical Streaming Retrieval**

DraftKV is a **training-free historical KV retrieval** framework for persistent video and interactive world generation. It combines **Draft Grounding** and **Draft Refinement** for hierarchical retrieval, and **Streaming CG-Attention** for efficient CPU-to-GPU KV access.

## Supported backbones

| Backbone | Source directory |
|---|---|
| LongLive-1.3B | `versions/longlive` |
| LingBot-World 1.3B | `versions/lingbot13` |
| LingBot-World 14B | `versions/h200_large` |
| SolarWM-33B | `versions/h200_large` |

## Deployment quick start

Example: **Linux x86_64, NVIDIA H200 (SM90), Python 3.10, LingBot-World 1.3B**. Install the matching CUDA toolkit, acquire approved model/data access, and supply the original native dependencies and an **ABI-matched FlashAttention-2 wheel**. This is a source-only repository, not a self-contained model distribution.

**1. Clone and inspect supported models**

```bash
git clone https://github.com/TillyEndless/DraftKV-public.git
cd DraftKV-public
./scripts/draftkv.sh list
```

**2. Install the model environment, native dependencies and assets**

```bash
./scripts/draftkv.sh install \
  --backbone lingbot1.3b \
  --python python3.10 \
  --cuda-home /usr/local/cuda-12.6 \
  --env-root /external/draftkv-envs \
  --asset-root /external/draftkv-assets \
  --dependency-root /external/authorized-native-sources \
  --fa2-wheel /external/wheels/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl \
  --fa2-sha256 REPLACE_WITH_ACTUAL_64_CHARACTER_SHA256 \
  --python-dev-root /external/python-dev/usr/include \
  --authorized-gated
```

Use **real paths and the actual wheel SHA-256**; the wheel name above is illustrative. `--authorized-gated` is for users who already have upstream approval. The installer builds a pinned current-reference FA3 and a separately attested B01; it does **not** reconstruct the historical modified-FA3 binary. To inspect paths before installation, add `--dry-run`.

**3. Prepare a source-verified ORBIA case**

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

Supply an **authorized original ORBIA dataset and verified index**. If its distribution lock is not present in the source checkout, also provide `--lock /external/orbia/orbia480.lock.json`. LongLive instead uses its original WorldAttention manifest (`--manifest`); see [Quickstart](docs/QUICKSTART.md). A raw image, arbitrary prompt or invented camera trajectory is not a valid replacement for an original input case.

**4. Check native compilation/import without generating video**

```bash
./scripts/draftkv.sh run \
  --backbone lingbot1.3b \
  --env-root /external/draftkv-envs \
  --asset-root /external/draftkv-assets \
  --input /external/prepared/lingbot13/cases/YOUR_ORBIA_CASE_ID/request.json \
  --output /external/runs/lingbot13-startup/native.mp4 \
  --gpu 0 --profile off --startup-only
```

A successful check returns `PORTABLE_NATIVE_STARTUP_READY`, **not a generated video**. A shared GPU can be used if sufficient CUDA resources are available; it is not suitable for latency benchmarking. Use a **new output directory** for each run.

**5. Run full video generation**

```bash
./scripts/draftkv.sh run \
  --backbone lingbot1.3b \
  --env-root /external/draftkv-envs \
  --asset-root /external/draftkv-assets \
  --input /external/prepared/lingbot13/cases/YOUR_ORBIA_CASE_ID/request.json \
  --output /external/runs/lingbot13-generation/native.mp4 \
  --gpu 0 --profile off
```

All environment, assets, input plans, compilation caches and generated outputs should remain **outside the Git checkout**. See [Installation](docs/INSTALL.md) for exact prerequisites, and [Profiling](docs/PROFILING.md) for available profiling options.

## Main commands and arguments

| Command / option | Meaning |
|---|---|
| `list` | Show available backbone identifiers. |
| `install` / `prepare` / `run` | Install the environment and dependencies / prepare verified inputs / launch native startup or video generation. |
| `--backbone` | `longlive1.3b`, `lingbot1.3b`, `lingbot14b` or `solarwm33b`. |
| `--env-root` / `--asset-root` | External directories for isolated environments and model assets. |
| `--python` / `--cuda-home` / `--python-dev-root` | Base Python 3.10, CUDA toolkit root and matching CPython headers (install). |
| `--dependency-root` | Authorized original native dependencies, checked against repository SHA-256 locks (install). |
| `--fa2-wheel` / `--fa2-sha256` | Authorized ABI-matching FlashAttention-2 wheel and its actual lowercase SHA-256 (install). |
| `--authorized-gated` / `--with-dataset` | Confirm existing gated upstream access / optionally install dataset assets (install). |
| `--dataset-root` / `--case-id` | Authorized original dataset root and requested case identifier (prepare). |
| `--index` / `--index-sha256` / `--lock` | Original ORBIA index, its hash and an external distribution lock when required (prepare). |
| `--manifest` / `--native-outputs` | Authorized LongLive WorldAttention manifest and original output policy (prepare, LongLive only). |
| `--input` / `--output` | Prepared request or original-schema plan, and a **new** video output path (run). |
| `--native-source` | Authorized original SolarWM native source tree if not already source-bound (run, SolarWM only). |
| `--gpu` | Physical GPU index for `nvidia-smi` (default `0`; run). |
| `--profile off|summary` | `off` (default): functional execution without a new parent profile; `summary`: portable deployment diagnostics, **not** the formally admitted paper E2E profile. |
| `--startup-only` | Compile/import original native extension and stop before model generation (run, requires `--profile off`). |
| `--dry-run` | Inspect `install` or `run` selection without performing the operation. |
| `--jobs` / `--build-fa3` | Parallel native build jobs and required pinned FA3 build (`install`; FA3 building is enabled by default). |

For every option and the original low-level runner, use `./scripts/draftkv.sh <command> --help` or `./run.sh --help`.

## Documentation and release scope

[Detailed Quickstart](docs/QUICKSTART.md) · [Installation](docs/INSTALL.md) · [Profiling options](docs/PROFILING.md) · [Third-party notices](docs/THIRD_PARTY_NOTICES.md)

The four JSON manifests are required for [frozen source integrity](RETAINED_SOURCE_SHA256.json), [external dependencies](EXTERNAL_DEPENDENCY_SHA256.json), [original command identities](ORIGINAL_COMMAND_MAPPING.json) and [installation/assets](scripts/install_manifest.json); their existing paths and contents are preserved.

Project-owned code is [Apache-2.0](LICENSE). Third-party code, model weights and gated datasets have separate licenses and access requirements. This source-only repository does not include model checkpoints, gated datasets or private native source archives.
