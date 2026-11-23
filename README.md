# DraftKV

**English** | [简体中文](README_CN.md)

**DraftKV: Protecting Persistent Worlds through Hierarchical Streaming Retrieval**

DraftKV is a training-free historical key-value (KV) retrieval framework for long-horizon video generation and interactive world models. It uses camera-aware and attention-aware retrieval to recover relevant past context, then streams selected full-resolution KV from CPU memory to the GPU for efficient attention computation.

The repository provides source-preserving implementations for LongLive-1.3B, LingBot-World 1.3B/14B, and SolarWM-33B, along with a command-line workflow for installation, input preparation, and inference.

## Method Overview

DraftKV combines three components:

- **Draft Grounding:** identifies geometrically relevant historical candidates using camera/frustum information.
- **Draft Refinement:** ranks candidate KV blocks with compact draft attention relevance scores.
- **Streaming CG-Attention:** overlaps CPU-to-GPU KV transfer with tiled attention computation.

The retrieval stages operate without model retraining. Full-resolution historical KV can reside in CPU memory, while compact draft descriptors remain on the GPU.

### Supported Backbones

| Backbone | Source directory |
| --- | --- |
| LongLive-1.3B | `versions/longlive` |
| LingBot-World 1.3B | `versions/lingbot13` |
| LingBot-World 14B | `versions/h200_large` |
| SolarWM-33B | `versions/h200_large` |

Use `./scripts/draftkv.sh list` to display the corresponding CLI backbone identifiers.

## Installation

The documented native deployment target is **Linux x86_64 with an NVIDIA H200 (SM90)**. Install Python 3.10, the matching CUDA toolkit and compiler, Ninja, and CPython development headers before running the installer. Model checkpoints, gated benchmark inputs, and restricted native dependencies must be obtained separately under their respective licenses. The installer automatically downloads a **SHA-256-pinned official FlashAttention-2 wheel** for the selected PyTorch environment.

Clone the source repository:

```bash
git clone https://github.com/TillyEndless/DraftKV-public.git
cd DraftKV-public
./scripts/draftkv.sh list
```

For example, install the LingBot-World 1.3B environment and model assets:

```bash
./scripts/draftkv.sh install \
  --backbone lingbot1.3b \
  --python python3.10 \
  --cuda-home /usr/local/cuda-12.6 \
  --env-root ${HOME}/.cache/draftkv/envs \
  --asset-root ${HOME}/.cache/draftkv/assets \
  --dependency-root /path/to/authorized-native-sources \
  --authorized-gated
```

Replace the example paths with real values. The matching FA2 wheel is **downloaded, SHA-256 verified, cached and installed automatically** from the official [FlashAttention releases](https://github.com/Dao-AILab/flash-attention/releases); no local wheel or digest is required. For offline installation, use `--fa2-wheel PATH --fa2-sha256 DIGEST`. `--authorized-gated` indicates that upstream access has already been granted. Add `--dry-run` to inspect the configuration without installing.

If the system lacks Python 3.10 development headers, additionally set `--python-dev-root /path/to/python3.10/includes`. The installer builds pinned reference FA3 and the necessary B01 native components; this is not a claim of binary identity with any earlier modified implementation. See [Installation](docs/INSTALL.md) for dependency requirements and verification commands.

## LingBot-World Quick Start

After installation, prepare a case from an **authorized original ORBIA dataset**:

```bash
./scripts/draftkv.sh prepare \
  --backbone lingbot1.3b \
  --env-root ${HOME}/.cache/draftkv/envs \
  --dataset-root /external/orbia \
  --index /external/orbia/installed-files.json \
  --index-sha256 REPLACE_WITH_ACTUAL_INDEX_SHA256 \
  --lock /external/orbia/orbia480.lock.json \
  --case-id YOUR_ORBIA_CASE_ID \
  --output /external/prepared/lingbot13
```

Use `--lock` when the original dataset distribution lock is not available in the checkout. This workflow requires genuine source-verified prompts, frames, and camera/control information; arbitrary images or fabricated camera trajectories are not supported inputs. LongLive instead uses its original WorldAttention manifest via `prepare --manifest`.

Optionally check native extension compilation and import **without generating a video**:

```bash
./scripts/draftkv.sh run \
  --backbone lingbot1.3b \
  --env-root ${HOME}/.cache/draftkv/envs \
  --asset-root ${HOME}/.cache/draftkv/assets \
  --input /external/prepared/lingbot13/cases/YOUR_ORBIA_CASE_ID/request.json \
  --output /external/runs/lingbot13-startup/native.mp4 \
  --gpu 0 --profile off --startup-only
```

A successful startup check reports `PORTABLE_NATIVE_STARTUP_READY`; it does not load the model or produce a video. To run full video generation, use a **different output directory** and omit `--startup-only`:

```bash
./scripts/draftkv.sh run \
  --backbone lingbot1.3b \
  --env-root ${HOME}/.cache/draftkv/envs \
  --asset-root ${HOME}/.cache/draftkv/assets \
  --input /external/prepared/lingbot13/cases/YOUR_ORBIA_CASE_ID/request.json \
  --output /external/runs/lingbot13-generation/native.mp4 \
  --gpu 0 --profile off
```

Keep environments, checkpoints, datasets, native build caches, prepared inputs, and outputs **outside the Git checkout**. See the [Quickstart guide](docs/QUICKSTART.md) for model-specific input requirements.

### Command-Line Options

| Command or option | Description |
| --- | --- |
| `list` / `install` / `prepare` / `run` | List models / install dependencies / prepare verified inputs / check native startup or generate video. |
| `--backbone` | `longlive1.3b`, `lingbot1.3b`, `lingbot14b`, or `solarwm33b`. |
| `--env-root` / `--asset-root` | External model environment and asset directories. |
| `--python` / `--cuda-home` / `--python-dev-root` | Python 3.10, CUDA toolkit path, and matching CPython headers (install). |
| `--dependency-root` | Authorized original native sources, validated with SHA-256 (install). |
| `--fa2-wheel` / `--fa2-sha256` | Optional offline override: local ABI-compatible FA2 wheel and SHA-256; normally fetched automatically (install). |
| `--authorized-gated` / `--with-dataset` | Confirm existing upstream permission / also install dataset assets (install). |
| `--dataset-root` / `--case-id` | Original dataset location and requested case (prepare). |
| `--index` / `--index-sha256` / `--lock` | Original ORBIA index, its checksum, and external dataset lock when needed (prepare). |
| `--manifest` / `--native-outputs` | Original LongLive manifest and output policy (prepare). |
| `--input` / `--output` | Source-verified request/plan and a new output path (run). |
| `--native-source` | Authorized original SolarWM native source directory when needed (run). |
| `--gpu` | Physical GPU index, default `0` (run). |
| `--profile off|summary` | `off` for normal execution; `summary` for optional deployment diagnostics, not paper benchmark results. |
| `--startup-only` | Build/import the native extension and exit before model inference (run). |
| `--dry-run` / `--jobs` | Preview installation/run selection / set native build parallelism (where supported). |

Full command help: `./scripts/draftkv.sh install --help`, `prepare --help`, and `run --help`. Profiling modes are described in [Profiling options](docs/PROFILING.md).

## Repository Layout

- `versions/` — preserved per-backbone source implementations and scientific configurations.
- `scripts/draftkv.sh` — public installation, preparation, and inference entry point.
- `scripts/` and `tools/` — dependency provisioning and runtime integration utilities.
- `docs/` — [Quickstart](docs/QUICKSTART.md), [Installation](docs/INSTALL.md), [Profiling](docs/PROFILING.md), and [Third-party notices](docs/THIRD_PARTY_NOTICES.md).
- `tests/` — CPU unit and integration tests.

The repository also includes checksum and deployment manifests to verify preserved sources, external dependencies, installation assets, and original command identities.

## License and Acknowledgements

Project-owned code is released under [Apache-2.0](LICENSE). Third-party components retain their own licenses, including applicable noncommercial restrictions; review the [third-party notices](docs/THIRD_PARTY_NOTICES.md) before redistribution or use.

DraftKV integrates with the [LongLive](https://github.com/NVlabs/LongLive), [LingBot-World](https://github.com/robbyant/lingbot-world-v2), and [SolarWM](https://github.com/Junchao-cs/SolarWM) model families. Model weights, gated datasets, and private native source archives are not bundled with this source release.
