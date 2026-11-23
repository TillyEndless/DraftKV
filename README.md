# DraftKV

**English** | [简体中文](README_CN.md)

**DraftKV: Protecting Persistent Worlds through Hierarchical Streaming Retrieval**

DraftKV is a training-free historical key–value (KV) retrieval framework for persistent long-horizon video generation and interactive world models. It retrieves relevant historical context at two levels of granularity and streams selected full-resolution KV from CPU memory to the GPU.

## Overview

DraftKV combines three components:

- **Draft Grounding** selects geometrically relevant historical candidates using camera/frustum correspondence.
- **Draft Refinement** ranks candidate historical KV blocks using compact draft attention scores.
- **Streaming CG-Attention** overlaps CPU-to-GPU KV transfers with tiled attention computation.

Historical full-resolution KV can reside in CPU memory while a compact Persistent Draft Bank stays on the GPU. No model retraining is required.

### Supported Backbones

| Backbone | Source directory | CLI identifier |
| --- | --- | --- |
| LongLive-1.3B | `versions/longlive` | `longlive1.3b` |
| LingBot-World 1.3B | `versions/lingbot13` | `lingbot1.3b` |
| LingBot-World 14B | `versions/h200_large` | `lingbot14b` |
| SolarWM-33B | `versions/h200_large` | `solarwm33b` |

## Installation

**Platform:** Linux x86_64 with NVIDIA H200 (SM90). The host must already have a working NVIDIA driver, CUDA toolkit/build toolchain, Python 3.10 with venv/pip, Git and curl. The network installer handles the model-specific Python packages and available model/data downloads; it is **not** a bare-metal OS or CUDA driver installer.

Clone the project:

```bash
git clone https://github.com/TillyEndless/DraftKV-public.git
cd DraftKV-public
```

### Online environment and asset setup

Each command below creates an isolated, pinned Python environment for the selected model and downloads its **available online** model checkpoints and optional dataset assets. It uses the user's existing Hugging Face login or `HF_TOKEN`, verifies published SHA-256 hashes, and downloads byte-identical public native source files from fixed upstream commits. Downloaded files are cached outside the Git checkout.

**LongLive-1.3B**

```bash
./scripts/draftkv.sh bootstrap --backbone longlive1.3b --with-dataset
```

**LingBot-World 1.3B**

```bash
./scripts/draftkv.sh bootstrap --backbone lingbot1.3b --with-dataset
```

**LingBot-World 14B**

```bash
./scripts/draftkv.sh bootstrap --backbone lingbot14b --with-dataset
```

**SolarWM-33B**

```bash
./scripts/draftkv.sh bootstrap --backbone solarwm33b --with-dataset
```

No local wheel path, dependency archive, manually calculated SHA-256, dataset index, or other placeholder file path is needed for these **network bootstrap commands**. Download sizes vary substantially by model; reserve adequate free storage, particularly for the 14B and 33B checkpoints. Running the same command again verifies and reuses previously downloaded files.

By default, environments are stored in `~/.cache/draftkv-install/<backbone>/` and downloaded assets in `~/.cache/draftkv/assets/`. Both can be relocated via `--env-root` and `--asset-root`.

### What the network bootstrap does—and does not do

| Component | Network bootstrap |
| --- | --- |
| Pinned Python/PyTorch environment | Installed from online package repositories |
| Model weights and downloadable supplementary assets | Downloaded and SHA-256 verified |
| Published dataset assets (`--with-dataset`) | Downloaded when a verified URL exists |
| Exact upstream native source files | Downloaded if the official file matches the original source lock |
| Unpublished modified native/B01 source archives | **Not available online** |
| Original ORBIA short-case inputs or unreleased evaluation inputs | **Not available online** |
| FlashAttention/B01/FA3 full native build and video generation | **Not performed by `bootstrap`** |

**Important:** A successful `bootstrap` means that online software packages and available assets have been prepared; it does **not** mean end-to-end video generation is ready. The original source-locked inference path still requires project-specific modified native files and verified input cases that currently have **no confirmed public download source**. The installer will not substitute similarly named upstream code or report E2E success without running a video. See [Native dependency requirements](docs/INSTALL.md#native-dependency-preflight) and [Quickstart](docs/QUICKSTART.md) for the remaining inference prerequisites.

## Command-Line Reference

| Command / option | Purpose |
| --- | --- |
| `bootstrap --backbone MODEL` | Install the model's pinned environment and fetch available network assets and byte-identical public native sources |
| `--with-dataset` | Also download publicly available, SHA-locked dataset assets; not a replacement for unreleased original cases |
| `--env-root DIR` / `--asset-root DIR` | Select external environment and asset/cache locations |
| `--python python3.10` | Select the already-installed Python 3.10 interpreter |
| `--dry-run` | Inspect the plan without installing or downloading |
| `environment` / `assets` | Execute the Python environment or online asset phase separately |
| `native-deps` | Inspect the original source-lock requirements for a specific Backbone |
| `install` / `prepare` / `run` | Advanced native build, source-verified input preparation, and video inference (additional original prerequisites apply) |

Check the exact CLI options using `./scripts/draftkv.sh bootstrap --help`. Detailed instructions are available in [Installation](docs/INSTALL.md), [Quickstart](docs/QUICKSTART.md), and [Profiling](docs/PROFILING.md).

## Repository Structure

- `versions/` — preserved Backbone-specific implementations and configurations.
- `scripts/draftkv.sh` — public command-line entry point.
- `scripts/` and `tools/` — environment setup, verified downloads, source binding and native build utilities.
- `docs/` — deployment documentation, profiling options and [third-party notices](docs/THIRD_PARTY_NOTICES.md).
- `tests/` — offline unit and integration tests.

## License and Acknowledgements

Project-owned code is released under [Apache-2.0](LICENSE). Third-party code, model weights and datasets are governed by their respective upstream licenses and access terms; see [Third-party notices](docs/THIRD_PARTY_NOTICES.md).

DraftKV builds on the [LongLive](https://github.com/NVlabs/LongLive), [LingBot-World](https://github.com/Robbyant/lingbot-world-v2), and [SolarWM](https://github.com/Junchao-cs/SolarWM) model families.
