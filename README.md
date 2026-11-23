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

**Platform:** Linux x86_64 with an NVIDIA CUDA GPU. Install **Conda (Miniconda or Anaconda)** to provide Python 3.10; no system-wide Python 3.10 installation is required. The host must already have a working driver, compatible CUDA toolkit/build toolchain, Git and curl. Conda does not automatically install the GPU driver or system CUDA toolkit. The network installer handles the model-specific Python packages and available model/data downloads; it is **not** a bare-metal OS or CUDA driver installer.

Clone the project:

```bash
git clone https://github.com/TillyEndless/DraftKV-public.git
cd DraftKV-public
```

### Step 1: Create and activate a Conda environment

Conda provides the base Python 3.10 interpreter. Create and activate it before running the installation commands:

```bash
conda create -n draftkv python=3.10 pip -y
conda activate draftkv
```

### Step 2: Install the selected Backbone's Python dependencies

For LingBot-World 1.3B, pass the active Conda interpreter explicitly:

```bash
./scripts/draftkv.sh environment --backbone lingbot1.3b --python "$(which python)"
```

The installer now installs pinned Python/PyTorch packages **directly into the activated `draftkv` Conda environment**; it does not create a nested venv. The active Conda interpreter is used throughout `environment`, `bootstrap`, `install`, `prepare`, and `run`. Package caches, native builds, and verification receipts remain outside Conda at `~/.cache/draftkv-install/lingbot1.3b/`. The active environment must be Python 3.10 and must not be Conda `base`.

For faster installation on servers that reach the official PyTorch wheel index slowly, you may explicitly use the CUDA-matched Aliyun wheel index:

```bash
./scripts/draftkv.sh environment --backbone lingbot1.3b \
  --torch-index-url https://mirrors.aliyun.com/pytorch-wheels/cu124
```

The original exact PyTorch package versions remain pinned. A third-party wheel mirror is **opt-in** and its binary contents are not independently SHA-256 verified by this installer; use the official index if byte-level provenance is required.

**Use one Conda environment per Backbone.** Their locked Torch/CUDA and other dependency versions differ. For other models, use names such as `draftkv-longlive`, `draftkv-lingbot14`, and `draftkv-solarwm` (`conda create -n NAME python=3.10 -y`). The installer prevents an environment already assigned to one Backbone from being reused by another.

### Step 3: Download model and dataset assets

```bash
./scripts/draftkv.sh assets --backbone lingbot1.3b \
  --asset-root "$HOME/.cache/draftkv/assets" --with-dataset
```

The base environment step does not install native FlashAttention or build B01/FA3. For the full native `install` stage, the installer detects the selected GPU (physical `--gpu` index, default `0`): **Hopper SM90 (H100/H200) selects FA3; other NVIDIA architectures (including RTX 5090/SM120) select FA2 and skip FA3 compilation**. FA2 remains a model-native dependency on Hopper. Streaming CG-Attention's own autotuning is unchanged; the selected installer backend is not a claim of completed GPU inference.

### One-command network bootstrap for all four Backbones

First create and activate a dedicated Conda environment in Step 1. `bootstrap` automatically selects its `bin/python`; the explicit `--python` form below is optional and must refer to that same interpreter:

```bash
./scripts/draftkv.sh bootstrap --backbone lingbot1.3b --with-dataset \
  --python "$(which python)"
```

For normal setup, the `bootstrap` command combines the environment and online asset stages, **automatically installs FA2 on all GPUs and builds FA3 additionally on Hopper SM90**, then fetches the exact byte-identical public native source subset. It uses your existing Hugging Face login or `HF_TOKEN`, verifies published SHA-256 hashes, and stores downloads outside the Git checkout.

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

By default, native build artifacts, installation receipts and download caches reside in `~/.cache/draftkv-install/<backbone>/` and assets in `~/.cache/draftkv/assets/`; the Python packages reside in the activated Conda environment. `--env-root` and `--asset-root` relocate these external directories, **not** the Conda interpreter.

### What the network bootstrap does—and does not do

| Component | Network bootstrap |
| --- | --- |
| Pinned Python/PyTorch environment | Installed from online package repositories |
| Model weights and downloadable supplementary assets | Downloaded and SHA-256 verified |
| Published dataset assets (`--with-dataset`) | Downloaded when a verified URL exists |
| Exact upstream native source files | Downloaded if the official file matches the original source lock |
| Unpublished modified native/B01 source archives | **Not available online** |
| Original ORBIA short-case inputs or unreleased evaluation inputs | **Not available online** |
| FlashAttention-2 (all GPUs, model-native dependency) | Official pinned wheel automatically downloaded, verified and installed |
| FlashAttention-3 (Hopper SM90 only) | Current-reference FA3 automatically built and installed from pinned upstream sources |
| Project-specific B01 native build and video generation | **Not performed by `bootstrap`** |

**Important:** A successful `bootstrap` means that online software packages and available assets have been prepared; it does **not** mean end-to-end video generation is ready. The original source-locked inference path still requires project-specific modified native files and verified input cases that currently have **no confirmed public download source**. The installer will not substitute similarly named upstream code or report E2E success without running a video. See [Native dependency requirements](docs/INSTALL.md#native-dependency-preflight) and [Quickstart](docs/QUICKSTART.md) for the remaining inference prerequisites.

## Command-Line Reference

| Command / option | Purpose |
| --- | --- |
| `bootstrap --backbone MODEL` | Install the model's pinned environment and fetch available network assets and byte-identical public native sources |
| `--with-dataset` | Also download publicly available, SHA-locked dataset assets; not a replacement for unreleased original cases |
| `--env-root DIR` / `--asset-root DIR` | Select external environment and asset/cache locations |
| `--python "$(which python)"` | Optionally confirm the interpreter is the activated Conda Python 3.10 |
| `--torch-index-url URL` | Optional official/Aliyun CUDA-matched wheel index for PyTorch installation |
| `--env-mode {auto,conda,venv}` | Auto-select active Conda (non-base); `venv` preserves the legacy isolated-venv workflow |
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
