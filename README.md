<p align="center">
  <img src="assets/readme/hero.svg" width="100%" alt="DraftKV: training-free hierarchical historical KV retrieval and streaming CPU-to-GPU attention for persistent video world models">
</p>

<p align="center">
  <a href="README.md"><strong>English</strong></a> · <a href="README_CN.md">简体中文</a> ·
  <a href="#how-it-works">How it works</a> · <a href="#quick-start">Quick start</a> ·
  <a href="#documentation">Documentation</a>
</p>

**DraftKV** is a training-free historical key–value (KV) retrieval framework for persistent long-horizon video generation and interactive world models. It finds relevant past context without keeping the full KV history on the GPU, then streams selected KV blocks into attention.

## How it works

<p align="center">
  <img src="assets/readme/workflow.png" width="100%" alt="DraftKV pipeline: query, geometry-based Draft Grounding, attention-based Draft Refinement, and tile-wise CPU-to-GPU KV streaming into GPU attention">
</p>

1. **Draft Grounding** uses camera/frustum correspondence to narrow historical retrieval to geometrically relevant candidates.
2. **Draft Refinement** scores candidate KV blocks with compact draft attention, selecting the history most relevant to the current query.
3. **Streaming CG-Attention** overlaps CPU-to-GPU transfers of selected full-resolution KV tiles with attention computation.

The **Persistent Draft Bank** stays on the GPU for retrieval; full-resolution historical KV is stored in CPU memory. DraftKV requires **no model retraining**.

## Quick start

**Requirements:** Linux x86_64, an NVIDIA CUDA GPU with a working driver and compatible CUDA toolkit/build tools, Git, curl, and Conda.

```bash
git clone https://github.com/TillyEndless/DraftKV.git
cd DraftKV

conda create -n draftkv python=3.10 pip -y
conda activate draftkv

./scripts/draftkv.sh bootstrap --backbone lingbot1.3b --with-dataset
```

This prepares the pinned Python environment, available model/data assets, and architecture-selected FlashAttention dependencies. The installer uses the **active Conda environment** rather than creating a nested virtual environment. Use a separate Conda environment for each Backbone.

To reuse existing model files or move downloads off the home filesystem, set `--asset-root /path/to/assets`; use `--env-root /path/to/cache` for installation caches. Files with matching SHA-256 hashes are reused.

For native builds, inference inputs, and video generation, follow the [full installation guide](docs/INSTALL.md) and [quickstart](docs/QUICKSTART.md).

## Supported backbones

| Model | `--backbone` | Source |
| --- | --- | --- |
| LongLive-1.3B | `longlive1.3b` | [`versions/longlive`](versions/longlive) |
| LingBot-World 1.3B | `lingbot1.3b` | [`versions/lingbot13`](versions/lingbot13) |
| LingBot-World 14B | `lingbot14b` | [`versions/h200_large`](versions/h200_large) |
| SolarWM-33B | `solarwm33b` | [`versions/h200_large`](versions/h200_large) |

Replace `lingbot1.3b` in the bootstrap command with the corresponding identifier. Use `./scripts/draftkv.sh bootstrap --help` to inspect CLI options. On Hopper SM90 (H100/H200), the installer can build FA3 in addition to the model-native FA2 dependency; other supported NVIDIA architectures use FA2.

## Documentation

- **[Installation](docs/INSTALL.md)** — environment setup, model assets, and native dependencies.
- **[Quickstart](docs/QUICKSTART.md)** — preparation and inference workflow.
- **[Profiling](docs/PROFILING.md)** — profiling entry points and measurements.
- **[Third-party notices](docs/THIRD_PARTY_NOTICES.md)** — upstream components and licenses.

The [`scripts/draftkv.sh`](scripts/draftkv.sh) CLI provides the public entry point. Backbone-specific source snapshots are under [`versions/`](versions), with offline checks under [`tests/`](tests).

## License and acknowledgements

Project-owned code is licensed under [Apache-2.0](LICENSE). Third-party source code, model weights, and datasets remain subject to their own licenses and access conditions; see [third-party notices](docs/THIRD_PARTY_NOTICES.md).

DraftKV builds on [LongLive](https://github.com/NVlabs/LongLive), [LingBot-World](https://github.com/Robbyant/lingbot-world-v2), and [SolarWM](https://github.com/Junchao-cs/SolarWM).
