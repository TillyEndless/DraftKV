# Installation

DraftKV supports four model-specific Python environments and source-verified network asset downloads. This guide separates **online bootstrap** from **end-to-end inference**, because not every original native source and benchmark input is publicly available.

## Requirements

- Linux x86_64 and an NVIDIA CUDA GPU with a compatible driver and CUDA toolkit/compiler. Hopper SM90 automatically selects FA3; other NVIDIA architectures automatically select FA2 without Hopper-only compilation.
- Python 3.10 with `venv` and `pip`, Git, curl, sufficient disk space for the selected model and dataset.
- Network access to PyPI, PyTorch wheel indices, official upstream GitHub commits and Hugging Face.
- For gated Hugging Face assets, an existing user-approved Hugging Face login or `HF_TOKEN`. The installer neither requests nor bypasses access permission.

This installer does not provision the operating system, system CUDA toolkit or GPU driver.

## Network bootstrap by Backbone

Clone the repository and select **one** of the commands below:

```bash
git clone https://github.com/TillyEndless/DraftKV-public.git
cd DraftKV-public
```

| Backbone | CLI name |
| --- | --- |
| LongLive-1.3B | `longlive1.3b` |
| LingBot-World 1.3B | `lingbot1.3b` |
| LingBot-World 14B | `lingbot14b` |
| SolarWM-33B | `solarwm33b` |

### LongLive-1.3B

```bash
./scripts/draftkv.sh bootstrap --backbone longlive1.3b --with-dataset
```

### LingBot-World 1.3B

```bash
./scripts/draftkv.sh bootstrap --backbone lingbot1.3b --with-dataset
```

### LingBot-World 14B

```bash
./scripts/draftkv.sh bootstrap --backbone lingbot14b --with-dataset
```

### SolarWM-33B

```bash
./scripts/draftkv.sh bootstrap --backbone solarwm33b --with-dataset
```

The bootstrap installs pinned Python/PyTorch packages, downloads the verified official FA2 wheel for model-native dependencies, **builds current-reference FA3 on Hopper SM90 only**, downloads SHA-verified online model/data assets, and fetches byte-identical public native sources from pinned upstream commits. The optional dataset flag downloads **only published, checksum-verifiable assets**.

Default storage: `~/.cache/draftkv-install/<backbone>/` for isolated environments, `~/.cache/draftkv/assets/` for online model/data files. Set `--env-root` and `--asset-root` to different external locations if more space is needed. An interrupted download can be resumed where the server permits; verified asset files are reused.

## Native dependency preflight

The original source integrity locks are preserved without modification. The compact [per-Backbone source selector](../scripts/native_sources.json) references fixed official Git commits when their exact bytes match the original files. A source file with a different SHA-256 cannot replace a DraftKV-modified counterpart.

| Backbone | Original native files selected for SM90 | Verified public upstream files | Original files without a verified URL |
| --- | ---: | ---: | ---: |
| LongLive-1.3B | 49 | 28 | 21 |
| LingBot-World 1.3B | 53 | 20 | 33 |
| LingBot-World 14B | 53 | 20 | 33 |
| SolarWM-33B | 4 | 0 | 4 |

To see exactly what is available and missing:

```bash
./scripts/draftkv.sh native-deps --backbone lingbot1.3b --dry-run
```

These counts describe the current frozen source-lock inventory, not the model weights. The full 500-row `EXTERNAL_DEPENDENCY_SHA256.json` is retained for original launcher compatibility. The per-Backbone selector omits unrelated model-source families and SM120 headers when targeting H200, but keeps any helper files referenced by complete original model source locks. B01 integrity verification also requires the original evidence archive.

## What bootstrap does not provide

The original modified native sources, custom B01 archives, some SolarWM native code and complete ORBIA short-case / LongLive benchmark inputs have no confirmed byte-identical public download locations. The online bootstrap **does not** reconstruct these files, build project-specific B01, or generate a video. Its Hopper FA3 build is the pinned public reference, not a reconstruction of any unavailable historical binary. The separate `install` / `prepare` / `run` commands remain available when the required source-locked originals and input cases are accessible; they must not be presented as a working zero-input network-only pipeline.

The bootstrap output explicitly reports `native_sources_unavailable` and `end_to_end_generation: NOT_EXECUTED`. A future complete E2E release would require verified online distribution of the missing native source identities and benchmark inputs, followed by a new actual H200 run.

## Further Reading

- [Quickstart](QUICKSTART.md) — model selection and online downloads.
- [Profiling](PROFILING.md) — native startup versus functional generation and diagnostics.
- [Third-party notices](THIRD_PARTY_NOTICES.md) — licenses and upstream attribution.
