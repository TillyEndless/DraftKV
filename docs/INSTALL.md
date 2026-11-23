# Installation

## Requirements

- Linux x86_64 and NVIDIA Hopper H200 (SM90) for the currently supported native deployment path.
- Python 3.10 with `venv`/pip, compatible CPython development headers (`Python.h`, `pyconfig.h`), a C++ toolchain, Ninja, an NVIDIA driver and a matching CUDA toolkit.
- User-authorized original model weights and native dependencies, plus the dataset inputs required by the chosen model.
- Network access to the official FlashAttention GitHub release (the installer automatically downloads a SHA-256-pinned FA2 wheel matching the selected Python/PyTorch/C++ ABI), plus a working CUDA toolchain for pinned FA3 and B01. For offline environments, provide an authorized ABI-compatible wheel with `--fa2-wheel` and `--fa2-sha256`.

Keep environments, model assets, native source archives, caches and generated files **outside** the source checkout. Different backbones have different pinned Python/PyTorch/CUDA dependencies: do not put them all in one virtual environment. The [installation manifest](../scripts/install_manifest.json) defines the dependency versions and available asset sources.

## Install a backbone

Choose a model from `./scripts/draftkv.sh list`; this example uses LingBot-World 1.3B:

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

If the system Python 3.10 development headers are not available, add `--python-dev-root /path/to/python3.10/includes`. FlashAttention-2 is downloaded from an official pinned release URL, verified against its SHA-256 and exact size, cached outside the source tree, and installed into the model environment. No manual wheel path is required. Optional offline override: `--fa2-wheel /path/to/matching.whl --fa2-sha256 ACTUAL_SHA256`. `--authorized-gated` requires independently granted upstream access; `--with-dataset` installs dataset assets when needed. `--jobs N` adjusts native build parallelism, and `--dry-run` inspects the selection without installing.

The installer checks the [external dependency hashes](../EXTERNAL_DEPENDENCY_SHA256.json) and uses authorized sources from `--dependency-root`; it never fetches restricted third-party archives for you. The pinned FA3 build uses separately verified reference sources. Model assets may require upstream authentication, manual approval or third-party license compliance.

## Native dependency preflight

Each model uses a compact, **SHA-256-equivalent** view of the original external source inventory. The archived `EXTERNAL_DEPENDENCY_SHA256.json` (500 records) is retained unchanged for the frozen original launchers; `scripts/native_sources.json` selects only the relevant source-lock files and records verified upstream URLs for byte-identical source files.

| Backbone | Native source-lock files selected (SM90 inference) | Verified upstream downloads | Remaining original-only files |
|---|---:|---:|---:|
| LongLive-1.3B | 49 | 28 | 21 |
| LingBot-World 1.3B | 53 | 20 | 33 |
| LingBot-World 14B | 53 | 20 | 33 |
| SolarWM-33B | 4 | 0 | 4 |

```bash
./scripts/draftkv.sh native-deps --backbone lingbot1.3b --dry-run
```

`--profile inference` (default) excludes unrelated model families and SM120 headers on H200; `--profile full` includes the complete 100-entry legacy closure for the selected source version. Some apparently optional training/profiling files remain in the source-lock group because the immutable original planner audits the complete original model tree. The B01 evidence archive also remains necessary for the original B01 source integrity check. **No frozen scientific source, hash or native ABI is modified by this selection.**

The downloader fetches only publicly available files that were verified byte-for-byte against a fixed upstream Git commit and the original SHA-256. If any original-only file is absent, `install` fails **before** downloading heavy packages or model weights and identifies the missing files. Provide those exact originals with `--dependency-root /path/to/original-source-tree` (version-prefixed layout); a similar upstream file or a different CUDA kernel is not an acceptable replacement. SolarWM also requires its separate original native model source at inference time.

## Verify dependencies

```bash
./scripts/verify_install.sh \
  --backbone lingbot1.3b \
  --env-root ${HOME}/.cache/draftkv/envs \
  --asset-root ${HOME}/.cache/draftkv/assets \
  --scope inference
```

Use `--scope draftkv` for native dependencies, and `--scope benchmark` when datasets and evaluators are required. Incomplete or unauthorized dependencies are reported rather than replaced with unverified versions. A package/import check is not itself a GPU generation test.

## Input and execution

Use the [Quickstart](QUICKSTART.md) to prepare authorized ORBIA or WorldAttention inputs and run `./scripts/draftkv.sh run`. The portable runner needs a verified prepared request or original-schema plan, not arbitrary raw inputs. SolarWM may need `--native-source /external/authorized-solarwm-source`.

For native startup checks, use `--profile off --startup-only`; on a shared GPU, ensure enough CUDA memory is free. For full inference use `--profile off`; optional `--profile summary` diagnostics are explained in [Profiling](PROFILING.md).

## Advanced and legacy interfaces

The low-level `./run.sh` interface accepts externally prepared original-schema plans and workers. `./run.sh check` and `./run.sh --help` describe the supported commands. The original paths and source-validation requirements remain enforced. Build/install helper options are available from `scripts/setup_env.sh`, `scripts/prepare_assets.sh` and `scripts/verify_install.sh`.

Project-owned code is licensed as described in [LICENSE](../LICENSE); third-party components and model assets have independent distribution and use terms documented in [Third-party notices](THIRD_PARTY_NOTICES.md).
