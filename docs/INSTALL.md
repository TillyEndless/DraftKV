# Installation

## Requirements

- Linux x86_64 and NVIDIA Hopper H200 (SM90) for the currently supported native deployment path.
- Python 3.10 with `venv`/pip, compatible CPython development headers (`Python.h`, `pyconfig.h`), a C++ toolchain, Ninja, an NVIDIA driver and a matching CUDA toolkit.
- User-authorized original model weights and native dependencies, plus the dataset inputs required by the chosen model.
- A FlashAttention-2 wheel compiled for the selected Python/PyTorch/C++ ABI and a working CUDA toolchain for building pinned FA3 and B01 components.

Keep environments, model assets, native source archives, caches and generated files **outside** the source checkout. Different backbones have different pinned Python/PyTorch/CUDA dependencies: do not put them all in one virtual environment. The [installation manifest](../scripts/install_manifest.json) defines the dependency versions and available asset sources.

## Install a backbone

Choose a model from `./scripts/draftkv.sh list`; this example uses LingBot-World 1.3B:

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

The wheel filename is an example: use the actual file and its lowercase SHA-256 digest. `--authorized-gated` means you have independently obtained permission for gated sources. Use `--with-dataset` only when dataset assets are also required. `--jobs N` sets native build parallelism. Use `--dry-run` to inspect installation selections without modifying the environment.

The installer checks the [external dependency hashes](../EXTERNAL_DEPENDENCY_SHA256.json) and uses authorized sources from `--dependency-root`; it never fetches restricted third-party archives for you. The pinned FA3 build uses separately verified reference sources. Model assets may require upstream authentication, manual approval or third-party license compliance.

## Verify dependencies

```bash
./scripts/verify_install.sh \
  --backbone lingbot1.3b \
  --env-root /external/draftkv-envs \
  --asset-root /external/draftkv-assets \
  --scope inference
```

Use `--scope draftkv` for native dependencies, and `--scope benchmark` when datasets and evaluators are required. Incomplete or unauthorized dependencies are reported rather than replaced with unverified versions. A package/import check is not itself a GPU generation test.

## Input and execution

Use the [Quickstart](QUICKSTART.md) to prepare authorized ORBIA or WorldAttention inputs and run `./scripts/draftkv.sh run`. The portable runner needs a verified prepared request or original-schema plan, not arbitrary raw inputs. SolarWM may need `--native-source /external/authorized-solarwm-source`.

For native startup checks, use `--profile off --startup-only`; on a shared GPU, ensure enough CUDA memory is free. For full inference use `--profile off`; optional `--profile summary` diagnostics are explained in [Profiling](PROFILING.md).

## Advanced and legacy interfaces

The low-level `./run.sh` interface accepts externally prepared original-schema plans and workers. `./run.sh check` and `./run.sh --help` describe the supported commands. The original paths and source-validation requirements remain enforced. Build/install helper options are available from `scripts/setup_env.sh`, `scripts/prepare_assets.sh` and `scripts/verify_install.sh`.

Project-owned code is licensed as described in [LICENSE](../LICENSE); third-party components and model assets have independent distribution and use terms documented in [Third-party notices](THIRD_PARTY_NOTICES.md).
