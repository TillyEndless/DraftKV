# Quickstart

DraftKV provides four model-specific, source-verified online bootstrap commands. The native installer targets **Linux x86_64 with an NVIDIA CUDA GPU**, Python 3.10 and a compatible CUDA compiler/toolkit. Hopper SM90 selects FA3; other architectures, including RTX 5090, select FA2.

Install Conda (Miniconda or Anaconda) to supply Python 3.10; a system-wide Python 3.10 is not required. The host must already have its NVIDIA driver, CUDA toolkit/compiler, Git and curl. Conda does not automatically install the GPU driver or system CUDA toolkit. Hugging Face access is taken from your existing account session or `HF_TOKEN`. No local FlashAttention wheel or manually assembled model directory is required for the **network bootstrap**. It downloads the official verified FA2 wheel for each model and builds current-reference FA3 additionally when Hopper SM90 is detected; project-specific B01 remains a separate stage.

## Clone

```bash
git clone https://github.com/TillyEndless/DraftKV-public.git
cd DraftKV-public
```

## Step 1: Create and activate Conda

```bash
conda create -n draftkv python=3.10 -y
conda activate draftkv
```

## Step 2: Install Backbone Python dependencies

For LingBot-World 1.3B:

```bash
./scripts/draftkv.sh environment --backbone lingbot1.3b --python "$(which python)"
```

Conda supplies the base interpreter; dependencies go into the independent `~/.cache/draftkv-install/<backbone>/venv/`, isolated from system Python and Conda environments (including `base` and `draftkv`). The CLI selects the Backbone venv automatically. Keep the Conda environment available because the venv relies on its interpreter.

## Step 3: Download model and dataset assets

```bash
./scripts/draftkv.sh assets --backbone lingbot1.3b \
  --asset-root "$HOME/.cache/draftkv/assets" --with-dataset
```

## Combined online bootstrap

After creating and activating Conda in Step 1, `bootstrap` combines the environment and asset stages with automatic FA2/FA3 installation and public native-source downloads. It defaults to `python3.10` on the active `PATH`. If that name is unavailable or selects another interpreter, explicitly use the active Conda Python:

```bash
./scripts/draftkv.sh bootstrap --backbone lingbot1.3b --with-dataset \
  --python "$(which python)"
```

Run the command for the desired Backbone. `--with-dataset` also fetches any verified dataset files published online.

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

Default storage is `~/.cache/draftkv-install/` for separate model environments and `~/.cache/draftkv/assets/` for checkpoints and online datasets. Use `--env-root` and `--asset-root` to choose a different external drive. Run the same command again to verify and reuse downloaded assets.

## Check native-source availability

```bash
./scripts/draftkv.sh native-deps --backbone lingbot1.3b --dry-run
```

The source selector identifies SHA-verified public upstream files, downloads those that are accessible and reports missing original modified files. These missing files cannot be replaced with similarly named upstream code. [Installation](INSTALL.md#native-dependency-preflight) documents the current per-Backbone source requirements.

## E2E inference boundary

**Online bootstrap and E2E video generation are different stages.** A successful bootstrap confirms that pinned Python packages, the selected FA2/FA3 installation components, and available published assets have been prepared. It does not imply that custom native/B01 source archives, a complete original ORBIA short case or LongLive benchmark input, the B01 native build or a GPU video run is available.

The full `install` / `prepare` / `run` workflow is source-locked and intentionally refuses missing originals. Once verified online releases exist for the remaining native and input files, they can be incorporated into the downloader without changing the scientific code.

See [Installation](INSTALL.md) for requirements and [Profiling](PROFILING.md) for the distinction between native startup checks, generated videos and deployment diagnostics.
