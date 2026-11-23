# Third-party and external boundaries

The root Apache-2.0 does not relicense dependencies.

Optional original profiling controllers and environment dependencies are separately installed and retain their own distribution rights.

* Necessary shared `forcing.geometry`, `forcing.budget`, `worldkv.source`, `shadowkv.source` and `worldtrace_landmark.solarwm` project integration files retain original paths and bytes. Their loaded third-party model bodies have separate rights.
* The seven retained LingBot `rag_portable` namespace leaves carry the original LingBot CC-BY-NC-SA-4.0 license (`versions/*/src/draftkv/vendor/backbones/lingbot14/LICENSE.txt`). They are required by the direct camera-frustum import. Preserve attribution and all applicable NC/SA terms.
* Existing FreeKV/InfLLM/InfiniGen/ShadowKV/WorldKV license leaves are preserved where bound metadata reads them. Keeping a license leaf is not a claim that the corresponding full baseline is distributed or supported.
* B01 complete archive, derived CUDA/C++ headers, modified FlashAttention/CUDA extension binaries, model implementation sources and the Solar SP8 dependency overlay are external. Exact source locks identify their expected content; they do not grant redistribution rights. No archive or compiled binary is included.
* `src/draftkv/retrieval/ragged2d.py` is a package-local copy of the host-side Ragged-2D implementation from [official Anemoi](https://github.com/anemoi-project/anemoi/blob/270ddf8c3f0a43be47cf586873e7758bc76b822d/anemoi/layers/attention/mpa/ragged_2d.py), commit `270ddf8c3f0a43be47cf586873e7758bc76b822d`, Copyright 2026 Anemoi Project Contributors, Apache-2.0. The original license is preserved in `licenses/Anemoi-LICENSE`. This attribution identifies the third-party implementation and does not change its license.

Required model-source and CUDA/C++ extension downloads are specified in `scripts/native_sources.json`, with SHA-256 checks on fetched upstream files.

* `src/draftkv/vendor/evaluator_compat/torchvision_legacy_video.py` contains a project compatibility adapter with video-writing statements matching PyTorch/torchvision v0.23.0 `torchvision/io/video.py` (with `frames` renamed to `video_array`). Those portions are BSD-3-Clause, Copyright (c) Soumith Chintala 2016, All rights reserved. The complete original terms and disclaimer are retained in `licenses/Torchvision-LICENSE`. This is independent of the DraftKV attention implementation and is not relicensed as exclusively project Apache code.
