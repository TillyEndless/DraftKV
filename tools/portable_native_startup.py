#!/usr/bin/env python3
"""Portable-only native precompilation before the frozen DraftKV worker/timer.

Build/load the original device-route extension in the *worker process* before
calling original worker.main(). This closes the h200_large profile-preparation
gap without editing frozen algorithms, profiling code or scientific plans.
Does not warm sampling kernels, run a model, or bypass compile admission.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

BACKBONES = {
    'longlive1.3b': 'LongLive1.3B',
    'lingbot1.3b': 'LingBot1.3B',
    'lingbot14b': 'LingBot14B',
    'solarwm33b': 'SolarWM33B',
}
CACHE_NAMES = ('DRAFTKV_PREPACK_BUILD', 'TORCH_EXTENSIONS_DIR',
               'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for blob in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(blob)
    return h.hexdigest()


def validate_plan(plan, requested):
    if requested not in BACKBONES or plan.get('backbone') != BACKBONES[requested]:
        raise RuntimeError('portable worker and requested backbone differ')
    if plan.get('algorithm') != 'draftkv' or plan.get('sp') != 1:
        raise RuntimeError('portable precompile only supports original SP1 DraftKV')
    settings = plan.get('algorithm_plugin', {}).get('config', {})
    if not (settings.get('device_archive_plan') is True and
            settings.get('gpu_krope_8x16') is True and
            settings.get('retrieval_ablation') == 'full' and
            isinstance(settings.get('b01'), dict)):
        raise RuntimeError('portable native preload requires approved device-archive Full DraftKV plan')
    if not Path(plan['run_dir']).is_absolute():
        raise RuntimeError('native startup receipt must be external and absolute')
    return plan


def precompile(plan, requested):
    validate_plan(plan, requested)
    # The installing model's own private cache must be used. Never silently
    # fall back to default /tmp or another user's compiler cache.
    bindings = {key: os.environ.get(key) for key in CACHE_NAMES}
    if any(not val or not Path(val).is_absolute() for val in bindings.values()):
        raise RuntimeError('portable native compiler caches must be absolute and explicitly scoped')
    import torch
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    if rank != 0 or int(os.environ.get('RANK', '0')) != 0:
        raise RuntimeError('portable native startup supports SP1 rank0 only')
    torch.cuda.set_device(rank)
    cap = tuple(torch.cuda.get_device_capability(rank))
    if cap != (9, 0):
        raise RuntimeError('portable B01 startup requires original H200 SM90 GPU')
    # The original module is already LRU-cached in-process after this call.
    # Thus the *same frozen* worker later uses it without an extra Ninja build
    # inside run_profiled's compilation observer (14B's previous blocker).
    module = importlib.import_module('draftkv.streaming.device_route').native()
    binary = Path(module.__file__).resolve()
    if not binary.is_file() or not binary.is_relative_to(Path(bindings['DRAFTKV_PREPACK_BUILD']).resolve()):
        raise RuntimeError('native module was not built inside the selected model cache')
    receipt = dict(schema_version=1, status='NATIVE_EXTENSION_READY',
                   scope='pre_generation_build_only_not_formal_profile_or_video',
                   backbone=plan['backbone'], plan_sha256=digest(plan['_plan_path']),
                   python=sys.executable, torch=str(torch.__version__),
                   torch_cuda=torch.version.cuda, capability=list(cap),
                   rank=rank, cache_bindings=bindings,
                   extension=dict(name=module.__name__, module_file=str(binary),
                                  sha256=digest(binary),
                                  source_identity=module.source_identity))
    folder = Path(plan['run_dir']) / 'traces'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / 'portable_native_startup_rank0.json'
    with output.open('x', encoding='utf-8') as file:
        json.dump(receipt, file, indent=2, sort_keys=True)
        file.write('\n')
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--backbone', choices=tuple(BACKBONES), required=True)
    p.add_argument('--startup-only', action='store_true',
                   help='stop after native compiler/module readiness; no model load or video')
    args = p.parse_args(argv)
    path = args.plan.resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    validate_plan(data, args.backbone)
    data['_plan_path'] = str(path)
    receipt = precompile(data, args.backbone)
    print(json.dumps(dict(status=receipt['status'], scope=receipt['scope'],
                          backbone=receipt['backbone']), sort_keys=True), flush=True)
    if args.startup_only:
        return 0
    # Call the unchanged original entry in this same Python process. Extension
    # lru_cache survives, but no algorithm / timer / generation function is patched.
    from draftkv.backbones.worker import main as original_main
    return original_main(['--plan', str(path)])


if __name__ == '__main__':
    sys.exit(main())
