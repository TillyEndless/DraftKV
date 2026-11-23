#!/usr/bin/env python3
"""Prepare all frozen Bench1 native inputs; does not load a model or start GPU work."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from draftkv.backbones.bench1_inputs import main
if __name__ == "__main__":
    raise SystemExit(main())
