# macOS `marl` environment

`environment-marl-macos.yml` is the cross-platform deployment file generated
from the local Windows `marl` environment on 2026-08-13.

## Install

Install Miniforge/Conda first, then from the repository root run:

```bash
conda env create -f environment-marl-macos.yml
conda activate marl-macos
```

The file detects the Mac architecture through pip markers:

- Apple Silicon (`arm64`, M1/M2/M3/M4): PyTorch `2.11.0` macOS arm64 wheel;
- Intel (`x86_64`): PyTorch `2.2.2` macOS x86_64 wheel.

The NumPy pin follows the same compatibility split: Apple Silicon uses
`numpy==2.2.6`, while Intel uses `numpy==1.26.4` because the older Intel
PyTorch wheel was built against the NumPy 1.x ABI.

## Verify

```bash
python - <<'PY'
import platform
import sys
import torch
import numpy
import scipy
import gymnasium

print("python:", sys.version.split()[0])
print("machine:", platform.machine())
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("mps available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
print("numpy/scipy/gymnasium:", numpy.__version__, scipy.__version__, gymnasium.__version__)
PY
```

## Important cross-system differences

The local Windows training environment is `torch==2.12.0+cu126`, which is a
CUDA build and cannot be reproduced on macOS because macOS does not provide
CUDA. The macOS environment is therefore suitable for CPU execution and, on
Apple Silicon, contains an MPS-capable PyTorch build. The current
`train_mec.py` device selection checks CUDA and otherwise chooses `cpu`; it
does not automatically select MPS. Do not claim a Mac MPS run is numerically
identical to the Windows CUDA runs without a separate matched validation.

The source code falls back from `gym` to `gymnasium`, so this configuration
does not install the legacy `gym` package. `shapely` and `paramiko` were not
installed in the local `marl` environment: Shapely is optional in the MEC
geometry path, while Paramiko is only needed by the optional remote-sync
script. Install them separately only if those specific paths are used:

```bash
python -m pip install shapely==2.0.7 paramiko==3.5.1
```

## Basic CPU smoke test

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_dynamic_md.py
```

PowerShell `.ps1` launchers can be used on macOS only after installing
PowerShell (`pwsh`). For a first deployment, use the Python test command above
to verify imports before starting a long training run.
