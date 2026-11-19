#!/usr/bin/env bash
# CSE407 starter: check Python, find or install conda, build an env, install PyTorch + ML packages.
# Usage:  bash setup_env.sh [env_name] [python_version]
# Example: bash setup_env.sh cse407 3.11
set -euo pipefail

ENV_NAME="${1:-cse407}"
PY_VERSION="${2:-3.11}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQS="$HERE/requirements.txt"

say() { printf '\n==> %s\n' "$*"; }

# 1. Report the system Python. The conda env brings its own Python, so this is for information only.
say "Checking system Python"
if command -v python3 >/dev/null 2>&1; then
    echo "Found: $(python3 --version) at $(command -v python3)"
else
    echo "No system python3 found. This is fine: conda will install Python $PY_VERSION inside the env."
fi

# 2. Find conda. If it is missing, install Miniforge into $HOME/miniforge3.
say "Checking conda"
if ! command -v conda >/dev/null 2>&1; then
    for c in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
        if [ -x "$c/bin/conda" ]; then export PATH="$c/bin:$PATH"; break; fi
    done
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Installing Miniforge to $HOME/miniforge3 ..."
    ARCH="$(uname -m)"            # the Spark reports aarch64
    URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh"
    curl -fsSL "$URL" -o /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
    rm -f /tmp/miniforge.sh
    export PATH="$HOME/miniforge3/bin:$PATH"
fi
echo "Using $(conda --version)"

# 3. Create the env if it does not exist.
say "Creating env '$ENV_NAME' with Python $PY_VERSION"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Env '$ENV_NAME' already exists. Reusing it."
else
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
fi

# 4. Pick the PyTorch wheel index that matches the GPU driver.
#    Override with:  TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash setup_env.sh
say "Choosing the PyTorch build"
if [ -z "${TORCH_INDEX:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_VER="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\..*/\1/p' | head -n1)"
        echo "nvidia-smi reports CUDA major version: ${CUDA_VER:-unknown}"
        if [ "${CUDA_VER:-0}" -ge 13 ]; then TORCH_INDEX="https://download.pytorch.org/whl/cu130"
        else TORCH_INDEX="https://download.pytorch.org/whl/cu128"; fi
    else
        echo "No GPU driver found. Installing the CPU build."
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    fi
fi
echo "Index: $TORCH_INDEX"

# 5. Install torch first, then everything else.
say "Installing PyTorch"
conda run -n "$ENV_NAME" --no-capture-output python -m pip install --upgrade pip
conda run -n "$ENV_NAME" --no-capture-output python -m pip install torch torchvision --index-url "$TORCH_INDEX"

say "Installing the other packages from requirements.txt"
conda run -n "$ENV_NAME" --no-capture-output python -m pip install -r "$REQS"

# 6. Check that it works.
say "Checking the install"
conda run -n "$ENV_NAME" --no-capture-output python - <<'EOF'
import sys, torch
print("python      :", sys.version.split()[0])
print("torch       :", torch.__version__)
print("cuda usable :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device      :", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    print("matmul check:", float((x @ x).sum().isfinite()))
else:
    print("WARNING: torch cannot see the GPU. Ask the instructor before you continue.")
EOF

# 7. Short training run. It writes a checkpoint, a plot and the run files under runs/smoke/.
say "Running a short test (configs/smoke.yaml)"
rm -rf "$HERE/runs/smoke"
( cd "$HERE" && conda run -n "$ENV_NAME" --no-capture-output python train.py --config configs/smoke.yaml )
echo
echo "Files written by the test run:"
( cd "$HERE" && find runs/smoke -type f | sort )

say "Done. Activate the env with:  conda activate $ENV_NAME"
echo "If 'conda activate' fails, run 'conda init bash' once and open a new terminal."
