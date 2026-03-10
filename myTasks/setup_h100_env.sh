# Maxwell Heefner
# ECE209AS Trustworthy AI
# 18 Feb 26
# setup_h100_env.sh
#!/usr/bin/env bash

# Documentation: CODEX Extension utilized to troubleshoot runtime errors for compatibility
# and aided in generation of this document to ensure a Colab H100 GPU had proper configurations
set -euo pipefail

ROOT_DIR="${1:-/content/VCD_project}"
ORIG_DIR="$ROOT_DIR/originalProject"
ENV_NAME="${2:-vcd39}"

if [[ ! -d "$ORIG_DIR" ]]; then
  echo "Missing originalProject directory: $ORIG_DIR"
  exit 1
fi

if [[ ! -x /usr/local/miniconda/bin/conda ]]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p /usr/local/miniconda
fi

source /usr/local/miniconda/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
conda config --set always_yes yes --set changeps1 no

if ! conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
  conda create -n "$ENV_NAME" python=3.9
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip

cd "$ORIG_DIR"
pip install -r requirements.txt
# H100 needs newer torch than repo pin.
pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch==2.2.2 torchvision==0.17.2
# Keep ABI compatibility with older compiled deps.
pip install --force-reinstall 'numpy<2'

echo "/usr/local/miniconda/envs/${ENV_NAME}/bin/python" > /tmp/myTasks_python_bin.txt
python - <<'PY'
import torch, numpy
print('Torch:', torch.__version__)
print('NumPy:', numpy.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY
