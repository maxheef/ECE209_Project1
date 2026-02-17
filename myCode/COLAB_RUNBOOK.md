# Colab Runbook

GPU H100 was used.

## 1) Configure
```python
REPO_URL = 'https://github.com/maxheef/ECE209_Project1.git'
BRANCH = 'main'
MODEL_PATH = 'liuhaotian/llava-v1.5-7b' 
SEED = 55
CD_ALPHA = 1.0
CD_BETA = 0.2
NOISE_STEP = 500
COCO_VAL_DIR = '/content/datasets/coco/val2014'

from pathlib import Path
Path('/tmp/vcd_params.env').write_text(
    f"REPO_URL='{REPO_URL}'\n"
    f"BRANCH='{BRANCH}'\n"
    f"MODEL_PATH='{MODEL_PATH}'\n"
    f"SEED='{SEED}'\n"
    f"CD_ALPHA='{CD_ALPHA}'\n"
    f"CD_BETA='{CD_BETA}'\n"
    f"NOISE_STEP='{NOISE_STEP}'\n"
    f"COCO_VAL_DIR='{COCO_VAL_DIR}'\n"
)
```

## 2) Clone/refresh repo
```bash
%%bash
set -euo pipefail
source /tmp/vcd_params.env

cd /content
if [ -d /content/VCD_project/.git ]; then
  git -C /content/VCD_project fetch --depth 1 origin "$BRANCH"
  git -C /content/VCD_project reset --hard FETCH_HEAD
else
  rm -rf /content/VCD_project
  GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch "$BRANCH" "$REPO_URL" /content/VCD_project
fi

if [ -d /content/VCD_project/VCD ] && [ -d /content/VCD_project/myCode ]; then
  PROJ=/content/VCD_project
elif [ -d /content/VCD_project/VCD_project/VCD ] && [ -d /content/VCD_project/VCD_project/myCode ]; then
  PROJ=/content/VCD_project/VCD_project
else
  echo 'Could not find expected layout under /content/VCD_project'
  find /content/VCD_project -maxdepth 3 -type d | sed -n '1,120p'
  exit 1
fi

echo "$PROJ" > /tmp/vcd_project_root.txt
echo "Using project root: $PROJ"
```

## 3) Create Python 3.9 env + install VCD deps
```bash
%%bash
set -euo pipefail
PROJ=$(cat /tmp/vcd_project_root.txt)

if [ ! -x /usr/local/miniconda/bin/conda ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p /usr/local/miniconda
fi

source /usr/local/miniconda/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
conda config --set always_yes yes --set changeps1 no
if ! conda env list | grep -q '^vcd39[[:space:]]'; then
  conda create -n vcd39 python=3.9
fi
conda activate vcd39
python -m pip install --upgrade pip
cd "$PROJ/VCD"
pip install -r requirements.txt
# Override PyTorch for H100 (sm_90) compatibility
pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch==2.2.2 torchvision==0.17.2
pip install --force-reinstall 'numpy<2'

echo /usr/local/miniconda/envs/vcd39/bin/python > /tmp/vcd_python_bin.txt
python -V
```

## 4) Download COCO val2014
```bash
%%bash
set -euo pipefail
source /tmp/vcd_params.env
mkdir -p /content/datasets/coco
cd /content/datasets/coco
if [ ! -d val2014 ]; then
  wget -q http://images.cocodataset.org/zips/val2014.zip
  unzip -q val2014.zip
fi
ls -la "$COCO_VAL_DIR" | sed -n '1,10p'
```

## 5) Run Table 1 (random/popular x regular/vcd)
```bash
%%bash
set -euo pipefail
source /tmp/vcd_params.env
PROJ=$(cat /tmp/vcd_project_root.txt)
PYBIN=$(cat /tmp/vcd_python_bin.txt)

cd "$PROJ"
PYTHON_BIN="$PYBIN" \
CD_ALPHA="$CD_ALPHA" CD_BETA="$CD_BETA" NOISE_STEP="$NOISE_STEP" \
bash myCode/run_table1_repro.sh \
  "$MODEL_PATH" \
  "$COCO_VAL_DIR" \
  "$SEED"
```

## 6) Show results
```bash
%%bash
set -euo pipefail
source /tmp/vcd_params.env
PROJ=$(cat /tmp/vcd_project_root.txt)
cd "$PROJ"
ls -la myCode/output
python3 myCode/table1_from_outputs.py --outputs myCode/output --gt-root VCD/experiments/data/POPE/coco --seed "$SEED"
```

Outputs stay local in: `/content/VCD_project/myCode/output` (or detected nested project root).
