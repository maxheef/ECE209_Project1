# Colab Runbook (Task A Reproduction)

Use this in a **GPU runtime** (A100/L4 preferred).

## 1) Setup + clone
```bash
%%bash
set -e
cd /content
if [ ! -d VCD_project ]; then
  GIT_TERMINAL_PROMPT=0 git clone --depth 1 https://github.com/maxheef/ECE209_Project1.git VCD_project
fi
cd VCD_project/VCD
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Download COCO val2014
```bash
%%bash
set -e
mkdir -p /content/datasets/coco
cd /content/datasets/coco
if [ ! -d val2014 ]; then
  wget -q http://images.cocodataset.org/zips/val2014.zip
  unzip -q val2014.zip
fi
```

## 3) Run Table 1 (random + popular, regular + VCD)
`model_path` can be a HF model id (e.g. `liuhaotian/llava-v1.5-7b`) or local path.

```bash
%%bash
set -e
cd /content/VCD_project
PYTHON_BIN=python3 \
CD_ALPHA=1.0 CD_BETA=0.2 NOISE_STEP=500 \
bash myCode/run_table1_repro.sh \
  liuhaotian/llava-v1.5-7b \
  /content/datasets/coco/val2014 \
  55
```

## 4) View metrics + table
```bash
%%bash
cd /content/VCD_project
ls -la myCode/output
cat myCode/output/metrics_random_regular_seed55.json
cat myCode/output/metrics_random_vcd_seed55.json
cat myCode/output/metrics_popular_regular_seed55.json
cat myCode/output/metrics_popular_vcd_seed55.json
python3 myCode/table1_from_outputs.py \
  --outputs myCode/output \
  --gt-root VCD/experiments/data/POPE/coco \
  --seed 55
```

## Practical notes
- First run is slow because model files are downloaded.
- If memory is tight, use a stronger GPU runtime or run one split at a time by editing `run_table1_repro.sh`.
- Keep seed fixed (`55`) for reproducibility in your report.
- Outputs remain local in `/content/VCD_project/myCode/output`.
