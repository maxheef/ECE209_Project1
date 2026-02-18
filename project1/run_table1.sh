# Maxwell Heefner
# ECE209AS Trustworthy AI
# 18 Feb 26
# run_table1.sh
#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <model_path> <coco_val2014_dir> [seed]"
  exit 1
fi

# All parameters taken from original project
MODEL_PATH="$1"
COCO_VAL_DIR="$2"
SEED="${3:-55}"

ROOT_DIR="${ROOT_DIR:-/content/VCD_project}"
ORIG_DIR="$ROOT_DIR/originalProject"
OUT_DIR="$ROOT_DIR/project1/output"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CD_ALPHA="${CD_ALPHA:-1.0}"
CD_BETA="${CD_BETA:-0.2}"
NOISE_STEP="${NOISE_STEP:-500}"

mkdir -p "$OUT_DIR"

# Verify if the originalProject is available
if [[ ! -d "$ORIG_DIR/experiments" ]]; then
  echo "Missing experiments directory: $ORIG_DIR/experiments"
  exit 1
fi

if [[ ! -d "$COCO_VAL_DIR" ]]; then
  echo "Missing COCO val2014 directory: $COCO_VAL_DIR"
  exit 1
fi

# Run all POPE types (random and popular)
for split in random popular; do
  question_file="$ORIG_DIR/experiments/data/POPE/coco/coco_pope_${split}.json"

  # Run all method types (regular and vcd)
  for method in regular vcd; do
    answers_file="$OUT_DIR/llava15_coco_pope_${split}_answers_${method}_seed${SEED}.jsonl"
    metrics_file="$OUT_DIR/metrics_${split}_${method}_seed${SEED}.txt"

    echo "Running split=$split method=$method"

    # .py script taken from original experiment
    cmd=(
      "$PYTHON_BIN" ./eval/object_hallucination_vqa_llava.py
      --model-path "$MODEL_PATH"
      --question-file "$question_file"
      --image-folder "$COCO_VAL_DIR"
      --answers-file "$answers_file"
      --seed "$SEED"
    )

    # Setup for VCD
    if [[ "$method" == "vcd" ]]; then
      cmd+=(
        --use_cd
        --cd_alpha "$CD_ALPHA"
        --cd_beta "$CD_BETA"
        --noise_step "$NOISE_STEP"
      )
    fi

    (
      cd "$ORIG_DIR/experiments"
      "${cmd[@]}"
      "$PYTHON_BIN" ./eval/eval_pope.py --gt_files "$question_file" --gen_files "$answers_file" | tee "$metrics_file"
    )

    echo "Saved: $answers_file"
    echo "Saved: $metrics_file"
  done
done

# Return all outputs and format into a Table
"$PYTHON_BIN" - <<'PY' "$OUT_DIR" "$SEED"
import re
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
seed = sys.argv[2]

def parse_metrics(path: Path):
    text = path.read_text()
    def g(name):
        m = re.search(rf"^{name}:\s*([0-9.]+)", text, re.MULTILINE)
        return float(m.group(1)) if m else float('nan')
    return {
        'accuracy': g('Accuracy'),
        'precision': g('Precision'),
        'recall': g('Recall'),
        'f1': g('F1'),
    }

print("| Split | Method | Accuracy | Precision | Recall | F1 |")
print("|---|---|---:|---:|---:|---:|")
for split in ['random', 'popular']:
    for method in ['regular', 'vcd']:
        p = out_dir / f"metrics_{split}_{method}_seed{seed}.txt"
        m = parse_metrics(p)
        print(f"| {split} | {method} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")
PY
