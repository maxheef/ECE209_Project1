#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <model_path> <coco_val2014_dir> [seed] [output_dir]"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="$ROOT_DIR/VCD/experiments"
MODEL_PATH="$1"
COCO_VAL_DIR="$2"
SEED="${3:-55}"
OUTPUT_DIR="${4:-$ROOT_DIR/myCode/output}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CD_ALPHA="${CD_ALPHA:-1.0}"
CD_BETA="${CD_BETA:-0.2}"
NOISE_STEP="${NOISE_STEP:-500}"

mkdir -p "$OUTPUT_DIR"

if [[ ! -d "$EXP_DIR" ]]; then
  echo "Missing experiments directory: $EXP_DIR"
  exit 1
fi

if [[ ! -d "$COCO_VAL_DIR" ]]; then
  echo "Missing COCO val2014 directory: $COCO_VAL_DIR"
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "No python interpreter found (tried: $PYTHON_BIN, python3)"
    exit 1
  fi
fi

for split in random popular; do
  QUESTION_FILE="$EXP_DIR/data/POPE/coco/coco_pope_${split}.json"

  for method in regular vcd; do
    ANSWERS_FILE="$OUTPUT_DIR/llava15_coco_pope_${split}_answers_${method}_seed${SEED}.jsonl"

    echo "Running split=$split method=$method"

    CMD=(
      "$PYTHON_BIN" ./eval/object_hallucination_vqa_llava.py
      --model-path "$MODEL_PATH"
      --question-file "$QUESTION_FILE"
      --image-folder "$COCO_VAL_DIR"
      --answers-file "$ANSWERS_FILE"
      --seed "$SEED"
    )

    if [[ "$method" == "vcd" ]]; then
      CMD+=(
        --use_cd
        --cd_alpha "$CD_ALPHA"
        --cd_beta "$CD_BETA"
        --noise_step "$NOISE_STEP"
      )
    fi

    (
      cd "$EXP_DIR"
      "${CMD[@]}"
    )

    "$PYTHON_BIN" "$ROOT_DIR/myCode/compute_pope_metrics.py" \
      --gt "$QUESTION_FILE" \
      --pred "$ANSWERS_FILE" \
      --pretty > "$OUTPUT_DIR/metrics_${split}_${method}_seed${SEED}.json"

    echo "Saved: $ANSWERS_FILE"
    echo "Saved: $OUTPUT_DIR/metrics_${split}_${method}_seed${SEED}.json"
  done
done

"$PYTHON_BIN" "$ROOT_DIR/myCode/table1_from_outputs.py" \
  --outputs "$OUTPUT_DIR" \
  --gt-root "$EXP_DIR/data/POPE/coco" \
  --seed "$SEED"
