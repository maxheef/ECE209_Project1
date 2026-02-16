# Task A Starter (LLaVA-1.5 + POPE random/popular)

This folder provides a minimal workflow to reproduce Table 1-style results with the VCD repo code.

## Files
- `run_table1_repro.sh`: Runs 4 experiments (`regular`/`vcd` x `random`/`popular`) and computes metrics.
- `compute_pope_metrics.py`: Computes Accuracy, Precision, Recall, F1 from POPE labels and answer JSONL.
- `table1_from_outputs.py`: Prints a compact Table 1-style summary from generated outputs.

## Prerequisites
1. Create env and install VCD dependencies:
   - `conda create -yn vcd python=3.9`
   - `conda activate vcd`
   - `cd /Users/Max/Documents/Academia/UCLA/ECE 209AS/Assignments/VCD_project/VCD`
   - `pip install -r requirements.txt`
2. Ensure model weights and images exist locally:
   - LLaVA-1.5 model path (example: `/path/to/llava-v1.5-7b`)
   - COCO `val2014` image folder path (example: `/path/to/coco/val2014`)

## Run
From project root:

```bash
bash /Users/Max/Documents/Academia/UCLA/ECE 209AS/Assignments/VCD_project/myCode/run_table1_repro.sh \
  /path/to/llava-v1.5-7b \
  /path/to/coco/val2014 \
  55
```

Optional VCD hyperparameters via env vars:

```bash
CD_ALPHA=1.0 CD_BETA=0.2 NOISE_STEP=500 bash myCode/run_table1_repro.sh <model_path> <coco_val2014> 55
```

Outputs are written to:
- `/Users/Max/Documents/Academia/UCLA/ECE 209AS/Assignments/VCD_project/myCode/output`

## Notes
- The original repo script `VCD/experiments/cd_scripts/llava1.5_pope.bash` always sets `--use_cd`; it does not produce a true regular baseline by itself.
- These scripts use the same answer parsing rule as `VCD/experiments/eval/eval_pope.py` (`"yes"`/`"no"` substring matching).

## Colab
If you are running on Google Colab, use:
- `/Users/Max/Documents/Academia/UCLA/ECE 209AS/Assignments/VCD_project/myCode/COLAB_RUNBOOK.md`
- `/Users/Max/Documents/Academia/UCLA/ECE 209AS/Assignments/VCD_project/myCode/VCD_Table1_Colab.ipynb`
