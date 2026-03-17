import os
import sys
import torch
import gc
import subprocess
import json
from pathlib import Path

# --- Constants ---
ROOT = '/content/VCD_project'
ORIG = f'{ROOT}/originalProject'
EXP_DIR = f'{ORIG}/experiments'
OUT_DIR = f'{ROOT}/output'
COCO_VAL = '/content/datasets/coco/val2014'
MODEL_PATH = 'liuhaotian/llava-v1.5-7b'
SEED = '55'

def run_vqa_step(split, method):
    os.makedirs(OUT_DIR, exist_ok=True)
    question_file = f"{EXP_DIR}/data/POPE/coco/coco_pope_{split}.json"
    answers_file = f"{OUT_DIR}/ans_{split}_{method}.jsonl"
    metrics_file = f"{OUT_DIR}/metrics_{split}_{method}.json" # Save as JSON for easy table building

    print(f"\n>>> Task A: Running {split} | {method}")

    cmd = [
        sys.executable, "./eval/object_hallucination_vqa_llava.py",
        "--model-path", MODEL_PATH,
        "--question-file", question_file,
        "--image-folder", COCO_VAL,
        "--answers-file", answers_file,
        "--seed", SEED
    ]

    if method == "vcd":
        cmd += ["--use_cd", "--cd_alpha", "1.0", "--cd_beta", "0.2", "--noise_step", "500"]

    os.chdir(EXP_DIR)
    subprocess.run(cmd, check=True)

    # Run Eval and capture output to parse metrics
    eval_cmd = [sys.executable, "./eval/eval_pope.py", "--gt_files", question_file, "--gen_files", answers_file]
    result = subprocess.run(eval_cmd, capture_output=True, text=True, check=True)
    
    # Logic to save the result.stdout into metrics_file (JSON) goes here...
    # For now, let's just save the raw text
    Path(metrics_file).write_text(result.stdout)
    
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    for split in ['random', 'popular']:
        for method in ['regular', 'vcd']:
            run_vqa_step(split, method)
