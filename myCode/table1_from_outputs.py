#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from compute_pope_metrics import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", required=True, help="Directory containing answer jsonl files")
    parser.add_argument("--gt-root", required=True, help="Path to VCD/experiments/data/POPE/coco")
    parser.add_argument("--seed", type=int, default=55)
    args = parser.parse_args()

    rows = []
    for split in ["random", "popular"]:
        gt = os.path.join(args.gt_root, f"coco_pope_{split}.json")
        for method in ["regular", "vcd"]:
            pred = os.path.join(
                args.outputs,
                f"llava15_coco_pope_{split}_answers_{method}_seed{args.seed}.jsonl",
            )
            if not os.path.exists(pred):
                continue
            m = compute_metrics(gt, pred)
            rows.append((split, method, m))

    if not rows:
        print("No matching outputs found.")
        return

    print("| Split | Method | Accuracy | Precision | Recall | F1 |")
    print("|---|---|---:|---:|---:|---:|")
    for split, method, m in rows:
        print(
            f"| {split} | {method} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |"
        )


if __name__ == "__main__":
    main()
