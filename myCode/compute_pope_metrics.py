#!/usr/bin/env python3
import argparse
import json
from typing import Dict, List


def _load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_metrics(gt_path: str, pred_path: str) -> Dict[str, float]:
    gt_rows = _load_jsonl(gt_path)
    pred_rows = _load_jsonl(pred_path)

    pred_by_id = {row["question_id"]: row for row in pred_rows}

    tp = tn = fp = fn = unknown = yes_answers = 0

    for row in gt_rows:
        qid = row["question_id"]
        gt = str(row["label"]).strip().lower()
        pred = str(pred_by_id.get(qid, {}).get("text", "")).strip().lower()

        if gt == "yes":
            if "yes" in pred:
                tp += 1
                yes_answers += 1
            else:
                fn += 1
        elif gt == "no":
            if "no" in pred:
                tn += 1
            else:
                fp += 1
                yes_answers += 1
        else:
            unknown += 1

    total = len(gt_rows)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    return {
        "total": total,
        "true_pos": tp,
        "true_neg": tn,
        "false_pos": fp,
        "false_neg": fn,
        "unknown": unknown,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "yes_rate": (yes_answers / total) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Ground truth POPE jsonl file")
    parser.add_argument("--pred", required=True, help="Model output jsonl file")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON")
    args = parser.parse_args()

    metrics = compute_metrics(args.gt, args.pred)
    if args.pretty:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
