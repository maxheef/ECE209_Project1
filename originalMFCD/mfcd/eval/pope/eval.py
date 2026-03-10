import argparse
import gc
import os
import json
import logging

import torch

from datasets import load_dataset
from eval_utils import *


def arg_parse():
    parser = argparse.ArgumentParser(description="POPE evaluation on LVLMs.")

    arg_configs = {
        "--dataset-path": {
            "type": str,
            "default": "lmms-lab/POPE",
            "help": "The path of the locally saved lmms-lab/POPE."
        },
        "--pope-type": {
            "type": str,
            "default": "random",
            "choices": ["random", "popular", "adversarial"],
            "help": "The type of pope."
        },
        "--log-path": {
            "type": str,
            "default": "./llava-sample.json",
            "help": "The path to save the logs."
        },
    }

    for arg_name, arg_config in arg_configs.items():
        parser.add_argument(arg_name, **arg_config)

    for arg_name, arg_config in BASE_ARGS.items():
        parser.add_argument(arg_name, **arg_config)

    args = parser.parse_args()
    return args


def eval(pred_list, label_list):
    pos, neg = 1, 0
    yes_ratio = pred_list.count(1) / len(pred_list)
    TP = TN = FP = FN = 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1
    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "yes_ratio": yes_ratio,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN
    }


def recorder(responses):
    pred_list = []
    neg_words = ["No", "not", "no", "NO"]
    for response in responses:
        response = response.replace('.', '').replace(',', '')
        words = response.split(' ')
        if any(word in neg_words for word in words) or any(word.endswith("n't") for word in words):
            pred_list.append(0)
        else:
            pred_list.append(1)
    return pred_list


def evaluate_model(args):
    # load evaluate information
    device = torch.device(args.device)
    model_info: ModelInfo = MODEL_INFOS[args.model_type]

    processor, generation_config, eval_sets = prepare_for_generate(
        args=args,
        model_info=model_info,
        device=device,
    )

    # load and prepare dataset
    dataset = load_dataset(
        path=args.dataset_path,
        name="Full",
        num_proc=args.num_workers,
    )[args.pope_type]

    prompts = batch_prepare_data(
        args=args,
        dataset=dataset,
        processor=processor,
        question_column_name="question",
        image_column_name="image",
    )

    # prevent memory leaks
    gc.collect(generation=2)

    # load model
    model = model_info.model_class.from_pretrained(
        args.model_name_or_path,
        **model_info.model_config,
        **model_info.special_model_config_for_dataset.get("pope", {})
    ).to(device=device)

    # generate responses
    responses = generate_responses(
        model=model,
        processor=processor,
        prompts=prompts,
        generation_config=generation_config,
    )

    model = model.to(device="cpu")
    del model

    # eval
    # convert string answer to integer format
    label_list = [
        1 if answer.lower() == "yes" else 0
        for answer in dataset["answer"]
    ]
    # convert string response to integer format
    pred_list = recorder(responses)
    # calculate eval indices
    eval_indices = eval(pred_list, label_list)

    eval_indices["eval-sets"] = eval_sets

    # print and save eval indices
    print(eval_indices)
    log_dir = os.path.dirname(args.log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(file=args.log_path, mode="w") as log_file:
        json.dump(
            obj=eval_indices,
            fp=log_file,
            indent=4,
        )


def main():
    for name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
    args = arg_parse()
    device_str = args.device
    if ":" in device_str and device_str.split(":")[0] == "npu":
        try:
            import torch_npu
        except ImportError:
            logging.log(
                level=logging.ERROR,
                msg="Ascend npu is not supported."
            )
            exit(1)
    evaluate_model(args)


if __name__ == "__main__":
    main()
