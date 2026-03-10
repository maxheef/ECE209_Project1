import gc
import torch

from typing import List, Any
from tqdm import tqdm
from transformers import BatchFeature, GenerationConfig, Qwen2_5_VLForConditionalGeneration, \
    LlavaForConditionalGeneration, LlavaNextForConditionalGeneration
from torch.utils.data import DataLoader
from joblib import Parallel, delayed

from utils import *
from processor import MFCDProcessor

BASE_ARGS = {
    "--max-new-tokens": {
        "type": int,
        "default": 2048,
        "help": "Max number of new tokens.",
    },
    "--model-name-or-path": {
        "type": str,
        "default": "llava-hf/llava-1.5-7b-hf",
        "help": "The name or path of the model to evaluate."
    },
    "--model-type": {
        "type": str,
        "default": "llava-1.5",
        "choices": ["qwen2.5-vl", "llava-1.5", "llava-next"],
        "help": "The model to evaluate."
    },
    "--num-workers": {
        "type": int,
        "default": 1,
        "help": "The number of parallel workers to prepare dataset."
    },
    "--eval-batch-size": {
        "type": int,
        "default": 1,
        "help": "The batch size for evaluation."
    },
    "--device": {
        "type": str,
        "default": "cuda:0",
        "help": "The devices to evaluate on."
    },
    "--temperature": {
        "type": float,
        "default": 1.0,
        "help": "The temperature in generation."
    },
    "--top-p": {
        "type": float,
        "default": 1.0,
        "help": "The top p in generation."
    },
    "--top-k": {
        "type": int,
        "default": 50,
        "help": "The top k in generation."
    },
    "--mfc-high-alpha": {
        "type": float,
        "default": 0.1,
        "help": "The high alpha in mfc generation."
    },
    "--mfc-low-alpha": {
        "type": float,
        "default": 0.1,
        "help": "The low alpha in mfc generation."
    },
    "--mfc-beta": {
        "type": float,
        "default": 1.0,
        "help": "The beta in mfc generation."
    },
    "--mfc-jsd": {
        "type": lambda x: str(x).lower() == "true",
        "default": False,
        "choices": [True, False],
        "help": "Whether to use the jsd in mfc generation."
    },
    "--mfc-entropy": {
        "type": lambda x: str(x).lower() == "true",
        "default": False,
        "choices": [True, False],
        "help": "Whether to use the entropy in mfc generation just like lcd.)."
    },
    "--mfc-high-pass-cutoff": {
        "type": float,
        "default": 120,
        "help": "The high cutoff in mfc processor."
    },
    "--mfc-low-pass-cutoff": {
        "type": float,
        "default": 30,
        "help": "The low cutoff in mfc processor."
    },
    "--mfc-filter-type": {
        "type": str,
        "default": "ideal",
        "choices": ["ideal", "gaussian"],
        "help": "The type of filter in mfc processor."
    },
}

MODEL_INFOS = {
    "qwen2.5-vl": ModelInfo(
        model_class=Qwen2_5_VLForConditionalGeneration,
        processor_config={
            "padding_side": "left"
        },
    ),
    "llava-1.5": ModelInfo(
        model_class=LlavaForConditionalGeneration,
    ),
    "llava-next": ModelInfo(
        model_class=LlavaNextForConditionalGeneration,
    ),
}


def generate_responses(
        model,
        processor: Any,
        prompts: List[BatchFeature],
        generation_config: GenerationConfig,
) -> List[str]:
    responses = []

    with torch.inference_mode():
        for prompt in tqdm(
                iterable=prompts,
                desc=f"Generating responses",
        ):
            prompt = prompt.to(dtype=model.dtype, device=model.device)
            outputs = model.generate(
                **prompt,
                generation_config=generation_config,
            )
            # avoid memory leaks
            prompt = prompt.to(device="cpu")
            responses.extend(
                (
                    processor.decode(outputs[i][prompt["input_ids"][i].shape[0]:], skip_special_tokens=True).strip()
                    for i in range(outputs.size(0))
                )
            )
            # avoid memory leaks
            del prompt, outputs
            gc.collect(generation=2)
            if model.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    gc.collect(generation=2)

    return responses


def prepare_for_generate(
        args,
        model_info: ModelInfo,
        device: torch.device = torch.device("cpu"),
):

    eval_sets = {}

    processor = MFCDProcessor.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        device=device,
        high_pass_cutoff=args.mfc_high_pass_cutoff,
        low_pass_cutoff=args.mfc_low_pass_cutoff,
        filter_type=args.mfc_filter_type,
        use_fast=True,
        **model_info.processor_config,
    )
    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=True,
        mfc_low_alpha=args.mfc_low_alpha,
        mfc_high_alpha=args.mfc_high_alpha,
        mfc_beta=args.mfc_beta,
        mfc_jsd=args.mfc_jsd,
        mfc_entropy=args.mfc_entropy,
        use_cache=True,
        max_new_tokens=args.max_new_tokens,
        **model_info.generation_config,
    )
    eval_sets["mfc_low_alpha"] = args.mfc_low_alpha
    eval_sets["mfc_high_alpha"] = args.mfc_high_alpha
    eval_sets["mfc_beta"] = args.mfc_beta
    eval_sets["mfc_high_pass_cutoff"] = args.mfc_high_pass_cutoff
    eval_sets["mfc_low_pass_cutoff"] = args.mfc_low_pass_cutoff
    eval_sets["mfc_jsd"] = args.mfc_jsd
    eval_sets["mfc_entropy"] = args.mfc_entropy
    eval_sets["temperature"] = args.temperature
    eval_sets["top_k"] = args.top_k
    eval_sets["top_p"] = args.top_p

    return processor, generation_config, eval_sets


def process_proxy(
        text,
        images,
        processor,
) -> BatchFeature:
    return processor(
        images=images,
        text=text,
        return_tensors="pt",
        padding=True
    )


def batch_prepare_data(
        args,
        dataset,
        processor,
        question_column_name: str = "question",
        image_column_name: str = "image",
) -> List[BatchFeature]:
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )
    text_list = []
    images_list = []
    for batch_data in tqdm(
            iterable=data_loader,
            total=len(data_loader),
            desc=f"Applying chat template",
    ):
        images = []
        text = []
        for data in batch_data:
            text.append(
                processor.apply_chat_template(
                    conversation=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                },
                                {
                                    "type": "text",
                                    "text": data[question_column_name],
                                }
                            ]
                        }
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )
            images.append(data[image_column_name])
        images_list.append(images)
        text_list.append(text)

    prompts = Parallel(n_jobs=args.num_workers)(
        delayed(process_proxy)(
            text=text,
            images=images,
            processor=processor,
        )
        for text, images in tqdm(
            iterable=zip(text_list, images_list),
            total=len(text_list),
            desc=f"Processing batch data",
        )
    )
    return prompts
