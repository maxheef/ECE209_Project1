import os
import sys
import json
import torch
import gc
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

# Constants
ROOT = '/content/VCD_project'
ORIG = f'{ROOT}/originalProject'
MFCD_PATH = f'{ROOT}/originalMFCD/mfcd'
COCO_VAL = '/content/datasets/coco/val2014'
MODEL_PATH = 'llava-hf/llava-1.5-7b-hf'
OUT_DIR = f'{ROOT}/output'

# Add MFCD to path
if MFCD_PATH not in sys.path:
    sys.path.insert(0, MFCD_PATH)

from eval_utils import (prepare_for_generate, generate_responses, 
                        batch_prepare_data, MODEL_INFOS)
from eval.pope.eval import eval as pope_eval, recorder

class POPELocal(Dataset):
    def __init__(self, json_path, image_root):
        with open(json_path, 'r') as f:
            self.rows = [json.loads(line) for line in f]
        self.image_root = Path(image_root)

    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(self.image_root / row['image']).convert('RGB')
        return {'question': row['text'], 'image': img, 'label': row['label'].lower()}

def run_mfcd(pope_type):
    device = torch.device('cuda:0')
    
    class Args: 
        mfc_high_alpha = 1.0; mfc_low_alpha = 1.0; mfc_beta = 0.3
        mfc_high_pass_cutoff = 0.1; mfc_low_pass_cutoff = 0.9
        mfc_filter_type = 'gaussian'; mfc_jsd = False; mfc_entropy = False
        model_name_or_path = MODEL_PATH; model_type = 'llava-1.5'
        num_workers = 4; eval_batch_size = 16; device = 'cuda:0'
        temperature = 1.2; top_p = 1.0; top_k = 50; max_new_tokens = 2048

    args = Args()
    model_info = MODEL_INFOS['llava-1.5']
    processor, gen_config, _ = prepare_for_generate(args=args, model_info=model_info, device=device)

    pope_json = f"{ORIG}/experiments/data/POPE/coco/coco_pope_{pope_type}.json"
    dataset = POPELocal(pope_json, COCO_VAL)
    prompts = batch_prepare_data(args=args, dataset=dataset, processor=processor, 
                                 question_column_name='question', image_column_name='image')

    model = model_info.model_class.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, **model_info.model_config
    ).to(device)

    responses = generate_responses(model=model, processor=processor, prompts=prompts, generation_config=gen_config)

    labels = [1 if d['label'] == 'yes' else 0 for d in dataset]
    results = pope_eval(recorder(responses), labels)
    
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return results

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for p_type in ['random', 'popular']:
        res = run_mfcd(p_type)
        with open(os.path.join(OUT_DIR, f'metrics_mfcd_{p_type}.json'), 'w') as f:
            json.dump(res, f, indent=2)
