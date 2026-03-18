# Maxwell Heefner
# ECE209AS Trustworthy AI
# 18 Feb 26
# Final Project

This folder contains a items for Task A and B that reuses code from `originalProject` and `originalMFCD`

## Files
- `/myTasks`: Contains all files written for the completion of Task A and B for the ECE209AS Final Project
  - `Main.ipynd`: Jupyter Notebook that will run all code and configure the Colab GPU for use and then re-create Table 1 from *Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding* then continue to accomplish improvements to VCD by running a MFCD comparison.
    - Note: All files that the H100 GPU will reference are cloned from GitHub into `/content` storage then writes the outputs to temporary storage on Colab `/content/VCD_project/myTasks/output`
  - `init_setup.py`: Will detect the Colab GPU being used (G4 or H100) and initialize the setup by downloading all requirements listed in `requirements.txt`
  - `analysis.py`: Generates a comparison table for all results of Regular/VCD/MFCD on Regular/Random Splits
  - `sync_repo.py`: Pulls the most recent Git Repository for use on the selected GPU
  - `/scripts`: Contains setup scripts for the Colab G4 or H100 GPU respectively
    - `setup_g4.py`: Will initialize the setup for a G4 GPU
    - `setup_h100.py`: Will initialize the setup for a H100 GPU
  - `/tasks`: Contains scripts to run Task A and Task B requirements
    - `task_a_vcd.py`: Runs 4 iterations (*random/popular* and *regular/vcd*) and re-creates Table 1 data from: *Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding* by *Leng et al.*
      - This file accomplishes this by calling `/eval/object_hallucination_vqa_llava.py` from the original VCD project and implementing all of the desired parameters (*SEED, Alpha, Beta, Noise Step, etc.*).
      - POPE datasets were used from `originalProject` for both random and popular, located here: `experiments/data/POPE/coco/coco_pope_random.json` and `experiments/data/POPE/coco/coco_pope_popular.json`
      - In order to evaluate the results `/eval/eval_pope.py` from `originalProject` was utilized to keep track of True/False Positve/Negative results and calculate *Accuracy*, *Precision*, *Recall* and *F1 Score*
      - VCD source files used by task_a_vcd.py (from originalProject)
      -  /originalProject/experiments/eval/object_hallucination_vqa_llava.py
      -  /originalProject/experiments/eval/eval_pope.py
      -  /originalProject/experiments/data/POPE/coco/coco_pope_random.json
      -  /originalProject/experiments/data/POPE/coco/coco_pope_popular.json
    - `task_b_mfcd.py`: Runs 2 iterations (*random/popular* and *MFCD*) and re-creates Table 3 data from: *Multi-Frequency Contrastive Decoding: Alleviating Hallucinations for Large Vision-Language Models* by *Liu et al.* for comparison with the data found in *Task A*.
      - MFCD source files used by task_b_mfcd.py
        - /originalMFCD/mfcd/eval_utils/__init__.py
        - /originalMFCD/mfcd/eval_utils/eval_utils.py
        - /originalMFCD/mfcd/eval/pope/eval.py
        -  /originalMFCD/mfcd/processor/processor.py
        -  /originalMFCD/mfcd/utils/ (helper functions used by MFCD evaluation pipeline)
        -  /originalMFCD/mfcd/transformers/ (MFCD‑bundled Transformers fork required by evaluation pipeline)

## Using an Differnt GPUs thorugh Colab
### H100 GPU
  - Multiple compatibility errors were noted when using the some of the `originalProject` files, because python 3.9 and numpy 1.x must be used

### G4 GPU
  - Multiple compatibility errors were noted when using the G4 GPU because many of the requirements needed sm_120 and a minimum of Python 3.10


## Outputs:
- `/content/VCD_project/myTasks/output`

## Documentaion statement
The papers located in `/Papers` were referenced to understand Mitigating Object Hallucinationw with VCD, in addition to the `https://github.com/DAMO-NLP-SG/VCD.git` repository which was cloned into `/originalProject`. For Task B, the Git Repository at: `https://github.com/liubq-dev/mfcd.git` was cloned into `/originalMFCD` for usage in re-creating MFCD results
The cloned Git Repository was used to run the experiment, specifically with `/eval/object_hallucination_vqa_llava.py`
In addition to the provided material and Git repository, the VSCode Extension of CODEX was utilized to aid with template generation and creation of `setup_g4.py` and `setup_h100.py` to fix compatibility errors with the Colab G4 & H100 GPU respectively. *Prettier* was used to help format and propse commands to generate clean and efficient code.

## Citation
@article{leng-etal-2023-vcd,
  author = {Leng, Sicong and Zhang, Hang and Chen, Guanzheng and Li, Xin and Lu, Shijian and Miao, Chunyan and Bing, Lidong},
  title = {Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding},
  journal = {arXiv preprint arXiv:2311.16922},
  year = {2023},
  url = {https://arxiv.org/abs/2311.16922}
}

@inproceedings{liu-etal-2025-mfcd,
  author = {Liu, Bingqian and Zhang, Fu and Chen, Guoqing and Cheng, Jingwei},
  title = {Multi-Frequency Contrastive Decoding: Alleviating Hallucinations for Large Vision-Language Models},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year = {2025},
  pages = {28568--28584},
  address = {Suzhou, China},
  publisher = {Association for Computational Linguistics}
}
