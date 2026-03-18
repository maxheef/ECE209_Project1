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
- `/scripts`: Contains setup scripts for the Colab G4 or H100 GPU respectively
  - `setup_g4.py`: Will initialize the setup for a G4 GPU
  - `setup_h100.py`: Will initialize the setup for a H100 GPU
- `run_table1.sh`: Runs 4 iterations (*random/popular* and *regular/vcd*) and re-creates Table 1 from: *Mitigating Object Hallucinations in Large Vision-Language Models through
Visual Contrastive Decoding* by *Leng et al.*
    - This file accomplishes this by calling `/eval/object_hallucination_vqa_llava.py` from the original VCD project and implementing all of the desired parameters (*SEED, Alpha, Beta, Noise Step, etc.*).
    - POPE datasets were used from `originalProject` for both random and popular, located here: `experiments/data/POPE/coco/coco_pope_random.json` and `experiments/data/POPE/coco/coco_pope_popular.json`
    - In order to evaluate the results `/eval/eval_pope.py` from `originalProject` was utilized to keep track of True/False Positve/Negative results and calculate *Accuracy*, *Precision*, *Recall* and *F1 Score*

## Using an H100 GPU thorugh Colab
- Multiple compatibility errors were noted when using the some of the `originalProject` files, because python 3.9 and numpy 1.x must be used


## Summary of the commands
### Clone GitHub repository to Colab GPU
```bash
%%bash
set -e
cd /content
if [ ! -d VCD_project/.git ]; then
  git clone --depth 1 https://github.com/maxheef/ECE209_Project1.git VCD_project
fi
```

### Configure the H100 GPU for python 3.9 and numpy 1.x
```bash
%%bash
set -e
cd /content/VCD_project
bash myTasks/setup_h100_env.sh /content/VCD_project
```

### Download the MSCOCO imageset
```bash
%%bash
set -e
mkdir -p /content/datasets/coco
cd /content/datasets/coco
if [ ! -d val2014 ]; then
  wget -q http://images.cocodataset.org/zips/val2014.zip
  unzip -q val2014.zip
fi
```

### Create Table 1 variables and output in neat format
```bash
%%bash
set -e
cd /content/VCD_project
PYBIN=$(cat /tmp/myTasks_python_bin.txt)
PYTHON_BIN="$PYBIN" bash myTasks/run_table1.sh \
  liuhaotian/llava-v1.5-7b \
  /content/datasets/coco/val2014 \
  55
```

Outputs:
- `/content/VCD_project/myTasks/output`

## Documentaion statement
The papers located in `/Papers` were referenced to understand Mitigating Object Hallucinationw with VCD, in addition to the `https://github.com/DAMO-NLP-SG/VCD.git` repository which was cloned into `/originalProject` 
The cloned Git Repository was used to run the experiment, specifically with `/eval/object_hallucination_vqa_llava.py`
In addition to the provided material and Git repository, the VSCode Extension of CODEX was utilized to aid with template generation and creation of `setup_h100_env.sh` to fix compatibility errors with the Colab H100 GPU. *Prettier* was used to help format and propse commands to generate clean and efficient code.

## Citation
@article{damonlpsg2023vcd,
  author = {Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, Lidong Bing},
  title = {Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding},
  year = 2023,
  journal = {arXiv preprint arXiv:2311.16922},
  url = {https://arxiv.org/abs/2311.16922}
}