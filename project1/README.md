# Project1

This folder contains a items for Task A that **reuses code from `originalProject`**

## Files
- `setup_h100_env.sh`: Creates Python 3.9 conda env and installs compatible deps for Colab H100 GPU.
- `run_table1.sh`: Runs 4 iterations (random/popular and regular/vcd) and re-creates Table 1 from: Mitigating Object Hallucinations in Large Vision-Language Models through
Visual Contrastive Decoding

## Using an H100 GPU thorugh Colab
```bash
%%bash
set -e
cd /content
if [ ! -d VCD_project/.git ]; then
  git clone --depth 1 https://github.com/maxheef/ECE209_Project1.git VCD_project
fi
```

```bash
%%bash
set -e
cd /content/VCD_project
bash project1/setup_h100_env.sh /content/VCD_project
```

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

```bash
%%bash
set -e
cd /content/VCD_project
PYBIN=$(cat /tmp/project1_python_bin.txt)
PYTHON_BIN="$PYBIN" bash project1/run_table1.sh \
  liuhaotian/llava-v1.5-7b \
  /content/datasets/coco/val2014 \
  55
```

Outputs:
- `/content/VCD_project/project1/output`
