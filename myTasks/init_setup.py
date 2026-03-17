import os
import subprocess
import torch
import sys
import shutil
from pathlib import Path

# --- Central Configuration ---
ROOT = '/content/VCD_project'
REPO_URL = 'https://github.com/maxheef/ECE209_Project1.git'
BRANCH = 'main'
CONDA_BASE = '/usr/local/miniconda'

def run_cmd(cmd, cwd=None):
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
        text=True, env=os.environ.copy(), cwd=cwd
    )
    for line in process.stdout:
        print(line, end='')
    process.wait()
    if process.returncode != 0:
        raise Exception(f"Command failed: {cmd}")

def setup_repo():
    """Syncs code only. Fast reset to match remote exactly."""
    print(f"--- Syncing Repository ({BRANCH}) ---")
    if os.path.exists(f"{ROOT}/.git"):
        run_cmd(f"git -C {ROOT} fetch --depth 1 origin {BRANCH}")
        run_cmd(f"git -C {ROOT} reset --hard FETCH_HEAD")
    else:
        if os.path.exists(ROOT): shutil.rmtree(ROOT)
        run_cmd(f"git clone --depth 1 --branch {BRANCH} {REPO_URL} {ROOT}")
    
    if ROOT not in sys.path: sys.path.append(ROOT)

def setup_hardware():
    """Heavy lift: Conda environments and Drivers (~4-6 mins)."""
    # Check if a major environment already exists to skip redundant installs
    vcd_env_path = f"{CONDA_BASE}/envs/vcd310"
    if os.path.exists(vcd_env_path):
        print("Conda environments already exist. Skipping hardware setup.")
        return

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"--- Detected {gpu_name}: Installing Environments ---")

    if "H100" in gpu_name:
        from scripts.setup_h100 import setup_h100_env
        setup_h100_env(root_dir=ROOT)
    else:
        from scripts.setup_g4 import setup_environments

    # Ensure the scripts folder is in the path so we can import the setup files
    scripts_path = os.path.join(ROOT, 'scripts')
    if scripts_path not in sys.path:
        sys.path.append(scripts_path)

    if "H100" in gpu_name:
        from setup_h100 import setup_h100_env # Changed from scripts.setup...
        setup_h100_env(root_dir=ROOT)
    else:
        from setup_g4 import setup_environments # Changed from scripts.setup...
        setup_environments()

def setup_datasets():
    """Heavy lift: COCO Dataset (~4 mins)."""
    DATASET_DIR = "/content/datasets/coco"
    VAL_DIR = os.path.join(DATASET_DIR, "val2014")
    ZIP_FILE = os.path.join(DATASET_DIR, "val2014.zip")
    ZIP_URL = "http://images.cocodataset.org/zips/val2014.zip"

    if os.path.exists(VAL_DIR):
        print("MSCOCO already exists. Skipping download.")
        return

    print("Downloading MSCOCO (~1GB)...")
    os.makedirs(DATASET_DIR, exist_ok=True)
    run_cmd(f"wget -q -c {ZIP_URL} -O {ZIP_FILE}")
    print("Unzipping dataset...")
    run_cmd(f"unzip -q {ZIP_FILE} -d {DATASET_DIR}")
    os.remove(ZIP_FILE)

if __name__ == "__main__":
    # 1. Always sync the repo first (fast)
    setup_repo()
    
    # 2. Setup hardware/envs (heavy - checks if already done)
    setup_hardware()
    
    # 3. Setup datasets (heavy - checks if already done)
    setup_datasets()
    
    # 4. Write paths to a known location for the Notebook to pick up
    vcd_bin = f"{CONDA_BASE}/envs/vcd310/bin/python"
    mfcd_bin = f"{CONDA_BASE}/envs/mfcd310/bin/python"
    
    Path('/tmp/vcd_bin.txt').write_text(vcd_bin)
    Path('/tmp/mfcd_bin.txt').write_text(mfcd_bin)

    print("\nInitialization Complete.")
