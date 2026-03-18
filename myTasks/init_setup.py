import os
import subprocess
import torch
import sys
import shutil
from pathlib import Path

# Configuration
ROOT = '/content/VCD_project'
REPO_URL = 'https://github.com/maxheef/ECE209_Project1.git'
BRANCH = 'main'
CONDA_BASE = '/usr/local/miniconda'

def run_cmd(cmd, cwd=None, quiet=True):
    """
    Executes shell commands. 
    If quiet=True, output is suppressed unless an error occurs.
    """
    stdout_dest = subprocess.DEVNULL if quiet else subprocess.PIPE
    
    # Use capture_output=True if quiet to store logs in case of failure
    if quiet:
        result = subprocess.run(cmd, shell=True, stdout=stdout_dest, stderr=stdout_dest, env=os.environ.copy(), cwd=cwd)
    else:
        # Stream output in real-time if not quiet
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ.copy(), cwd=cwd)
        for line in process.stdout:
            print(line, end='')
        process.wait()
        result = process

    if result.returncode != 0:
        raise Exception(f"Command failed: {cmd}")

def setup_repo():
    """
    Pulls the latest code from the specified GitHub repository and branch.
    If the repository already exists, it performs a hard reset to the latest commit.
    """
    print(f"--- Syncing Repository ({BRANCH}) ---")
    if os.path.exists(f"{ROOT}/.git"):
        # Silence git fetch/reset
        run_cmd(f"git -C {ROOT} fetch --depth 1 origin {BRANCH}", quiet=True)
        run_cmd(f"git -C {ROOT} reset --hard FETCH_HEAD", quiet=True)
    else:
        if os.path.exists(ROOT): shutil.rmtree(ROOT)
        run_cmd(f"git clone --depth 1 --branch {BRANCH} {REPO_URL} {ROOT}", quiet=True)
    
    if ROOT not in sys.path: sys.path.append(ROOT)
    print("Repository synced successfully.")

def setup_hardware():
    """ 
    Detect GPU and set up environments accordingly. 
    With Python 3.10 and pinned versions for compatibility. 
    """
    vcd_env_path = f"{CONDA_BASE}/envs/vcd310"
    if os.path.exists(vcd_env_path):
        print("Conda environments already exist. Skipping hardware setup.")
        return

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"--- Detected {gpu_name}: Installing Environments (this takes ~5 min) ---")

    if "H100" in gpu_name:
        from scripts.setup_h100 import setup_h100_env
        setup_h100_env(root_dir=ROOT)
    else:
        from scripts.setup_g4 import setup_environments
        setup_environments()
    print("Hardware environments installed.")

def setup_datasets():
    """
    Download the MSCOCO validation set if not already present.
    This is required for both VCD and MFCD tasks.
    """
    DATASET_DIR = "/content/datasets/coco"
    VAL_DIR = os.path.join(DATASET_DIR, "val2014")
    ZIP_FILE = os.path.join(DATASET_DIR, "val2014.zip")
    ZIP_URL = "http://images.cocodataset.org/zips/val2014.zip"

    if os.path.exists(VAL_DIR):
        print("MSCOCO already exists. Skipping download.")
        return

    print("Downloading MSCOCO (~1GB)... (approx. 2 min)")
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    run_cmd(f"wget -q -c {ZIP_URL} -O {ZIP_FILE}", quiet=True)
    
    print("Unzipping dataset... (approx. 2 min)")
    run_cmd(f"unzip -q {ZIP_FILE} -d {DATASET_DIR}", quiet=True)
    os.remove(ZIP_FILE)
    print("Dataset extracted and ready.")

if __name__ == "__main__":
    setup_repo()
    setup_hardware()
    setup_datasets()
    print("\n[SUCCESS] All components are initialized and quiet.")
