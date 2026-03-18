import os
import subprocess
import shutil
import torch
from pathlib import Path

def run_cmd(cmd, cwd=None):
    """Executes shell commands and streams output."""
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
        text=True, env=os.environ.copy(), cwd=cwd
    )
    for line in process.stdout:
        print(line, end='')
    process.wait()
    if process.returncode != 0:
        raise Exception(f"Command failed: {cmd}")

def setup_h100_env(root_dir="/content/VCD_project", env_name="vcd39"):
    orig_dir = os.path.join(root_dir, "originalProject")
    req_file = os.path.join(root_dir, "requirements.txt")
    conda_path = "/usr/local/miniconda"
    conda_bin = f"{conda_path}/bin/conda"
    env_path = f"{conda_path}/envs/{env_name}"
    python_bin = f"{env_path}/bin/python"
    pip_bin = f"{env_path}/bin/pip"

    # 1) Validation
    if not os.path.isdir(orig_dir):
        raise FileNotFoundError(f"Missing originalProject directory: {orig_dir}")

    # 2) Install Miniconda if missing
    if not os.path.exists(conda_bin):
        print("Installing Miniconda...")
        run_cmd('wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh')
        run_cmd(f'bash /tmp/miniconda.sh -b -p {conda_path}')

    # 3) Initialize Conda settings
    run_cmd(f"{conda_bin} config --set always_yes yes --set changeps1 no")
    # Accept TOS if needed
    for channel in ["main", "r"]:
        run_cmd(f"{conda_bin} tos accept --override-channels --channel https://repo.anaconda.com/pkgs/{channel} || true")

    # 4) Create Environment
    if not os.path.exists(env_path):
        print(f"Creating environment {env_name} (Python 3.9)...")
        run_cmd(f"{conda_bin} create -n {env_name} python=3.9 -y")

    # 5) Install Requirements & Hardware-Specific Pins
    print("Installing H100 optimized packages...")
    run_cmd(f"{pip_bin} install --upgrade pip")
    if not os.path.exists(req_file):
        raise FileNotFoundError(f"Missing requirements.txt: {req_file}")
    run_cmd(f"{pip_bin} install -r {req_file}")
    
    # H100-specific: Newer Torch (cu121) for performance and compatibility
    run_cmd(f"{pip_bin} install --upgrade --index-url https://download.pytorch.org torch==2.2.2 torchvision==0.17.2")
    
    # ABI compatibility pin
    run_cmd(f"{pip_bin} install --force-reinstall 'numpy<2'")

    # 6) Record Python path for Orchestrator
    Path('/tmp/vcd_bin.txt').write_text(python_bin)
    
    # 7) Verification
    print("\n--- H100 Verification ---")
    verify_cmd = f"{python_bin} -c 'import torch, numpy; print(\"Torch:\", torch.__version__); print(\"CUDA Available:\", torch.cuda.is_available()); print(\"GPU:\", torch.cuda.get_device_name(0))'"
    run_cmd(verify_cmd)

if __name__ == "__main__":
    setup_h100_env()
