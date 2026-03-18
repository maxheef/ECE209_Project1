import os
import subprocess
import shutil
import torch
from pathlib import Path

def run_cmd(cmd, quiet=True):
    """ Utility function to run shell commands with optional quiet mode."""
    if quiet:
        # Run silently in the background
        result = subprocess.run(
            cmd, shell=True, stdout=subprocess.DEVNULL, 
            stderr=subprocess.PIPE, text=True, env=os.environ.copy()
        )
        if result.returncode != 0:
            # If it fails, print the error
            print(f"\nError details: {result.stderr}")
            raise Exception(f"Command failed: {cmd}")
    else:
        # Standard streaming output
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ.copy())
        for line in process.stdout:
            print(line, end='')
        process.wait()
        if process.returncode != 0:
            raise Exception(f"Command failed: {cmd}")

def setup_environments():
    # Configuration
    CONDA_PATH = '/usr/local/miniconda'
    PROJECT_DIR = '/content/VCD_project'
    REQ_FILE = f'{PROJECT_DIR}/requirements.txt'

    VCD_ENV = 'vcd310'
    MFCD_ENV = 'mfcd310'
    VCD_PY = f'{CONDA_PATH}/envs/{VCD_ENV}/bin/python'
    VCD_PIP = f'{CONDA_PATH}/envs/{VCD_ENV}/bin/pip'
    MFCD_PY = f'{CONDA_PATH}/envs/{MFCD_ENV}/bin/python'
    MFCD_PIP = f'{CONDA_PATH}/envs/{MFCD_ENV}/bin/pip'

    # Setup Miniconda
    if not os.path.exists(CONDA_PATH):
        print("Installing Miniconda...")
        run_cmd('wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh')
        run_cmd(f'bash /tmp/miniconda.sh -b -p {CONDA_PATH}')

    conda_bin = f'{CONDA_PATH}/bin/conda'
    os.environ['CONDA_PLUGINS_AUTO_ACCEPT_TOS'] = 'yes'
    run_cmd(f"{conda_bin} tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main")
    run_cmd(f"{conda_bin} tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r")

    # Create envs
    for env_name in [VCD_ENV, MFCD_ENV]:
        env_path = f'{CONDA_PATH}/envs/{env_name}'
        if os.path.exists(env_path):
            shutil.rmtree(env_path)
        print(f'Creating environment: {env_name}...')
        run_cmd(f"{conda_bin} create -n {env_name} python=3.10 -y")
 
    # Install base requirements
    if not os.path.exists(REQ_FILE):
         raise FileNotFoundError(f"Missing {REQ_FILE}")

    print("Installing base requirements (this takes ~3 mins)...")
    for pip_bin in [VCD_PIP, MFCD_PIP]:
        run_cmd(f"{pip_bin} install --upgrade pip setuptools wheel")
        run_cmd(f"{pip_bin} install --no-cache-dir -r {REQ_FILE}")

    # Pin versions
    print("Applying version pins for VCD and MFCD...")
    run_cmd(f"{VCD_PIP} install --no-cache-dir 'transformers==4.31.0' 'tokenizers==0.13.3' 'numpy<2'")
    run_cmd(f"{MFCD_PIP} install --no-cache-dir 'tokenizers==0.21.0' 'numpy<2'")

    # Save binary paths (legacy + current)
    Path('/tmp/vcd_py_path.txt').write_text(VCD_PY)
    Path('/tmp/mfcd_py_path.txt').write_text(MFCD_PY)
    Path('/tmp/vcd_bin.txt').write_text(VCD_PY)
    Path('/tmp/mfcd_bin.txt').write_text(MFCD_PY)
    
    print(f"G4 Environment Setup Complete.")

if __name__ == '__main__':
    setup_environments()
