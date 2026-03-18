import os
import subprocess
from pathlib import Path

ROOT = '/content/VCD_project'
BRANCH = os.getenv('BRANCH', 'main')
REPO_URL = os.getenv('REPO_URL', 'https://github.com/maxheef/ECE209_Project1.git')

def sync():
    """
    Syncs the local repository with the latest code from the specified GitHub repository and branch.
    If the repository already exists, it performs a hard reset to the latest commit to ensure a clean state.
    """
    print(f"--- Syncing Repository ({BRANCH}) ---")
    if os.path.exists(f"{ROOT}/.git"):
        # Fast update
        subprocess.run(f"git -C {ROOT} fetch --depth 1 origin {BRANCH}", shell=True, check=True)
        subprocess.run(f"git -C {ROOT} reset --hard FETCH_HEAD", shell=True, check=True)
    else:
        # Fresh clone
        if os.path.exists(ROOT): 
            import shutil
            shutil.rmtree(ROOT)
        subprocess.run(f"git clone --depth 1 --branch {BRANCH} {REPO_URL} {ROOT}", shell=True, check=True)

    # Verify critical path
    if not os.path.isdir(f"{ROOT}/originalMFCD/mfcd"):
        print(f"Error: MFCD directory missing in {ROOT}")
        subprocess.run(f"find {ROOT} -maxdepth 3 -type d | head -n 10", shell=True)
        raise FileNotFoundError("Repo sync failed to find MFCD core.")
    
    print("Sync complete. Code is up to date.")

if __name__ == "__main__":
    sync()
