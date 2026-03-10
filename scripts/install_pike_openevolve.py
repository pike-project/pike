"""
Install pike-openevolve into local/deps/pike-openevolve.

Clones the repo if it doesn't already exist, then runs `pip install -e .`.
Re-running is safe and will update the installation.

Usage:
    python scripts/install_pike_openevolve.py
"""

import os
import sys
import subprocess
from pathlib import Path

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))
root_dir = (curr_dir / "..").resolve()
deps_dir = root_dir / "local/deps"
pike_oe_dir = deps_dir / "pike-openevolve"


def main():
    if not os.environ.get("VIRTUAL_ENV"):
        raise RuntimeError("Must be run inside a virtual environment (VIRTUAL_ENV not set).")

    if not pike_oe_dir.exists():
        print("Cloning pike-openevolve...")
        deps_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "git@github.com:pike-project/pike-openevolve.git"],
            check=True,
            cwd=deps_dir,
        )
    else:
        print(f"pike-openevolve already cloned at {pike_oe_dir}, skipping clone.")

    print("Installing pike-openevolve...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        check=True,
        cwd=pike_oe_dir,
    )
    print(f"pike-openevolve installed at: {pike_oe_dir}")


if __name__ == "__main__":
    main()
