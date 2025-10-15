import os
from pathlib import Path
import argparse

# this script is intended for converting the "prev" output directory structure
# to the openevolve output directory structure

def convert(src_dir, dst_dir):
    print(src_dir)
    print(dst_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    os.makedirs(dst_dir, exist_ok=True)

    convert(src_dir, dst_dir)
