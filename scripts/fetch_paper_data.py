"""Download and extract paper data from HuggingFace into data/paper-data/."""

import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "paper-data"
DATASET_REPO = "knagaitsev/pike-data-compressed"
FILENAME = "pike-data.tar.gz"
URL = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{FILENAME}"


def download_file(url: str, dest: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
        import shutil

        cached = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=FILENAME,
            repo_type="dataset",
        )
        shutil.copy(cached, dest)
    except ImportError:
        _download_fallback(url, dest)


def _download_fallback(url: str, dest: Path) -> None:
    try:
        import requests

        print(f"Downloading {url} ...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    except ImportError:
        import urllib.request

        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)


def main() -> None:
    if OUTPUT_DIR.exists():
        print(f"Error: {OUTPUT_DIR} already exists. Remove it before re-running.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True)

    tar_path = OUTPUT_DIR / FILENAME
    download_file(URL, tar_path)

    print(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(OUTPUT_DIR, filter="data")

    tar_path.unlink()

    extracted = OUTPUT_DIR / "pike-data"
    print(extracted.resolve())


if __name__ == "__main__":
    main()
