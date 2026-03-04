import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def copy_all_files(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")

    for path in src_dir.rglob("*"):
        if path.is_file():
            relative_path = path.relative_to(src_dir)
            target_path = (dst_dir / relative_path).resolve()
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target_path)


def copy_level_baseline_runtimes(input_dir: Path, output_dir: Path, level: str) -> None:
    level_dirname = f"h100_level_{level}"

    src_dir = (input_dir / "baseline-runtimes" / level_dirname).resolve()

    runtimes_dst = (
        output_dir
        / level_dirname
        / "results"
        / "data"
        / "runtimes"
    ).resolve()

    money_dst = (
        output_dir
        / level_dirname
        / "results"
        / "data"
        / "runtimes_money_budget"
    ).resolve()

    copy_all_files(src_dir, runtimes_dst)
    copy_all_files(src_dir, money_dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="Path to input directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to output directory")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    for level in ["3-pike", "5"]:
        copy_level_baseline_runtimes(input_dir, output_dir, level)

        # merged_budget: run twice (with and without cost stopping)
        for use_cost in [False, True]:
            cmd = [sys.executable, "scripts/analyze/merged_budget.py",
                   "--input-dir", str(input_dir),
                   "--output-dir", str(output_dir),
                   "--level", level]
            if use_cost:
                cmd.append("--use-cost-stopping")
            subprocess.run(cmd, check=True)

        # plot_trajectories
        subprocess.run([sys.executable, "scripts/results/plot_trajectories.py",
                        "--output-dir", str(output_dir),
                        "--level", level], check=True)

        # plot_overall_speedup
        subprocess.run([sys.executable, "scripts/results/plot_overall_speedup.py",
                        "--output-dir", str(output_dir),
                        "--level", level], check=True)


if __name__ == "__main__":
    main()
