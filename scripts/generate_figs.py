import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.analyze.merged_budget import run_level as merged_budget_run_level
from scripts.results.plot_trajectories import main as plot_trajectories
from scripts.results.plot_overall_speedup import main as plot_overall_speedup


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
    parser.add_argument("--level", type=str, default=None, choices=["3-pike", "5"],
                        help="Process only this level.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    levels = [args.level] if args.level else ["3-pike", "5"]

    for level in levels:
        level_dirname = f"h100_level_{level}"
        baseline_src = (input_dir / "baseline-runtimes" / level_dirname).resolve()
        if not baseline_src.exists():
            print(f"Warning: baseline runtimes not found for level {level} at {baseline_src}, skipping level.")
            continue

        copy_level_baseline_runtimes(input_dir, output_dir, level)

        # merged_budget: run twice (with and without cost stopping)
        for use_cost in [False, True]:
            merged_budget_run_level(input_dir, output_dir, level,
                                    use_cost_stopping=use_cost)

        plot_trajectories(output_dir, level)
        plot_overall_speedup(output_dir, level)


if __name__ == "__main__":
    main()
