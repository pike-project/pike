import argparse
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import gmean

_TASK_BLACKLIST = {
    "3-pike": {36, 37, 38, 39, 40, 41, 42},
    "5": set(),
}

_BASELINE_FILES = {
    "metr.json": "metr",
    "compile.json": "torch.compile",
    "tensorrt.json": "tensorrt",
}

# Per-(level, baseline label) task IDs whose speedup is forced to 1.0.
# Used for benchmarks manually determined to be invalid.
_BASELINE_SPEEDUP_FORCE_1 = {
    ("5", "metr"): {1},
}


def compute_baseline_speedups(runtimes_dir: Path, level: str) -> dict:
    eager_path = runtimes_dir / "eager.json"
    if not eager_path.exists():
        return {}

    with open(eager_path) as f:
        eager_data = json.load(f)

    eager_map = {
        v["problem_id"]: v["runtime"]
        for v in eager_data.get("results", [])
        if v.get("runtime") is not None and v["runtime"] > 0
    }

    blacklist = _TASK_BLACKLIST.get(level, set())
    result = {}

    for filename, label in _BASELINE_FILES.items():
        baseline_path = runtimes_dir / filename
        if not baseline_path.exists():
            continue

        with open(baseline_path) as f:
            baseline_data = json.load(f)

        baseline_map = {
            entry["problem_id"]: entry["runtime"]
            for entry in baseline_data.get("results", [])
            if entry.get("problem_id") is not None
            and entry.get("runtime") is not None
            and entry["runtime"] > 0
        }

        force_1 = _BASELINE_SPEEDUP_FORCE_1.get((level, label), set())
        speedup_list = []
        for pid, eager_rt in eager_map.items():
            if pid in blacklist:
                continue
            if pid in force_1:
                speedup_list.append(1.0)
                continue
            baseline_rt = baseline_map.get(pid)
            if baseline_rt is not None:
                speedup = max(1.0, eager_rt / baseline_rt)
            else:
                speedup = 1.0
            speedup_list.append(speedup)

        if speedup_list:
            result[label] = float(gmean(speedup_list))

    return result


def main(output_dir: Path, level: str):
    target_dirname = f"h100_level_{level}"
    results_dir = (output_dir / target_dirname / "results").resolve()
    data_dir = results_dir / "data"
    figs_dir = results_dir / "figs"

    overall_speedups_dir = data_dir / "overall_speedups"

    speedups_filename = "speedups.json"
    speedups_money_budget_filename = "speedups_money_budget.json"

    with open(overall_speedups_dir / speedups_filename) as f:
        speedups = json.load(f)

    with open(overall_speedups_dir / speedups_money_budget_filename) as f:
        speedups_money_budget = json.load(f)

    runtimes_dir = data_dir / "runtimes"
    baseline_speedups = compute_baseline_speedups(runtimes_dir, level)
    speedups.update(baseline_speedups)

    if level == "3-pike":
        f, ax = plt.subplots(1, figsize=(4.7, 3))
    else:
        f, ax = plt.subplots(1, figsize=(3.5, 1.9))

    # EFA = Error Fixing Agent
    # IBA = Initial Brainstorming Agent

    label_mapping = [
        ("prev_agents", "PIKE-B", "#2EBEC7"),
        ("prev_agents_cheap_efa", "PIKE-B (cheap EFA)", "#5BD0D4"),
        ("prev_noagents", "PIKE-B (no EFA)", "#5BD0D4"),
        ("prev_agents_no_iba", "PIKE-B (no IBA)", "#5BD0D4"),
        ("openevolve_agents", "PIKE-O", "#ffa600"),
        ("openevolve_noagents", "PIKE-O (no EFA)", "#ffc559"),
        ("openevolve_agents_mutation", "PIKE-O (mut)", "#ffc559"),
        ("openevolve_agents_no_parallel_eval", "PIKE-O (mut,npar)", "#ffc559"),
        ("openevolve_agents_no_parallel_eval_no_islands", "PIKE-O (mut,npar,1isl)", "#ffc559"),
        ("openevolve_agents_mut_nopar_noisl_exploitonly", "PIKE-O (mut,npar,1isl,EO)", "#ffc559"),
        ("openevolve_agents_mut_nopar_noisl_exploitonly_shortlib", "PIKE-O (mut,npar,1isl,EO,SL)", "#ffc559"),
        ("metr", "METR", "#ff6583"),
        ("torch.compile", "torch.compile", "#c07bdf"),
        ("tensorrt", "TensorRT", "#CBA0DE"),
    ]

    titles = []
    values = []
    checkpoint_values = []
    col = []

    known_labels = {label for label, *_ in label_mapping}
    for (label, title, color) in label_mapping:
        if label in speedups:
            titles.append(title)

            values.append(speedups[label])
            col.append(color)

            if label in speedups_money_budget:
                checkpoint_values.append(speedups_money_budget[label])
            else:
                checkpoint_values.append(None)

    # Also include any runs in speedups.json not covered by label_mapping
    for label, value in speedups.items():
        if label not in known_labels:
            titles.insert(0, label)
            values.insert(0, value)
            col.insert(0, "#BBBBBB")
            checkpoint_values.insert(0, speedups_money_budget.get(label))

    plt.ylabel('Speedup')

    bars = plt.bar(titles, values, color=col, linewidth=1, edgecolor='black')

    # --- Add numbers on top of bars ---
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha='center',
            va='bottom',
            fontsize=8,
            rotation=0,
        )

    # --- Add checkpoint lines ---
    for bar, checkpoint in zip(bars, checkpoint_values):
        if checkpoint is None:
            continue  # skip this bar
        x = bar.get_x()
        width = bar.get_width()

        # Draw solid gray checkpoint line
        ax.hlines(
            y=checkpoint,
            xmin=x + 0.02,
            xmax=x + width,
            color="black",
            linewidth=1.2,
            linestyle=":",
            alpha=0.6,
        )

        # Place checkpoint label just below the line
        ax.text(
            x + width / 2,
            checkpoint - 0.05,  # slightly below the line
            f"{checkpoint:.2f}",
            ha='center',
            va='top',
            fontsize=8,
            color='black',
            alpha=0.6,
        )

    if level == "3-pike":
        plt.xticks(rotation=45, ha='right')
    else:
        plt.xticks(rotation=25, ha='right')
    plt.grid(axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.gca().set_axisbelow(True)

    # --- Layout adjustments ---
    if level == "3-pike":
        plt.subplots_adjust(left=0.13, bottom=0.54, top=0.98, right=0.99)
    else:
        plt.tight_layout(pad=0.1)

    x_div = (len(values) - 3) - 0.5
    y_max = max(values)
    plt.axvline(x=x_div, linestyle="--", color="black", linewidth=0.8)
    plt.xlim(-0.5, len(titles) - 0.5)
    plt.ylim(0, y_max * 1.2)

    speedup_figs_dir = Path.resolve(figs_dir / "overall_speedup")
    os.makedirs(speedup_figs_dir, exist_ok=True)
    plt.savefig(speedup_figs_dir / "overall_speedup.pdf")
    print(f"✅ Plot saved to: {speedup_figs_dir / 'overall_speedup.pdf'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, help="Path to input directory (unused, for interface consistency)")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--level", type=str, required=True, choices=["3-pike", "5"], help="Level to process")
    args = parser.parse_args()

    main(Path(args.output_dir).resolve(), args.level)
