import os
import json
from pathlib import Path
import matplotlib.pyplot as plt

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

# target_dirname = "h100_level3-metr"
target_dirname = "h100_level5"

results_dir = (curr_dir / f"{target_dirname}/results").resolve()
data_dir = results_dir / "data"
figs_dir = results_dir / "figs"

overall_speedups_dir = data_dir / "overall_speedups"

speedups_filename = "speedups.json"
speedups_money_budget_filename = "speedups_money_budget.json"

with open(overall_speedups_dir / speedups_filename) as f:
    speedups = json.load(f)

with open(overall_speedups_dir / speedups_money_budget_filename) as f:
    speedups_money_budget = json.load(f)

if target_dirname == "h100_level3-metr":
    speedups["metr"] = 1.40
    speedups["torch.compile"] = 1.64
    speedups["tensorrt"] = 1.41
else:
    speedups["metr"] = 1.50
    speedups["torch.compile"] = 1.29
    speedups["tensorrt"] = 1.25

def main():
    if target_dirname == "h100_level3-metr":
        f, ax = plt.subplots(1, figsize=(4.7, 3))
    else:
        f, ax = plt.subplots(1, figsize=(3.5, 1.9))
    # f, ax = plt.subplots(1, figsize=(6, 4.5))

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

    for (label, title, color) in label_mapping:
        if label in speedups:
            titles.append(title)

            values.append(speedups[label])
            col.append(color)

            if label in speedups_money_budget:
                checkpoint_values.append(speedups_money_budget[label])
            else:
                checkpoint_values.append(None)

    # plt.title("3-metr Speedups (Eager)")
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

    if target_dirname == "h100_level3-metr":
        plt.xticks(rotation=45, ha='right')
    else:
        plt.xticks(rotation=25, ha='right')
    plt.grid(axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.gca().set_axisbelow(True)

    # --- Layout adjustments ---
    if target_dirname == "h100_level3-metr":
        plt.subplots_adjust(left=0.13, bottom=0.54, top=0.98, right=0.99)
    else:
        # plt.subplots_adjust(left=0.14, bottom=0.52, top=0.98, right=0.99)

        plt.tight_layout(pad=0.1)

    x_div = (len(values) - 3) - 0.5
    y_max = max(values)
    plt.axvline(x=x_div, linestyle="--", color="black", linewidth=0.8)
    plt.xlim(-0.5, len(titles) - 0.5)
    plt.ylim(0, y_max * 1.2)

    speedup_figs_dir = Path.resolve(figs_dir / "overall_speedup")
    os.makedirs(speedup_figs_dir, exist_ok=True)
    plt.savefig(speedup_figs_dir / "overall_speedup.pdf")

if __name__ == "__main__":
    main()
