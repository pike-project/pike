import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D


def geometric_mean(series):
    """Compute geometric mean, ignoring zeros or NaNs."""
    valid = series[series > 0].dropna()
    if len(valid) == 0:
        return np.nan
    return np.exp(np.log(valid).mean())

def main(output_dir: Path, level: str, paper: bool = False):
    target_dirname = f"h100_level_{level}"

    output_filename = "all_trajectories_side_by_side"
    output_path = (output_dir / target_dirname / "results/figs/convergence" / f"{output_filename}.pdf").resolve()

    # Create a figure with two subplots, side-by-side
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(5, 3), sharey=True)

    axes = [ax_left, ax_right]
    money_budget_options = [False, True]

    if level == "3-pike":
        plot_map = [
            ("prev_agents", "PIKE-B", "#2E94C7", "-"),
            ("prev_agents_cheap_efa", "PIKE-B (cheap EFA)", "#2E94C7", "--"),
            ("prev_noagents", "PIKE-B (no EFA)", "#2E94C7", ":"),
            ("openevolve_agents", "PIKE-O", "#ffa600", "-"),
            ("openevolve_noagents", "PIKE-O (no EFA)", "#ffa600", ":"),
            ("openevolve_agents_mut_nopar_noisl_exploitonly_shortlib", "PIKE-O (mut,npar,1isl,EO,SL)", "#ffa600", "--"),
        ]
    else:
        plot_map = [
            ("openevolve_agents", "PIKE-O", "#ffa600", "-"),
            ("openevolve_agents_no_parallel_eval_no_islands", "PIKE-O (mut,npar,1isl)", "#ffa600", "--"),
            ("openevolve_agents_mut_nopar_noisl_exploitonly_shortlib", "PIKE-O (mut,npar,1isl,EO,SL)", "#ffa600", ":"),
            ("prev_agents", "PIKE-B", "#2E94C7", "-"),
        ]

    extra_handles = []
    extras = []

    # Loop to generate each plot
    for ax, money_budget in zip(axes, money_budget_options):
        # Configure settings based on whether we're using a money budget
        if money_budget:
            table_dirname = "tables_money_budget"
            xlabel = "Cost per Task ($)"
            idx_column_name = "cost"
        else:
            table_dirname = "tables"
            xlabel = "LLM Queries per Task"
            idx_column_name = "attempt"

        data_dir = (output_dir / target_dirname / "results/data" / table_dirname / "speedup_trajectories").resolve()

        if paper:
            if level == "3-pike":
                extra_color = "#902ebd"
                ax.axhline(y=1.64, color=extra_color, linestyle='--', linewidth=1.5)
                extras = [
                    "torch.compile",
                ]
                extra_handles = [Line2D([0], [0], color=extra_color, label=t, linestyle='--', linewidth=1.5) for t in extras]
            else:
                extra_color = "#ff6583"
                ax.axhline(y=1.50, color=extra_color, linestyle='--', linewidth=1.5)
                extras = [
                    "METR",
                ]
                extra_handles = [Line2D([0], [0], color=extra_color, label=t, linestyle='--', linewidth=1.5) for t in extras]

        # Plot lines in the order defined by plot_map
        known_keys = {file_key for file_key, *_ in plot_map}
        for file_key, label, col, linestyle in plot_map:
            csv_path = data_dir / f"{file_key}.csv"
            if not csv_path.exists():
                # print(f"Missing CSV: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            if idx_column_name not in df.columns:
                print(f"Skipping {csv_path}: missing '{idx_column_name}' column.")
                continue

            # Compute geometric mean across all task columns per attempt
            task_cols = [c for c in df.columns if c != idx_column_name]
            df['geomean_speedup'] = df[task_cols].apply(geometric_mean, axis=1)

            ax.plot(df[idx_column_name], df['geomean_speedup'], label=label, linewidth=2, color=col, linestyle=linestyle)

        # Also plot any extra CSVs present that are not in plot_map (non-paper mode only)
        if not paper and data_dir.exists():
            for csv_path in sorted(data_dir.glob("*.csv")):
                file_key = csv_path.stem
                if file_key in known_keys:
                    continue
                df = pd.read_csv(csv_path)
                if idx_column_name not in df.columns:
                    continue
                task_cols = [c for c in df.columns if c != idx_column_name]
                df['geomean_speedup'] = df[task_cols].apply(geometric_mean, axis=1)
                ax.plot(df[idx_column_name], df['geomean_speedup'], label=file_key, linewidth=2, linestyle="-")

        # --- Figure appearance for this subplot ---
        if ax == ax_left:
            ax.set_xlabel(f"(a) {xlabel}", fontsize=12)
            ax.xaxis.label.set_position((0.4, -0.1))
            ax.set_ylabel("Geomean Speedup", fontsize=12)
        else:
            ax.set_xlabel(f"(b) {xlabel}", fontsize=12)

        if money_budget:
            if level == "5":
                ax.set_xticks(np.arange(0, 51, 10))
            else:
                ax.set_xticks(np.arange(0, 26, 5))

        ax.grid(True, color='gray', linestyle='-', linewidth=1, alpha=0.15)

    # --- Shared Legend ---
    handles, labels = ax_left.get_legend_handles_labels()

    if paper:
        if level == "3-pike":
            handles.insert(3, extra_handles[0])
            labels.insert(3, extras[0])
        else:
            handles += extra_handles
            labels += extras

    legend_pos = (0.5, 1.13)
    if level == "3-pike":
        legend_pos = (0.5, 1.17)

    fig.legend(handles=handles, labels=labels, fontsize=10, loc='upper center', bbox_to_anchor=legend_pos, ncol=2, frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.88])

    # --- Save to PDF ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches='tight')
    plt.close()

    print(f"✅ Plot saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, help="Path to input directory (unused, for interface consistency)")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--level", type=str, required=True, choices=["3-pike", "5"], help="Level to process")
    parser.add_argument("--paper", action="store_true", help="Paper mode: only plot the specified labels and draw special reference lines")
    args = parser.parse_args()

    main(Path(args.output_dir).resolve(), args.level, paper=args.paper)
