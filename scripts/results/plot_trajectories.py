import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

# This flag is now controlled within the main function to generate two plots
# MONEY_BUDGET = True

# target_dirname = "h100_level3-metr"
target_dirname = "h100_level5"

# These are now set inside the plotting loop
# table_dirname = "tables"
# output_filename = "all_trajectories"
# xlabel = "LLM Queries per Task"
#
# if MONEY_BUDGET:
#     table_dirname = "tables_money_budget"
#     output_filename = "all_trajectories_money_budget"
#     xlabel = "Cost per Task ($)"

def geometric_mean(series):
    """Compute geometric mean, ignoring zeros or NaNs."""
    valid = series[series > 0].dropna()
    if len(valid) == 0:
        return np.nan
    return np.exp(np.log(valid).mean())

def main():
    curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))
    # The output filename is now static for the combined plot
    output_filename = "all_trajectories_side_by_side"
    output_path = (curr_dir / f"{target_dirname}/results/figs/convergence/{output_filename}.pdf").resolve()

    # Create a figure with two subplots, side-by-side
    if target_dirname == "h100_level5":
        # fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(4.7, 2.6), sharey=True)
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(5, 3), sharey=True)
    else:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(5, 3), sharey=True)

    axes = [ax_left, ax_right]
    money_budget_options = [False, True]

    # "#2E94C7"
    # "#5BA4D4"

    if target_dirname == "h100_level3-metr":
        plot_map = [
            ("prev_agents", "PIKE-B", "#2E94C7", "-"),
            ("prev_agents_cheap_efa", "PIKE-B (cheap EFA)", "#2E94C7", "--"),
            ("prev_noagents", "PIKE-B (no EFA)", "#2E94C7", ":"),
            ("openevolve_agents", "PIKE-O", "#ffa600", "-"),
            ("openevolve_noagents", "PIKE-O (no EFA)", "#ffa600", ":"),
            ("openevolve_agents_mut_nopar_noisl_exploitonly_shortlib", "PIKE-O (mut,npar,1isl,EO,SL)", "#ffa600", "--"),
            # ("openevolve_agents_no_parallel_eval_no_islands", "PIKE-O\n(mut,npar,1isl)", "#ffa600", "--"),
            # ("prev_agents_no_iba", "PIKE-B (no IBA)", "blue", "-."),
            # ("openevolve_agents_mutation", "PIKE-O (mut.)", "orange", ":"),
            # Add more entries in the desired order:
            # ("another_file", "Nice Title"),
        ]
    else:
        plot_map = [
            ("openevolve_agents", "PIKE-O", "#ffa600", "-"),
            ("openevolve_agents_no_parallel_eval_no_islands", "PIKE-O (mut,npar,1isl)", "#ffa600", "--"),
            # ("openevolve_agents_mut_nopar_noisl_exploitonly", "PIKE-O (mut, nopar, noisl,\nexploitonly)", "#ffa600", "-."),
            ("openevolve_agents_mut_nopar_noisl_exploitonly_shortlib", "PIKE-O (mut,npar,1isl,EO,SL)", "#ffa600", ":"),
            ("prev_agents", "PIKE-B", "#2E94C7", "-"),
        ]

    # Loop to generate each plot
    for ax, money_budget in zip(axes, money_budget_options):
        # Configure settings based on whether we're using a money budget
        if money_budget:
            table_dirname = "tables_money_budget"
            xlabel = "Cost per Task ($)"
            idx_column_name = "cost"
            # ax.set_title("With Money Budget", fontsize=12)
        else:
            table_dirname = "tables"
            xlabel = "LLM Queries per Task"
            idx_column_name = "attempt"
            # ax.set_title("Without Money Budget", fontsize=12)

        data_dir = (curr_dir / f"{target_dirname}/results/data/{table_dirname}/speedup_trajectories").resolve()

        if target_dirname == "h100_level3-metr":
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

        # Plot lines in the order defined by title_map
        for file_key, label, col, linestyle in plot_map:
            csv_path = data_dir / f"{file_key}.csv"
            if not csv_path.exists():
                print(f"⚠️ Missing CSV: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            if idx_column_name not in df.columns:
                print(f"Skipping {csv_path}: missing '{idx_column_name}' column.")
                continue

            # Compute geometric mean across all task columns per attempt
            task_cols = [c for c in df.columns if c != idx_column_name]
            df['geomean_speedup'] = df[task_cols].apply(geometric_mean, axis=1)

            ax.plot(df[idx_column_name], df['geomean_speedup'], label=label, linewidth=2, color=col, linestyle=linestyle)

        # --- Figure appearance for this subplot ---
        # Add (a) and (b) labels under the respective figures
        if ax == ax_left:
            ax.set_xlabel(f"(a) {xlabel}", fontsize=12)
            ax.xaxis.label.set_position((0.4, -0.1))
            ax.set_ylabel("Geomean Speedup", fontsize=12)
        else:
            ax.set_xlabel(f"(b) {xlabel}", fontsize=12)


        if money_budget:
            if target_dirname == "h100_level5":
                ax.set_xticks(np.arange(0, 51, 10))
            else:
                ax.set_xticks(np.arange(0, 26, 5))

        ax.grid(True, color='gray', linestyle='-', linewidth=1, alpha=0.15)

    # --- Shared Legend ---
    # Combine plot lines and TODOs from the left plot (handles/labels are the same for both)
    handles, labels = ax_left.get_legend_handles_labels()
    # The 'extras' and 'extra_handles' are overwritten in the loop, but they are the same for both plots
    # if target_dirname == "h100_level3-metr":
    #     handles += extra_handles
    #     labels += extras
    # else:
    #     handles.insert(2, extra_handles[0])
    #     labels.insert(2, extras[0])

    # handles += extra_handles
    # labels += extras
    if target_dirname == "h100_level3-metr":
        handles.insert(3, extra_handles[0])
        labels.insert(3, extras[0])
    else:
        handles += extra_handles
        labels += extras

    # Place the legend above the two subplots
    # num_legend_items = len(handles)
    
    # legend_pos = (0.5, 1.05)
    legend_pos = (0.5, 1.13)
    if target_dirname == "h100_level3-metr":
        legend_pos = (0.5, 1.17)
    
    fig.legend(handles=handles, labels=labels, fontsize=10, loc='upper center', bbox_to_anchor=legend_pos, ncol=2, frameon=True)


    # plt.title("Speedup by Task Cost (Level 3-metr, H100)", fontsize=14)
    # plt.legend(handles=handles, labels=labels, fontsize=10, loc='upper left')
    # plt.legend(handles=handles, labels=labels, fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

    # Adjust layout to make space for the legend
    fig.tight_layout(rect=[0, 0, 1, 0.88]) # rect=[left, bottom, right, top]

    # plt.subplots_adjust(bottom=0.15)

    # --- Save to PDF ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches='tight')
    plt.close()

    print(f"✅ Plot saved to: {output_path}")

if __name__ == "__main__":
    main()
