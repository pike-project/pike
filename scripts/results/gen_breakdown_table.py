import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import json

# ==============================================================================
# --- Main Script Configuration ---
# ==============================================================================

# Per-(level, file-stem) task IDs whose speedup is forced to 1.0.
# Used for benchmarks manually determined to be invalid.
_BASELINE_SPEEDUP_FORCE_1 = {
    ("5", "metr"): {1},
}


def main(output_dir: Path, level: str, paper: bool = True, kernelbench_dir: Path | None = None):
    task_blacklist_speedup_1_map = {
        "5": set(),
        "3-pike": set(),
    }

    # This blacklist sets speedup to 1, but does not remove task entirely
    task_blacklist_speedup_1 = task_blacklist_speedup_1_map.get(level, set())

    task_blacklist_map = {
        "5": set(),
        "3-pike": {
            36,
            37,
            38,
            39,
            40,
            41,
            42,
        },
    }

    task_blacklist = task_blacklist_map.get(level, set())

    target_dirname = f"h100_level_{level}"
    results_dir = output_dir / target_dirname / "results"
    data_dir = results_dir / "data" / "runtimes"
    figs_dir = results_dir / "figs" / "breakdown"
    tables_dir = results_dir / "data" / "tables" / "geomean_speedups"

    if kernelbench_dir is None:
        kernelbench_dir = Path(__file__).resolve().parent.parent.parent / "KernelBench"
    level_dir = kernelbench_dir / f"level{level}"

    breakdown_all = not paper
    output_label = "breakdown_all" if breakdown_all else "breakdown_paper"

    if level == "3-pike":
        if breakdown_all:
            included_files = [
                "eager",
                "prev_agents",
                "openevolve_agents",

                "prev_noagents",
                "prev_agents_cheap_efa",
                "prev_agents_no_iba",
                "openevolve_noagents",
                "openevolve_agents_mutation",
                # "openevolve_agents_mutation_aggressive",
                "openevolve_agents_no_parallel_eval",
                "openevolve_agents_no_parallel_eval_no_islands",
                "openevolve_agents_mut_nopar_noisl_exploitonly",
                "openevolve_agents_mut_nopar_noisl_exploitonly_shortlib",

                "metr",
                "compile",
                "tensorrt"
            ]
        else:
            included_files = [
                "eager",
                # "prev_agents_median",
                "prev_agents",
                # "openevolve_agents",
                # "openevolve_agents_no_parallel_eval_no_islands",
                "openevolve_agents_mut_nopar_noisl_exploitonly_shortlib",

                "metr",
                "compile",
                "tensorrt"
            ]
    elif level == "5":
        included_files = [
            "eager",
            "prev_agents",
            # "openevolve_agents",

            # "openevolve_agents_no_parallel_eval_no_islands",
            # "openevolve_agents_mut_nopar_noisl_exploitonly",

            "openevolve_agents_mut_nopar_noisl_exploitonly_shortlib",

            "metr",
            "compile",
            "tensorrt"
        ]


    plot_mode = "bar"  # choose "line" or "bar"

    primary_sort_map = {
        ("3-pike", True): "prev_agents",   # paper mode
        ("3-pike", False): "prev_agents",          # all mode
        ("5", True): "prev_agents",
        ("5", False): "prev_agents",
    }
    primary_str_match = primary_sort_map.get((level, paper), "prev_agents")
    # primary_str_match = "ours (oe, agents)"

    # --- LaTeX Table Configuration ---
    task_name_remapping = {
        "DenseNet121TransitionLayer": "DenseNet121TL",
        "GoogleNetInceptionModule": "GoogleNetIM",
        "DenseNet121DenseBlock": "DenseNet121DB",
        "GRUBirectional": "GRUBi",
        "GRUBidirectionalHidden": "GRUBiHidden",
        "MinGPTCausalAttention": "MinGPTCausAtt",
        "Mamba2ReturnFinalState": "Mamba2ReturnF",
        "SqueezeNetFireModule": "SqueezeNet",
        "MiniGPTBlock": "MinGPTBlock",
        "EfficientNetMBConv": "EfficientNetM",
        "ReLUSelfAttention": "ReLUSelfAtt",
        "LTSMBidirectional": "LSTMBi",
        "SwinTransformerV2": "SwinV2",
        "ResNetBasicBlock": "ResNetBB",
        "VisionAttention": "VisionAtt",
        "ShallowWideMLP": "ShallowWideMLP",

        "deepseek_v3_1gpu": "DeepSeek3",
        "rwkv_pytorch": "RWKVTorch",
        "llama2_decode": "Llama2Dec",
        "llama2": "Llama2",
        "stablediffusion3_mmdit": "StableDiff3",
        "deepseekv3_MLA_1gpu": "DeepSeek3MLA",
        "hunyuanvideo_transformer": "HunyuanTrans",
        "deepseekv3_MLA_decode": "DeepSeek3MLAD",
        "deepseekv3_MOE_largebatch": "DeepSeek3MOEl",
        "s4": "S4",
        "hunyuanvideo_vae_encoder": "HunyuanEnc",
        "hunyuanvideo_vae_decoder": "HunyuanDec",
        "deepseekv3_MOE_smallbatch": "DeepSeek3MOEs",
        "mamba2_pytorch": "Mamba2",
    }

    title_remapping = {
        "prev_agents_median": "PB-m",
        "prev_agents": "PB",
        "openevolve_agents": "PO",

        "prev_noagents": "PBne",
        "prev_agents_cheap_efa": "PBce",
        "prev_agents_no_iba": "PBni",
        "openevolve_noagents": "POne",
        "openevolve_agents_mutation": "POm1",
        # "openevolve_agents_mutation_aggressive": "PO-m2",

        "openevolve_agents_no_parallel_eval": "POm2",
        "openevolve_agents_no_parallel_eval_no_islands": "POx",

        "openevolve_agents_mut_nopar_noisl_exploitonly": "POy",
        "openevolve_agents_mut_nopar_noisl_exploitonly_shortlib": "PO'",

        "torch.compile": "TI",
        "METR": "MET",
        "TensorRT": "TRT",
        # add more mappings as needed
    }

    # --- Load all runtimes ---
    all_methods = {}
    file_to_title_map = {}

    for file in data_dir.glob("*.json"):
        if file.stem not in included_files:
            continue

        with open(file) as f:
            data = json.load(f)
        title = data["title"]
        results = {int(entry["problem_id"]): entry["runtime"] for entry in data["results"]}
        all_methods[title] = results

        # Build map dynamically
        file_to_title_map[file.stem] = title

    # Find eager baseline
    eager_title = file_to_title_map.get("eager", None)
    if eager_title is None:
        raise ValueError("Missing baseline 'eager.json' in data_dir")
    eager_key = next((k for k in all_methods if k.lower() == eager_title.lower()), None)
    eager_runtimes = all_methods[eager_key]

    # --- Determine the universe of tasks and apply blacklist ---
    # All tasks that exist in the baseline are the initial set we will evaluate.
    all_tasks_from_eager = sorted(list(eager_runtimes.keys()))
    print(f"Found {len(all_tasks_from_eager)} tasks in the '{eager_title}' baseline.")

    included_tasks = []

    for t in all_tasks_from_eager:
        if t in task_blacklist:
            continue
        included_tasks.append(t)

    # Get the primary key which will be used for sorting the tasks later
    primary_key = file_to_title_map.get(primary_str_match, None)
    if primary_key not in all_methods:
        raise ValueError(f"Missing primary key method '{primary_str_match}' for sorting")

    # --- Compute speedups ---
    # For any task that exists in "eager" but not in another method, its speedup is 1.0.
    # The speedups will be calculated in the order of `included_tasks` (i.e., by task number).
    methods_speedups = {title: [] for title in all_methods if title != eager_key}
    methods_speedups_unclamped = {title: [] for title in all_methods if title != eager_key}
    title_to_file_stem = {v: k for k, v in file_to_title_map.items()}

    for title, runtimes in all_methods.items():
        if title == eager_key:
            continue

        file_stem = title_to_file_stem.get(title, "")
        force_1 = _BASELINE_SPEEDUP_FORCE_1.get((level, file_stem), set())

        print(f"------ Running: {title} --------")
        for task in included_tasks:
            eager_runtime = eager_runtimes[task]
            method_runtime = runtimes.get(task)

            if method_runtime is None:
                speedup = 1.0
                speedup_unclamped = None  # failed run
            else:
                speedup_unclamped = eager_runtime / method_runtime
                # Optional policy: prevent reporting slowdowns, clamping to a minimum of 1.0x.
                if speedup_unclamped < 1.0:
                    speedup = 1.0
                    print(f"Speedup clamp: {task}, speedup: {speedup_unclamped:.4f}")
                else:
                    speedup = speedup_unclamped

            if int(task) in task_blacklist:
                continue

            if task in task_blacklist_speedup_1 or task in force_1:
                speedup = 1.0
                speedup_unclamped = 1.0

            methods_speedups[title].append(speedup)
            methods_speedups_unclamped[title].append(speedup_unclamped)

    print("------------------------")


    # --- Task labels ---
    task_labels_map = {}
    for filename in os.listdir(level_dir):
        if not filename.endswith(".py"):
            continue
        try:
            # Split only on the first underscore to separate task number from name
            parts = filename.split('_', 1)
            task = int(parts[0])

            if task in task_blacklist:
                continue

            # Remove the .py extension from the second part
            label = parts[1].rsplit('.py', 1)[0]
            # if breakdown_all:
            #     task_labels_map[task] = f"{label} ({task})"
            # else:
            #     task_labels_map[task] = label
            task_labels_map[task] = label
        except (IndexError, ValueError):
            # This handles filenames that don't match the "123_TaskName.py" format
            print(f"Warning: Could not parse task info from filename: {filename}")
            continue


    # --- Sort tasks based on the speedup of the primary key approach ---
    # 1. Get the speedups for the primary method, which will be our sorting key.
    primary_speedups = methods_speedups[primary_key]

    # 2. Pair each task ID with its corresponding speedup from the primary method.
    #    `included_tasks` is sorted by task ID, and `primary_speedups` has the same order.
    task_speedup_pairs = list(zip(included_tasks, primary_speedups))

    # 3. Sort these pairs in descending order based on the speedup value.
    sorted_task_speedup_pairs = sorted(
        task_speedup_pairs, key=lambda item: item[1]
    )

    # 4. From the sorted pairs, extract the new order of task IDs.
    sorted_tasks = [task for task, speedup in sorted_task_speedup_pairs]

    # 5. To reorder all other methods' speedups correctly, we need a map from
    #    the task ID to its original index in the unsorted list.
    task_to_original_index = {task: i for i, task in enumerate(included_tasks)}

    # 6. Create a new dictionary to hold the reordered speedup lists for all methods.
    sorted_methods_speedups = {}
    sorted_methods_speedups_unclamped = {}
    for title, original_speedup_list in methods_speedups.items():
        # For each method, create a new list by picking values from the original list
        # in the new `sorted_tasks` order.
        new_speedup_list = [
            original_speedup_list[task_to_original_index[task_id]]
            for task_id in sorted_tasks
        ]
        sorted_methods_speedups[title] = new_speedup_list
        new_speedup_list_unclamped = [
            methods_speedups_unclamped[title][task_to_original_index[task_id]]
            for task_id in sorted_tasks
        ]
        sorted_methods_speedups_unclamped[title] = new_speedup_list_unclamped

    # 7. Create the task labels for the plot's x-axis in the new sorted order.
    labels_sorted = [task_labels_map.get(t, str(t)) for t in sorted_tasks]

    # 8. Overwrite the original speedups dictionary. All downstream code (plotting, tables)
    #    will now use the data sorted by the primary key's performance.
    methods_speedups = sorted_methods_speedups
    methods_speedups_unclamped = sorted_methods_speedups_unclamped


    # --- Compute geometric mean ---
    geomeans = {}
    for name, values in methods_speedups.items():
        # print(name, len(values))
        arr = np.array(values, dtype=float)
        arr = arr[arr > 0]  # filter out invalid/missing
        geomeans[name] = np.exp(np.mean(np.log(arr)))

    # --- Compute unclamped geometric mean (None/failed tasks treated as 1.0) ---
    geomeans_unclamped = {}
    for name, values in methods_speedups_unclamped.items():
        arr = np.array([v if v is not None else 1.0 for v in values], dtype=float)
        arr = arr[arr > 0]
        geomeans_unclamped[name] = np.exp(np.mean(np.log(arr)))

    # --- Compute TI-relative unclamped geomean (speedup vs torch.compile) ---
    compile_title = file_to_title_map.get("compile")
    geomeans_ti_unclamped = {}
    if compile_title and compile_title in all_methods:
        compile_runtimes = all_methods[compile_title]
        for name in methods_speedups.keys():
            method_runtimes_raw = all_methods.get(name, {})
            ratios = []
            for task in sorted_tasks:
                cr = compile_runtimes.get(task)
                mr = method_runtimes_raw.get(task)
                if cr is None or mr is None:
                    continue
                ratios.append(cr / mr)
            if ratios:
                arr = np.array(ratios, dtype=float)
                arr = arr[arr > 0]
                geomeans_ti_unclamped[name] = np.exp(np.mean(np.log(arr)))

    # --- Compute summary stats: success rate, slower counts ---
    n_tasks = len(sorted_tasks)
    success_rates = {}
    slower_counts = {}
    slower_ti_counts = {}

    for name, values in methods_speedups_unclamped.items():
        n_success = sum(1 for v in values if v is not None)
        success_rates[name] = 100.0 * n_success / n_tasks if n_tasks > 0 else 0.0
        count = 0
        for task, v in zip(sorted_tasks, values):
            if v is None:  # method failed for this task
                continue
            if eager_runtimes.get(task) is None:  # eager baseline failed for this task
                continue
            if v < 1.0:
                count += 1
        slower_counts[name] = count

    if compile_title and compile_title in all_methods:
        for name in methods_speedups_unclamped.keys():
            method_runtimes_raw = all_methods.get(name, {})
            count = 0
            for task in sorted_tasks:
                cr = compile_runtimes.get(task)
                mr = method_runtimes_raw.get(task)
                if cr is None or mr is None:
                    continue
                if cr / mr < 1.0:
                    count += 1
            slower_ti_counts[name] = count

    # --- Plotting ---
    x = np.arange(len(labels_sorted))
    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Enforce plotting order using included_files → titles
    plot_order = []
    for f in included_files:
        if f == "eager":  # skip baseline
            continue
        title = file_to_title_map.get(f, None)
        if title and title in methods_speedups:
            plot_order.append(title)

    if plot_mode == "line":
        for name in plot_order:
            values = methods_speedups[name]
            ax.plot(
                x,
                values,
                label=f"{name} (gmean={geomeans[name]:.2f})",
                linewidth=2,
            )
    elif plot_mode == "bar":
        n_methods = len(plot_order)
        width = 0.8 / n_methods  # total width of the group is 0.8

        for i, name in enumerate(plot_order):
            values = methods_speedups[name]
            offsets = x - 0.4 + i * width + width / 2  # center the group around x
            ax.bar(
                offsets,
                values,
                width=width,
                label=f"{name} (gmean={geomeans[name]:.2f})",
                alpha=0.9
            )
    else:
        raise ValueError(f"Unknown plot_mode: {plot_mode}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_sorted, rotation=45, ha="right")
    plt.title(f"Level {level} Speedup Over PyTorch Eager (H100)")
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.ylabel("Speedup")
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.3, top=0.95)
    ax.legend(loc='upper left', fontsize=10)
    plt.yscale("log")

    # --- Save plot ---
    # os.makedirs(figs_dir, exist_ok=True)
    # fig.savefig(figs_dir / f"{output_label}_{plot_mode}.pdf")
    # print(f"Plot saved to: {figs_dir / f'{output_label}_{plot_mode}.pdf'}")
    plt.close()

    # --- Prepare DataFrame for CSV and TeX ---
    # Enforce CSV columns order the same as included_files
    ordered_cols = [file_to_title_map[f] for f in included_files if f != "eager" and f in file_to_title_map]
    df = pd.DataFrame({name: methods_speedups[name] for name in ordered_cols}, index=labels_sorted)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Task"}, inplace=True)

    # --- Create output directories ---
    csv_dir = tables_dir / "csv"
    tex_dir = tables_dir / "tex"
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(tex_dir, exist_ok=True)

    # --- Save CSV ---
    csv_path = csv_dir / f"{output_label}.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")

    # --- Save TeX Table using the new function ---
    tex_path = tex_dir / f"{output_label}.tex"
    # Build unclamped DataFrame (same structure as df but uses unclamped values, with None for failed runs)
    df_unclamped = pd.DataFrame(
        {name: methods_speedups_unclamped[name] for name in ordered_cols},
        index=labels_sorted,
    )
    df_unclamped.reset_index(inplace=True)
    df_unclamped.rename(columns={"index": "Task"}, inplace=True)
    generate_latex_table(df, geomeans, task_name_remapping, title_remapping, tex_path,
                         unclamped_df=df_unclamped, geomeans_unclamped=geomeans_unclamped,
                         geomeans_ti_unclamped=geomeans_ti_unclamped,
                         success_rates=success_rates, slower_counts=slower_counts,
                         slower_ti_counts=slower_ti_counts)

    # --- Print Geomean Summary ---
    print("\nGeomean speedups:")
    for k in ordered_cols:
        print(f"- {k}: {geomeans[k]:.3f}")

def generate_latex_table(df, geomeans, task_remapping, title_remapping, output_path,
                          unclamped_df=None, geomeans_unclamped=None, geomeans_ti_unclamped=None,
                          success_rates=None, slower_counts=None, slower_ti_counts=None):
    """
    Converts a pandas DataFrame of speedups to a formatted LaTeX table.

    Args:
        df (pd.DataFrame): DataFrame with a 'Task' column and method speedup columns (clamped).
        geomeans (dict): Dictionary mapping original method titles to their geomean speedups (clamped).
        task_remapping (dict): Dictionary to remap long task names to shorter ones.
        title_remapping (dict): Dictionary to remap long method titles to shorter ones.
        output_path (Path): Path object where the .tex file will be saved.
        unclamped_df (pd.DataFrame | None): DataFrame with unclamped speedups (None for failed runs).
        geomeans_unclamped (dict | None): Geomeans computed from unclamped values.
    """
    # Use unclamped data for display if provided, otherwise fall back to clamped
    display_df = unclamped_df if unclamped_df is not None else df

    # Create a working copy of the display DataFrame
    df_processed = display_df.copy()

    # --- Step 1: Remap task and column names ---
    df_processed["Task"] = df_processed["Task"].map(lambda x: task_remapping.get(x, x))

    # Create a robust mapping from remapped titles back to original titles
    # This is needed to look up the correct geomean value for each column
    original_titles = df_processed.columns[1:]
    remapped_to_original = {
        title_remapping.get(orig, orig): orig for orig in original_titles
    }

    # Apply the title remapping to the DataFrame columns
    df_processed.rename(columns=title_remapping, inplace=True)

    # --- Step 2: Format numbers and bold the max value in each row ---
    # Convert all speedup values to strings with 2 decimal places; None → em-dash
    df_rounded = df_processed.copy()
    for col in df_processed.columns[1:]:
        df_rounded[col] = df_processed[col].map(
            lambda x: "{—}" if pd.isna(x) else (f"{x:.2f}" if pd.api.types.is_number(x) else str(x))
        )

    def bold_max(row):
        """Helper function to find and bold the maximum value in a DataFrame row."""
        # Convert string numbers to numeric, coercing errors to NaN (dashes become NaN)
        numeric_vals = pd.to_numeric(row[1:], errors="coerce")
        if numeric_vals.notna().any():
            max_val = numeric_vals.max()
            for col in row.index[1:]:
                try:
                    # Compare using a small tolerance for floating point precision
                    if abs(float(row[col]) - max_val) < 1e-9:
                        row[col] = f"\\textbf{{{row[col]}}}"
                except (ValueError, TypeError):
                    continue
        return row

    df_bold = df_rounded.apply(bold_max, axis=1)

    # --- Step 3: Add two Geomean rows (clamped and unclamped) ---
    def make_geo_row(label, geo_dict):
        row_data = {"Task": label}
        for remapped_title in df_bold.columns[1:]:
            original_title = remapped_to_original.get(remapped_title)
            if original_title and original_title in geo_dict:
                val = geo_dict[original_title]
                row_data[remapped_title] = f"{val:.2f}"
            else:
                row_data[remapped_title] = "-"
        return pd.Series(row_data)

    geo_clamped_row = make_geo_row("gmean (clamped)", geomeans)
    geo_unclamped_row = (
        make_geo_row("gmean (uncl.)", geomeans_unclamped)
        if geomeans_unclamped is not None
        else None
    )

    geo_ti_uncl_row = (
        make_geo_row("gmean (TI)", geomeans_ti_unclamped)
        if geomeans_ti_unclamped is not None
        else None
    )

    def make_stat_row(label, stat_dict, fmt):
        row_data = {"Task": label}
        for remapped_title in df_bold.columns[1:]:
            original_title = remapped_to_original.get(remapped_title)
            if original_title and original_title in stat_dict:
                val = stat_dict[original_title]
                row_data[remapped_title] = fmt.format(val=val)
            else:
                row_data[remapped_title] = "-"
        return pd.Series(row_data)

    rows_to_append = [geo_clamped_row]
    if geo_unclamped_row is not None:
        rows_to_append.append(geo_unclamped_row)
    if geo_ti_uncl_row is not None:
        rows_to_append.append(geo_ti_uncl_row)
    if success_rates is not None:
        rows_to_append.append(make_stat_row("success", success_rates, r"{val:.0f}\%"))
    if slower_counts is not None:
        rows_to_append.append(make_stat_row("slower", slower_counts, "{val:.0f}"))
    if slower_ti_counts is not None:
        rows_to_append.append(make_stat_row("slower (TI)", slower_ti_counts, "{val:.0f}"))

    df_with_geo = pd.concat([df_bold, pd.DataFrame(rows_to_append)], ignore_index=True)
    # Bold max in each geomean row (not stat rows)
    n_geo_rows = (1
                  + (1 if geo_unclamped_row is not None else 0)
                  + (1 if geo_ti_uncl_row is not None else 0))
    for idx in range(len(df_bold), len(df_bold) + n_geo_rows):
        df_with_geo.iloc[idx] = bold_max(df_with_geo.iloc[idx])

    # --- Step 4: Convert to LaTeX format and save ---
    # Use 'l' for left-align (Task) and 'r' for right-align (numbers)
    column_format = 'l' + 'r' * (len(df_with_geo.columns) - 1)
    latex_table = df_with_geo.to_latex(
        index=False, escape=False, column_format=column_format
    )

    # Insert a horizontal line rule (`\midrule`) before the first Geomean row for clarity
    lines = latex_table.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("gmean"):
            lines.insert(i, "\\midrule")
            break
    latex_table = "\n".join(lines)

    # Save the final LaTeX table to the specified file
    with open(output_path, "w") as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--level", type=str, required=True, choices=["3-pike", "5"])
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--kernelbench-dir", type=str, default=None)
    args = parser.parse_args()
    kb_dir = Path(args.kernelbench_dir).resolve() if args.kernelbench_dir else None
    main(Path(args.output_dir).resolve(), args.level, paper=args.paper, kernelbench_dir=kb_dir)
