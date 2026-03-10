import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import json

# ==============================================================================
# --- Main Script Configuration ---
# ==============================================================================
curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

def gen_breakdown_table(target_dirname, level):
    # target_dirname = "h100_level5"
    # level = "5"

    # target_dirname = "h100_level_3-pike"
    # level = "3-pike"

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

    data_dir = (curr_dir / f"{target_dirname}/results/data/runtimes").resolve()
    figs_dir = (curr_dir / f"{target_dirname}/results/figs/breakdown").resolve()
    tables_dir = curr_dir / f"{target_dirname}/results/data/tables/geomean_speedups"
    level_dir = (curr_dir / f"../../KernelBench/level{level}").resolve()

    breakdown_all = False

    if level == "3-pike":
        if breakdown_all:
            output_label = "breakdown_all"

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
            output_label = "breakdown_paper"

            included_files = [
                "eager",
                "prev_agents_median",
                "prev_agents",
                # "openevolve_agents",
                # "openevolve_agents_no_parallel_eval_no_islands",
                # "openevolve_agents_mut_nopar_noisl_exploitonly_shortlib",

                "metr",
                "compile",
                "tensorrt"
            ]
    elif level == "5":
        breakdown_all = False

        output_label = "breakdown_paper"

        included_files = [
            "eager",
            "prev_agents",
            "openevolve_agents",

            # "openevolve_agents_no_parallel_eval_no_islands",
            # "openevolve_agents_mut_nopar_noisl_exploitonly",

            # "openevolve_agents_mut_nopar_noisl_exploitonly_shortlib",

            "metr",
            "compile",
            "tensorrt"
        ]


    plot_mode = "bar"  # choose "line" or "bar"

    primary_str_match = "prev_agents_median"
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

    for title, runtimes in all_methods.items():
        if title == eager_key:
            continue
        
        print(f"------ Running: {title} --------")
        for task in included_tasks:
            eager_runtime = eager_runtimes[task]
            method_runtime = runtimes.get(task)

            # Core logic: If a task exists in eager but is missing for a given method,
            # we consider it as having no speedup (speedup = 1.0).
            if method_runtime is None:
                # speedup = 1.0
                speedup = 0.0000000001
                print(f"No runtime for method: {task}")
            else:
                speedup = eager_runtime / method_runtime
                # Optional policy: prevent reporting slowdowns, clamping to a minimum of 1.0x.
                if speedup < 1.0:
                    speedup = 1.0
                    print(f"Speedup clamp: {task}, speedup: {speedup}")

            if int(task) in task_blacklist:
                continue

            # print(task, title)
            # if task == 1 and title == "METR":
            #     speedup = 1.0

            if task in task_blacklist_speedup_1:
                speedup = 1.0

            methods_speedups[title].append(speedup)

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
    for title, original_speedup_list in methods_speedups.items():
        # For each method, create a new list by picking values from the original list
        # in the new `sorted_tasks` order.
        new_speedup_list = [
            original_speedup_list[task_to_original_index[task_id]]
            for task_id in sorted_tasks
        ]
        sorted_methods_speedups[title] = new_speedup_list

    # 7. Create the task labels for the plot's x-axis in the new sorted order.
    labels_sorted = [task_labels_map.get(t, str(t)) for t in sorted_tasks]

    # 8. Overwrite the original speedups dictionary. All downstream code (plotting, tables)
    #    will now use the data sorted by the primary key's performance.
    methods_speedups = sorted_methods_speedups


    # --- Compute geometric mean ---
    geomeans = {}
    for name, values in methods_speedups.items():
        # print(name, len(values))
        arr = np.array(values, dtype=float)
        arr = arr[arr > 0]  # filter out invalid/missing
        geomeans[name] = np.exp(np.mean(np.log(arr)))

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
    generate_latex_table(df, geomeans, task_name_remapping, title_remapping, tex_path)

    # --- Print Geomean Summary ---
    print("\nGeomean speedups:")
    for k in ordered_cols:
        print(f"- {k}: {geomeans[k]:.3f}")

def generate_latex_table(df, geomeans, task_remapping, title_remapping, output_path):
    """
    Converts a pandas DataFrame of speedups to a formatted LaTeX table.

    Args:
        df (pd.DataFrame): DataFrame with a 'Task' column and method speedup columns.
        geomeans (dict): Dictionary mapping original method titles to their geomean speedups.
        task_remapping (dict): Dictionary to remap long task names to shorter ones.
        title_remapping (dict): Dictionary to remap long method titles to shorter ones.
        output_path (Path): Path object where the .tex file will be saved.
    """
    # Create a working copy of the DataFrame
    df_processed = df.copy()

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
    # Convert all speedup values to strings with 2 decimal places
    df_rounded = df_processed.copy()
    for col in df_processed.columns[1:]:
        df_rounded[col] = df_processed[col].map(
            lambda x: f"{x:.2f}" if pd.api.types.is_number(x) else str(x)
        )

    def bold_max(row):
        """Helper function to find and bold the maximum value in a DataFrame row."""
        # Convert string numbers to numeric, coercing errors to NaN
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

    # --- Step 3: Add and format the Geomean row ---
    geo_row_data = {"Task": "Geomean"}
    for remapped_title in df_bold.columns[1:]:
        original_title = remapped_to_original.get(remapped_title)
        if original_title and original_title in geomeans:
            val = geomeans[original_title]
            geo_row_data[remapped_title] = f"{val:.2f}"
        else:
            geo_row_data[remapped_title] = "-"  # Placeholder if not found

    geo_row = pd.Series(geo_row_data)

    # Append the Geomean row and bold its maximum value
    df_with_geo = pd.concat([df_bold, pd.DataFrame([geo_row])], ignore_index=True)
    df_with_geo.iloc[-1] = bold_max(df_with_geo.iloc[-1])

    # --- Step 4: Convert to LaTeX format and save ---
    # Use 'l' for left-align (Task) and 'r' for right-align (numbers)
    column_format = 'l' + 'r' * (len(df_with_geo.columns) - 1)
    latex_table = df_with_geo.to_latex(
        index=False, escape=False, column_format=column_format
    )

    # Insert a horizontal line rule (`\midrule`) before the Geomean row for clarity
    lines = latex_table.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("Geomean"):
            lines.insert(i, "\\midrule")
            break
    latex_table = "\n".join(lines)

    # Save the final LaTeX table to the specified file
    with open(output_path, "w") as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {output_path}")
