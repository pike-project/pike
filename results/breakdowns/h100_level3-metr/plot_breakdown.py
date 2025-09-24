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
data_dir = (curr_dir / "results/data/runtimes").resolve()

output_label = "prev"
plot_mode = "line"  # choose "line" or "bar"

# included_files = ["eager", "prev_agents", "compile"]
# included_files = ["eager", "oe_agents", "compile", "tensorrt"]
included_files = ["eager", "prev_agents", "compile", "metr", "tensorrt"]

primary_str_match = "ours (prev, agents)"
# primary_str_match = "ours (oe, agents)"

# --- LaTeX Table Configuration ---
task_name_remapping = {
    "DenseNet121TransitionLayer": "DenseNet121TL",
    "GoogleNetInceptionModule": "GoogleNetIM",
    "DenseNet121DenseBlock": "DenseNet121DB",
    "GRUBidirectionalHidden": "GRUBidirectionalH",
    "MinGPTCausalAttention": "MinGPTCausalAtt",
    "Mamba2ReturnFinalState": "Mamba2ReturnFinalS",
    "SqueezeNetFireModule": "SqueezeNetFireMod",
    # add more mappings as needed
}

title_remapping = {
    "Ours (prev, agents)": "Ours",
    "Ours (oe, agents)": "Ours",
    "Ours (openevolve)": "Ours",
    "torch.compile": "comp",
    "METR": "METR",
    "TensorRT": "TRT",
    # add more mappings as needed
}

# ==============================================================================
# --- LaTeX Table Generation Function ---
# ==============================================================================

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

# ==============================================================================
# --- Main Script Logic ---
# ==============================================================================

# --- Load all runtimes ---
all_methods = {}
file_to_title_map = {}

for file in data_dir.glob("*.json"):
    if file.stem not in included_files:
        continue

    with open(file) as f:
        data = json.load(f)
    title = data["title"]
    results = {entry["problem_id"]: entry["runtime"] for entry in data["results"]}
    all_methods[title] = results

    # Build map dynamically
    file_to_title_map[file.stem] = title

# Find eager baseline
eager_title = file_to_title_map.get("eager", None)
if eager_title is None:
    raise ValueError("Missing baseline 'Eager.json' in data_dir")
eager_key = next((k for k in all_methods if k.lower() == eager_title.lower()), None)
eager_runtimes = all_methods[eager_key]

# --- Determine tasks from our primary sorting key ---
primary_key = next((k for k in all_methods if k.lower() == primary_str_match.lower()), None)
if primary_key is None:
    raise ValueError(f"Missing primary key method '{primary_str_match}' in data_dir")
included_tasks = list(all_methods[primary_key].keys())

# --- Compute speedups ---
methods_speedups = {title: [] for title in all_methods if title != eager_key}

for task in included_tasks:
    eager_runtime = eager_runtimes.get(task)
    if eager_runtime is None:
        print(f"Warning: Task {task} missing baseline runtime, skipping.")
        continue

    for title, runtimes in all_methods.items():
        if title == eager_key:
            continue
        method_runtime = runtimes.get(task)
        if method_runtime is None or eager_runtime is None:
            speedup = 1.0
            print(f"Warning: Task {task}, method '{title}' has runtime None. Setting speedup=1.")
        else:
            speedup = eager_runtime / method_runtime
            if speedup < 1.0:
                speedup = 1.0
        methods_speedups[title].append(speedup)

# --- Task labels ---
task_labels_map = {}
level_dir = (curr_dir / "../../../KernelBench/level3-metr").resolve()
for filename in os.listdir(level_dir):
    if not filename.endswith(".py"):
        continue
    task = int(filename.split("_")[0])
    label = filename.split("_")[1].split(".py")[0]
    task_labels_map[task] = label

labels = [task_labels_map.get(t, str(t)) for t in included_tasks]

# --- Sort tasks by primary key ascending ---
sort_indices = np.argsort(methods_speedups[primary_key])  # ascending
included_tasks = [included_tasks[i] for i in sort_indices]
labels_sorted = [labels[i] for i in sort_indices]
methods_speedups = {k: [v[i] for i in sort_indices] for k, v in methods_speedups.items()}

# --- Compute geometric mean ---
geomeans = {}
for name, values in methods_speedups.items():
    arr = np.array(values, dtype=float)
    arr = arr[arr > 0]  # filter out invalid/missing
    geomeans[name] = np.exp(np.mean(np.log(arr)))

# --- Plotting ---
x = np.arange(len(included_tasks))
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
plt.title("Level 3-metr Speedup Over PyTorch Eager (H100)")
plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
plt.ylabel("Speedup")
plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.3, top=0.95)
ax.legend(loc='upper left', fontsize=10)
plt.yscale("log")

# --- Save plot ---
figs_dir = (curr_dir / "results/figs/breakdown").resolve()
os.makedirs(figs_dir, exist_ok=True)
fig.savefig(figs_dir / f"{output_label}_{plot_mode}.pdf")
print(f"Plot saved to: {figs_dir / f'{output_label}_{plot_mode}.pdf'}")

# --- Prepare DataFrame for CSV and TeX ---
# Enforce CSV columns order the same as included_files
ordered_cols = [file_to_title_map[f] for f in included_files if f != "eager" and f in file_to_title_map]
df = pd.DataFrame({name: methods_speedups[name] for name in ordered_cols}, index=labels_sorted)
df.reset_index(inplace=True)
df.rename(columns={"index": "Task"}, inplace=True)

# --- Create output directories ---
tables_dir = curr_dir / "results/data/tables"
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
