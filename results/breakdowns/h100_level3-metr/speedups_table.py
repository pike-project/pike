import pandas as pd
from scipy.stats import gmean

# Input and output paths
input_csv = "data/speedups_table.csv"
output_tex = "data/speedups_table.tex"

# Optional remappings
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
    "Ours (OE, agents)": "Ours",
    "torch.compile": "comp",
    "METR": "METR",
    "TensorRT": "TRT",
    # add more mappings as needed
}

# Read CSV
df = pd.read_csv(input_csv)

# Remap task names
df["Task"] = df["Task"].map(lambda x: task_name_remapping.get(x, x))

# Remap column titles
df.rename(columns=title_remapping, inplace=True)

# Compute geomean for each method (skip Task column)
geomeans = {col: gmean(df[col].astype(float).values) for col in df.columns[1:]}

# Round floats to 2 decimal places (as strings for LaTeX)
df_rounded = df.copy()
for col in df.columns[1:]:
    df_rounded[col] = df[col].map(lambda x: f"{x:.2f}" if pd.api.types.is_number(x) else x)

# Bold max values per row
def bold_max(row):
    numeric_vals = pd.to_numeric(row[1:], errors="coerce")
    max_val = numeric_vals.max(skipna=True)
    for col in row.index[1:]:
        try:
            val = float(row[col])
            if round(val, 2) == round(max_val, 2):
                row[col] = f"\\textbf{{{row[col]}}}"
        except Exception:
            continue
    return row

df_bold = df_rounded.apply(bold_max, axis=1)

# Add Geomean row
geo_row = {"Task": "Geomean"}
for col in df.columns[1:]:
    val = round(geomeans[col], 2)
    geo_row[col] = f"{val:.2f}"
geo_row = pd.Series(geo_row)

df_bold = pd.concat([df_bold, pd.DataFrame([geo_row])], ignore_index=True)

# Bold max in Geomean row
df_bold.iloc[-1] = bold_max(df_bold.iloc[-1])

# Convert to LaTeX
latex_table = df_bold.to_latex(index=False, escape=False)

# Insert a horizontal line before the Geomean row
lines = latex_table.splitlines()
for i, line in enumerate(lines):
    if line.strip().startswith("Geomean"):
        lines.insert(i, "\\midrule")
        break
latex_table = "\n".join(lines)

# Save LaTeX table
with open(output_tex, "w") as f:
    f.write(latex_table)

print(f"LaTeX table written to {output_tex}")
