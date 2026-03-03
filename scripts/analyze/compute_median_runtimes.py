import os
import json
import shutil
from pathlib import Path
from scipy.stats import gmean
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gmean

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

runtimes_dir = (curr_dir / "../../results/ours/h100_level3-metr/results/data/runtimes").resolve()

filenames = [
    "prev_agents_run0.json",
    "prev_agents_run1.json",
    "prev_agents_run2.json",
]

data = []

for filename in filenames:
    with open(runtimes_dir / filename) as f:
        data.append(json.load(f))

output_data = {
    "title": "prev_agents_median",
    "results": [],
}

final_vals = []

for i in range(len(data[0]["results"])):
    task = data[0]["results"][i]["problem_id"]
    vals = []
    for d in data:
        p = d["results"][i]

        d_task = p["problem_id"]

        if d_task != task:
            raise Exception("Bad")

        vals.append(p["runtime"])
    
    vals_np = np.array(vals)

    median = np.median(vals_np)
    
    print(f"Task: {task}, vals: {vals}, median: {median}")

    final_vals.append(median)

    output_data["results"].append({
        "problem_id": task,
        "runtime": median,
    })

# final_vals_np = np.array(final_vals)

with open(runtimes_dir / "prev_agents_median.json", "w") as f:
    json.dump(output_data, f, indent=4)
