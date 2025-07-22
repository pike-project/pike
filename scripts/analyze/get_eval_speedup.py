import os
import json
from pathlib import Path
import numpy as np
from scipy.stats import gmean

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

def main():
    baseline_path = Path("/pscratch/sd/k/kir/llm/KernelBench-data/runs/final_run2/baseline_compile.json")
    comp_path = Path("/pscratch/sd/k/kir/llm/KernelBench/results/eval_solutions/good_kernels_src_filtered.json")

    with open(baseline_path) as f:
        baseline_data = json.load(f)
    
    with open(comp_path) as f:
        comp_data = json.load(f)
    
    if len(baseline_data) != len(comp_data):
        raise Exception("baseline and comp data not same length")

    speedups_np = np.zeros(len(baseline_data))

    for idx, (b_val, c_val) in enumerate(zip(baseline_data, comp_data)):
        if b_val["problem_id"] != c_val["problem_id"]:
            raise Exception("problem_id values do not match")
        
        runtime_baseline = b_val["results"]["eval_results"]["runtime"]
        runtime_comp = c_val["results"]["eval_results"]["runtime"]

        speedup = runtime_baseline / runtime_comp

        speedups_np[idx] = speedup
    
    speedup_gmean = gmean(speedups_np)

    print(f"Speedup Geomean: {speedup_gmean}")

if __name__ == "__main__":
    main()
