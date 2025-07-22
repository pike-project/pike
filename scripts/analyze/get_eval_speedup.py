import os
import json
from pathlib import Path
import numpy as np
from scipy.stats import gmean

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

def main():
    # baseline_path = Path("/pscratch/sd/k/kir/llm/KernelBench-data/runs/final_run2/baseline_compile.json")
    # comp_path = Path("/pscratch/sd/k/kir/llm/KernelBench/results/eval_solutions/good_kernels_src_filtered.json")
    baseline_path = Path("/pscratch/sd/k/kir/llm/KernelBench/results/eval_solutions/baseline/compile/2025_07_22_15_33_29.json")
    comp_path = Path("/pscratch/sd/k/kir/llm/KernelBench/results/eval_solutions/metr/2025_07_22_14_41_23.json")

    with open(baseline_path) as f:
        baseline_data = json.load(f)
    
    with open(comp_path) as f:
        comp_data = json.load(f)
    
    # if len(baseline_data) != len(comp_data):
    #     raise Exception("baseline and comp data not same length")

    speedups = []

    for c_val in comp_data:
        problem_id = c_val["problem_id"]

        c_res_full = c_val["results"]
        if "eval_results" not in c_res_full:
            continue

        c_results = c_res_full["eval_results"]

        if "loaded" not in c_results or not c_results["loaded"] or "correct" not in c_results or not c_results["correct"]:
            continue
        
        runtime_comp = c_results["runtime"]

        b_val = None
        for b_res in baseline_data:
            if b_res["problem_id"] == problem_id:
                b_val = b_res
                break

        if b_val is None:
            raise Exception(f"Matching problem id in baseline not found: {problem_id}")
        
        runtime_baseline = b_val["results"]["eval_results"]["runtime"]

        speedup = runtime_baseline / runtime_comp

        print(f"Task {problem_id} speedup: {speedup}")

        speedups.append(speedup)
    
    speedups_np = np.array(speedups)

    speedup_gmean = gmean(speedups_np)

    print(f"Speedup Geomean: {speedup_gmean}")

if __name__ == "__main__":
    main()
