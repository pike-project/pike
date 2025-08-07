import os
import json
from pathlib import Path
import numpy as np
from scipy.stats import gmean
import argparse

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

def main(baseline_path, comp_path):
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

        b_val = None
        for b_res in baseline_data:
            if b_res["problem_id"] == problem_id:
                b_val = b_res
                break

        if b_val is None:
            print(f"Matching problem id in baseline not found: {problem_id}")
            continue

        try:
            runtime_baseline = b_val["results"]["eval_results"]["runtime"]
        except Exception as e:
            print(f"Baseline execution failed for task: {problem_id}")
            continue

        try:
            if "eval_results" not in c_res_full:
                continue

            c_results = c_res_full["eval_results"]

            if "loaded" not in c_results or not c_results["loaded"] or "correct" not in c_results or not c_results["correct"]:
                continue
            
            runtime_comp = c_results["runtime"]

            speedup = runtime_baseline / runtime_comp
        except Exception as e:
            print(c_res_full)
            raise e

        print(f"Task {problem_id} speedup: {speedup}")

        speedups.append(speedup)
    
    speedups_np = np.array(speedups)

    speedup_gmean = gmean(speedups_np)

    print(f"Speedup Geomean: {speedup_gmean}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_path", type=str, required=True)
    parser.add_argument("--comp_path", type=str, required=True)
    args = parser.parse_args()

    main(Path(args.baseline_path), Path(args.comp_path))
