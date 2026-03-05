import argparse
import os
import json
import shutil
from pathlib import Path
from scipy.stats import gmean
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- Constants ---

BLACKLIST = {
    # "h100_level_3-pike_trial_4": {
    #     # (13, 229, 0), # Blacklists a specific attempt
    #     13, # Sets speedup to 1.0 for a task
    # },
    # "h100_level_3-pike_trial_0": {
    #     (40, 4, 297, 1), # Blacklists a specific attempt
    #     # 42, # Sets speedup to 1.0 for a task
    # },
    # "h100_level_3-pike_prev_noagents_trial_0": {
    #     39,
    # },
    # "h100_level_3-pike_prev_noagents_trial_1": {
    #     37,
    #     39,
    #     42,
    # },
    # "h100_level_3-pike_prev_agents_cheap_efa_0": {
    #     41,
    # }
    "h100_level_5_prev_agents_trial_2": {
        # (6, 36, 2),
        # (6, 22, 2),
        # (6, 80, 1),
        # (6, 31, 0),
        # (6, 12, 1),
        # (6, 66, 1),
        # (6, 64, 0),
        # (6, 71, 3),
        # (6, 60, 4),
        # (6, 82, 1),
        # (6, 40, 0),
        # (6, 41, 0),
        # (6, 45, 3),
        # (6, 53, 0),
        # (6, 65, 4),
    },
}

code_blacklist = {
    # "torch.cuda.CUDAGraph",
    "torch.jit.trace",
    # "float16",
    # "half"
}

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

runs_map = {
    "3-pike": [
        ("h100_level_3-pike_pike-b", "prev_agents"),
        ("h100_level_3-pike_pike-b_cheap-efa", "prev_agents_cheap_efa"),
        ("h100_level_3-pike_pike-b_no-efa", "prev_noagents"),
        ("h100_level_3-pike_pike-b_no-iba", "prev_agents_no_iba"),
        ("h100_level_3-pike_pike-o", "openevolve_agents"),
        ("h100_level_3-pike_pike-o_no-efa", "openevolve_noagents"),

        ("h100_level_3-pike_pike-o_mut", "openevolve_agents_mutation"),
        ("h100_level_3-pike_pike-o_mut-npar", "openevolve_agents_no_parallel_eval"),
        ("h100_level_3-pike_pike-o_mut-npar-1isl", "openevolve_agents_no_parallel_eval_no_islands"),
        ("h100_level_3-pike_pike-o_mut-npar-1isl-eo", "openevolve_agents_mut_nopar_noisl_exploitonly"),
        ("h100_level_3-pike_pike-o_mut-npar-1isl-eo-sl", "openevolve_agents_mut_nopar_noisl_exploitonly_shortlib"),

        ("h100_level_3-pike_pike-b_o3-mini-high", "o3-mini-high"),

        ("h100_level_3-pike_pike-b_oss-120b", "oss-120b"),
    ],
    "5": [
        ("h100_level_5_pike-b", "prev_agents"),
        ("h100_level_5_pike-o", "openevolve_agents"),
        ("h100_level_5_pike-o_mut-npar-1isl", "openevolve_agents_no_parallel_eval_no_islands"),
        ("h100_level_5_pike-o_mut-npar-1isl-eo", "openevolve_agents_mut_nopar_noisl_exploitonly"),
        ("h100_level_5_pike-o_mut-npar-1isl-eo-sl", "openevolve_agents_mut_nopar_noisl_exploitonly_shortlib"),
    ],
}


# --- Helper Functions ---

def get_eager_runtime(data, task_number: int):
    """Return eager runtime for given integer task id (or None if missing)."""
    for v in data.get("results", []):
        # tolerate either "problem_id" or "problem" keys if formats vary
        if v.get("problem_id") == task_number or v.get("problem") == task_number:
            return v.get("runtime")
    return None


def numeric_suffix(name: str, prefix: str) -> int:
    """Extract integer suffix from names like 'task12', 'phase_3', 'agent_4', 'step_7'."""
    try:
        return int(name.replace(prefix, "").replace("_", ""))
    except ValueError:
        raise Exception(f"Numeric suffix failed: name -> {name}, prefix -> {prefix}")


def get_llm_query_cost(res_path, is_gemini_pro=True):
    if os.path.exists(res_path):
        with open(res_path) as f:
            res = json.load(f)

        if "usage_metadata" in res:
            usage = res["usage_metadata"]
            prompt_tokens = usage["prompt_token_count"]
            total_tokens = usage["total_token_count"]
        elif "usage" in res:
            usage = res["usage"]
            prompt_tokens = usage["prompt_tokens"]
            total_tokens = usage["total_tokens"]

        res_tokens = total_tokens - prompt_tokens

        if is_gemini_pro:
            return 1.25 * (prompt_tokens / 1e6) + 10 * (res_tokens / 1e6)
        else:
            # flash
            return 0.30 * (prompt_tokens / 1e6) + 2.50 * (res_tokens / 1e6)

    return 0.0


# --- Directory Traversal Logic ---

def get_progress_iters_attempts(task_path, task_number, target_attempt, run_name,
                                 use_cost_stopping_condition, total_step_count,
                                 target_cost, cost_step):
    """
    Traversal logic for the OpenEvolve structure: iter_output/iter/attempts.
    Walks attempts in sorted numeric order, tracking best runtime.
    Returns: (progress_list, best_runtime, best_code_path, best_combo_info)
    """
    cumulative = 0
    best_runtime = float("inf")
    best_code_path = None
    best_combo = None # (iter_num, attempt_num)
    progress_list = []
    cumulative_cost_list = []
    task_cost = 0

    current_blacklist = BLACKLIST.get(run_name, set())

    output_dir = os.path.join(task_path, "output")

    iter_output_dir = os.path.join(output_dir, "iter_output")
    if not os.path.exists(iter_output_dir):
        if use_cost_stopping_condition:
            return [None] * total_step_count, None, None, None, 0, 0, []

        return [None] * target_attempt, None, None, None, 0, 0, []

    ideas_dir = os.path.join(output_dir, "ideas")
    if os.path.exists(ideas_dir):
        res_file = os.path.join(ideas_dir, "raw_response.json")

        idea_cost = get_llm_query_cost(res_file)
        task_cost += idea_cost

    iter_names = sorted([d for d in os.listdir(iter_output_dir) if d.startswith("iter_")], key=lambda x: numeric_suffix(x, "iter_"))

    efa_costs = []

    stop_processing = False
    for iter_name in iter_names:
        iter_num = numeric_suffix(iter_name, "iter_")
        attempts_dir = os.path.join(iter_output_dir, iter_name, "attempts")
        if not os.path.exists(attempts_dir): continue

        attempt_names = sorted([d for d in os.listdir(attempts_dir) if d.startswith("attempt_")], key=lambda x: numeric_suffix(x, "attempt_"))

        for attempt_name in attempt_names:
            attempt_num = numeric_suffix(attempt_name, "attempt_")
            cumulative += 1

            if (task_number, iter_num, attempt_num) not in current_blacklist:
                metrics_file = os.path.join(attempts_dir, attempt_name, "metrics_artifacts.json")
                code_file = os.path.join(attempts_dir, attempt_name, "code.py")

                res_file = os.path.join(attempts_dir, attempt_name, "raw_response.json")

                is_gemini_pro = True
                if run_name == "h100_level_3-pike_pike-b_cheap-efa" and attempt_num > 0:
                    is_gemini_pro = False

                query_cost = get_llm_query_cost(res_file, is_gemini_pro)
                task_cost += query_cost
                cumulative_cost_list.append(task_cost)

                if attempt_num > 0:
                    efa_costs.append(query_cost)

                code_allowed = True
                if os.path.exists(code_file):
                    with open(code_file) as f:
                        code = f.read()

                    for blacklist_match in code_blacklist:
                        if blacklist_match in code:
                            code_allowed = False

                if code_allowed and os.path.exists(metrics_file):
                    try:
                        with open(metrics_file, "r") as f: metrics_data = json.load(f)
                        runtime = metrics_data.get("metrics", {}).get("runtime")
                        if runtime is not None and runtime < best_runtime:
                            best_runtime = runtime
                            best_code_path = code_file if os.path.exists(code_file) else None
                            best_combo = (iter_num, attempt_num)
                    except Exception: pass # Ignore corrupted files or files with missing keys

            progress_list.append(None if best_runtime == float("inf") else best_runtime)
            if use_cost_stopping_condition:
                if task_cost >= target_cost:
                    stop_processing = True
                    break
            else:
                if cumulative >= target_attempt:
                    stop_processing = True
                    break
        if stop_processing:
            break

    print(f"Task cost: ${task_cost:.2f}, steps completed: {cumulative}")

    if use_cost_stopping_condition:
        final_progress_list = []
        curr_cost_max = cost_step
        next_prog_val = float("inf")
        for _ in range(total_step_count):
            for idx, c in enumerate(cumulative_cost_list):
                next_prog_val = progress_list[idx]

                if c > curr_cost_max:
                    break

            final_progress_list.append(next_prog_val)
            curr_cost_max += cost_step
    else:
        final_progress_list = progress_list

        # Pad the progress list if needed
        final_best_for_padding = None if best_runtime == float("inf") else best_runtime
        while len(progress_list) < target_attempt:
            progress_list.append(final_best_for_padding)

    final_best_runtime = None if best_runtime == float("inf") else best_runtime
    return final_progress_list, final_best_runtime, best_code_path, best_combo, cumulative, task_cost, efa_costs


# --- Main Execution Logic ---

def run(run_name, output_label, run_num, input_dir, results_dir, sol_dest_dir,
        runtimes_dirname, tables_dirname, task_blacklist, target_attempt,
        total_step_count, use_cost_stopping_condition, output_solutions,
        target_cost, cost_step):

    root_dir = (input_dir / "full-pike-runs" / run_name / f"runs/runs/run_{run_num}/run/tasks").resolve()

    runtimes_dir = results_dir / f"data/{runtimes_dirname}"
    convergence_dir = results_dir / "figs/convergence"
    speedup_traj_dir = results_dir / f"data/{tables_dirname}/speedup_trajectories"

    os.makedirs(runtimes_dir, exist_ok=True)
    os.makedirs(convergence_dir, exist_ok=True)
    os.makedirs(speedup_traj_dir, exist_ok=True)

    eager_path = runtimes_dir / "eager.json"
    output_path = runtimes_dir / f"{output_label}.json"
    plot_path = convergence_dir / f"{output_label}_convergence.pdf"
    all_trajectories_path = speedup_traj_dir / f"{output_label}.csv"

    # --- Load Eager Runtimes (with robust error handling) ---
    try:
        with open(eager_path) as f:
            eager_runtimes = json.load(f)
    except FileNotFoundError:
        print(f"Warning: eager runtimes file not found at {eager_path}. Continuing with empty eager runtimes (speedups will be skipped).")
        eager_runtimes = {"results": []}
    except Exception as e:
        print(f"Warning: failed to load eager runtimes ({e}). Continuing with empty eager runtimes (speedups will be skipped).")
        eager_runtimes = {"results": []}

    if not isinstance(eager_runtimes, dict) or "results" not in eager_runtimes or not isinstance(eager_runtimes["results"], list):
        print("Warning: eager runtimes file has unexpected format. Expected JSON object with a 'results' list. Continuing with empty results.")
        eager_runtimes = {"results": []}

    sol_dest_dir.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    task_names_unfiltered = sorted([d for d in os.listdir(root_dir) if d.startswith("task")], key=lambda x: int(x.split("task")[1].split("_")[0]))

    task_names = []
    for task_name in task_names_unfiltered:
        if int(task_name.split("task")[1].split("_")[0]) in task_blacklist:
            continue

        task_names.append(task_name)

    results = []
    speedup_list = []
    all_speedups_progress = []
    included_task_names_for_csv = []

    current_blacklist = BLACKLIST.get(run_name, set())

    print(f"Processing run: {run_name}")

    task_step_counts = []
    task_costs = []

    all_efa_costs = []

    for task_name in task_names:
        task_number = int(task_name.split("task")[1].split("_")[0])
        task_path = os.path.join(root_dir, task_name)

        is_task_speedup_blacklisted = task_number in current_blacklist

        progress, best, best_code_path, best_combo, task_step_count, task_cost, efa_costs = get_progress_iters_attempts(
            task_path, task_number, target_attempt, run_name,
            use_cost_stopping_condition, total_step_count, target_cost, cost_step,
        )

        all_efa_costs += efa_costs

        task_step_counts.append(task_step_count)
        task_costs.append(task_cost)

        eager = get_eager_runtime(eager_runtimes, task_number)

        # --- Process Final Result ---
        if best is not None:
            if best_combo:
                combo_str = f"iter_{best_combo[0]}/attempt_{best_combo[1]}"
            else:
                combo_str = "N/A"

            speedup = None
            best_runtime_to_save = float("inf")
            if eager is not None:
                if is_task_speedup_blacklisted:
                    speedup = 1.0
                    best_runtime_to_save = eager
                elif best > 0:
                    speedup = eager / best
                    best_runtime_to_save = best

            results.append({"problem_id": task_number, "runtime": best_runtime_to_save})

            if speedup is not None:
                speedup_list.append(speedup)
                if is_task_speedup_blacklisted:
                    print(f"{task_name} (id={task_number}): best={best:.6f}, eager=N/A, speedup={speedup:.3f} (task blacklisted), at {combo_str}")
                else:
                    print(f"{task_name} (id={task_number}): best={best:.6f}, eager={eager:.6f}, speedup={speedup:.3f}, at {combo_str}")
            else:
                print(f"{task_name} (id={task_number}): best={best:.6f}, eager=N/A, speedup=N/A, at {combo_str}")

            if output_solutions and best_code_path:
                dest_file = sol_dest_dir / f"task_{task_number}.py"
                shutil.copy(best_code_path, dest_file)
        else:
            print(f"{task_name} (id={task_number}): missing runtime, speedup set to 1.0")
            if eager is not None:
                speedup_list.append(1.0)

        if eager is not None:
            if is_task_speedup_blacklisted:
                speedup_progress = [1.0] * total_step_count
            else:
                speedup_progress = [max(1.0, eager / r) if r is not None and r > 0 else 1.0 for r in progress]
            all_speedups_progress.append(speedup_progress)
            task_id = int(task_name.split("task")[1].split("_")[0])
            included_task_names_for_csv.append(f"task_{task_id}")

    all_efa_costs_np = np.array(all_efa_costs)
    efa_query_cost_mean = np.mean(all_efa_costs_np)

    # --- Finalize and Save Results ---

    # 1. Write JSON runtimes
    output_data = {"title": output_label, "results": sorted(results, key=lambda x: x["problem_id"])}
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"\nRuntimes written to {output_path}")

    mean_task_cost = np.mean(np.array(task_costs))
    print(f"\nMean task cost: ${mean_task_cost:.2f}")

    print(f"EFA Query cost mean: ${efa_query_cost_mean:.3f}")

    mean_steps_per_task = np.mean(np.array(task_step_counts))
    print(f"\nMean steps per task: {mean_steps_per_task}")

    # 2. Print geometric mean of speedups
    geomean = None
    if speedup_list:
        speedup_lt1 = [v for v in speedup_list if v < 1]
        print(f"Task count where best speedup is < 1: {len(speedup_lt1)}")
        speedup_list = [v if v > 1 else 1 for v in speedup_list]
        print(speedup_list)

        geomean = gmean(speedup_list)
        print(f"Geometric mean speedup across {len(speedup_list)} tasks = {geomean:.3f}")

    # 3. Generate and save the all_trajectories CSV
    if all_speedups_progress and included_task_names_for_csv:
        df = pd.DataFrame(dict(zip(included_task_names_for_csv, all_speedups_progress)))

        if use_cost_stopping_condition:
            df_idx = []
            for s in range(1, total_step_count + 1):
                df_idx.append(s * cost_step)

            df.index = df_idx
            df.index.name = "cost"
        else:
            df.index = pd.RangeIndex(start=1, stop=total_step_count + 1, name="attempt")

        all_trajectories_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(all_trajectories_path)
        print(f"Speedup trajectories saved to {all_trajectories_path}")

    return geomean


def run_level(input_dir: Path, output_dir: Path, level: str,
              use_cost_stopping: bool = False, output_solutions: bool = False,
              run_name: str = None):
    use_cost_stopping_condition = use_cost_stopping

    target_attempt = 300
    if level == "3-pike":
        target_cost = 25.0
    else:
        target_cost = 50.0
    cost_step = 0.2
    if use_cost_stopping_condition:
        total_step_count = round(target_cost / cost_step)
    else:
        total_step_count = target_attempt

    target_dirname = f"h100_level_{level}"
    results_dir = (output_dir / target_dirname / "results").resolve()

    if use_cost_stopping_condition:
        runtimes_dirname = "runtimes_money_budget"
        tables_dirname = "tables_money_budget"
        overall_speedups_filename = "speedups_money_budget.json"
    else:
        runtimes_dirname = "runtimes"
        tables_dirname = "tables"
        overall_speedups_filename = "speedups.json"

    task_blacklist = task_blacklist_map.get(level, set())

    if run_name is not None:
        # Find the output label from runs_map, falling back to the run name itself
        label_map = dict(runs_map.get(level, []))
        output_label = label_map.get(run_name, run_name)
        runs = [(run_name, output_label)]
    else:
        runs = runs_map[level]

    speedups = {}
    for (run_name, output_label) in runs:
        sol_dest_dir = (input_dir / "best-pike-kernels" / run_name).resolve()
        geomean_np = run(
            run_name, output_label, 0,
            input_dir, results_dir, sol_dest_dir,
            runtimes_dirname, tables_dirname, task_blacklist,
            target_attempt, total_step_count, use_cost_stopping_condition,
            output_solutions, target_cost, cost_step,
        )
        if geomean_np is not None:
            speedups[output_label] = float(geomean_np)

    print("\n========= ALL SPEEDUPS =========")
    print(json.dumps(speedups, indent=4))

    overall_speedups_dir = results_dir / "data/overall_speedups"
    os.makedirs(overall_speedups_dir, exist_ok=True)

    with open(overall_speedups_dir / overall_speedups_filename, "w") as f:
        json.dump(speedups, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="Path to input data directory (contains full-pike-runs, best-pike-kernels)")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--level", type=str, required=True, choices=["3-pike", "5"], help="Level to process")
    parser.add_argument("--use-cost-stopping", action="store_true", help="Use cost-based stopping condition instead of attempt count")
    parser.add_argument("--output-solutions", action="store_true", help="Copy best solutions to sol_dest_dir")
    args = parser.parse_args()

    run_level(
        Path(args.input_dir).resolve(),
        Path(args.output_dir).resolve(),
        args.level,
        use_cost_stopping=args.use_cost_stopping,
        output_solutions=args.output_solutions,
    )


if __name__ == "__main__":
    main()
