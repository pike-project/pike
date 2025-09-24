import os
import json
import shutil
from pathlib import Path
from scipy.stats import gmean

target_attempt = 300

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

root_dir = (curr_dir / "../../data/runs/h100_level_3-metr_trial_0/levels/level_3-metr").resolve()

eager_path = (curr_dir / "../../results/breakdowns/h100_level3-metr/data/runtimes/eager.json").resolve()
sol_dest_dir = (curr_dir / "../../best_agent_solutions/h100/level3-metr/prev_agents_300/best_solutions").resolve()
output_path = (curr_dir / "../../results/breakdowns/h100_level3-metr/data/runtimes/prev_agents.json").resolve()


# Load eager runtimes
with open(eager_path) as f:
    eager_runtimes = json.load(f)


def get_runtime(data, task_number: int):
    """Return eager runtime for given integer task id."""
    for v in data["results"]:
        if v["problem_id"] == task_number:
            return v["runtime"]
    return None


def numeric_suffix(name: str, prefix: str) -> int:
    """Extract integer suffix from names like 'task12', 'phase_3', 'agent_4', 'step_7'."""
    try:
        return int(name.replace(prefix, "").replace("_", ""))
    except ValueError:
        raise Exception(f"Numeric suffix failed: name -> {name}, prefix -> {prefix}")


def best_runtime_until_step(task_path, target_attempt):
    """
    Walk steps in sorted numeric order across all phases/agents,
    track the best runtime seen so far.
    Stop at target_attempt and return the best runtime and corresponding kernel.py path.
    """
    cumulative = 0
    best_runtime = float("inf")
    best_code_path = None

    phases_root = os.path.join(task_path, "phases")
    if not os.path.exists(phases_root):
        return None, None

    phase_names = [
        d for d in os.listdir(phases_root)
        if os.path.isdir(os.path.join(phases_root, d)) and d.startswith("phase_")
    ]
    phase_names.sort(key=lambda x: numeric_suffix(x, "phase_"))

    for phase_name in phase_names:
        phase_path = os.path.join(phases_root, phase_name)
        agents_root = os.path.join(phase_path, "agents")
        if not os.path.exists(agents_root):
            continue

        agent_names = [
            d for d in os.listdir(agents_root)
            if os.path.isdir(os.path.join(agents_root, d)) and d.startswith("agent_")
        ]
        agent_names.sort(key=lambda x: numeric_suffix(x, "agent_"))

        for agent_name in agent_names:
            agent_path = os.path.join(agents_root, agent_name)
            step_names = [
                d for d in os.listdir(agent_path)
                if os.path.isdir(os.path.join(agent_path, d)) and d.startswith("step_")
            ]
            step_names.sort(key=lambda x: numeric_suffix(x, "step_"))

            for step_name in step_names:
                cumulative += 1
                step_path = os.path.join(agent_path, step_name)
                eval_file = os.path.join(step_path, "eval_results.json")
                code_file = os.path.join(step_path, "kernel.py")

                if os.path.exists(eval_file):
                    try:
                        with open(eval_file, "r") as f:
                            eval_data = json.load(f)
                        runtime = eval_data.get("eval_results", {}).get("runtime")
                        if runtime is not None and runtime < best_runtime:
                            best_runtime = runtime
                            if os.path.exists(code_file):
                                best_code_path = code_file
                    except Exception:
                        pass

                if cumulative == target_attempt:
                    return (best_runtime if best_runtime != float("inf") else None,
                            best_code_path)

    return (None if best_runtime == float("inf") else best_runtime,
            best_code_path)


if __name__ == "__main__":
    sol_dest_dir.mkdir(parents=True, exist_ok=True)

    # Collect tasks
    task_names = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("task")
    ]
    task_names.sort(key=lambda x: numeric_suffix(x, "task"))

    results = []
    speedup_list = []

    for task_name in task_names:
        task_number = numeric_suffix(task_name, "task")
        task_path = os.path.join(root_dir, task_name)
        best, best_code_path = best_runtime_until_step(task_path, target_attempt)
        eager = get_runtime(eager_runtimes, task_number)

        if best is not None:
            # Save runtime to results
            results.append({
                "problem_id": task_number,
                "runtime": best
            })
            print(f"{task_name} (id={task_number}): best runtime = {best:.6f}")

            # Compute speedup if eager exists
            if eager is not None and best > 0:
                speedup = eager / best

                if task_number == 40:
                    speedup = 1.0

                speedup_list.append(speedup)
                print(f"  eager={eager:.6f}, speedup={speedup:.3f}")
            else:
                print("  missing eager runtime, skipping speedup")

            # Save best code
            if best_code_path:
                dest_file = sol_dest_dir / f"task_{task_number}.py"
                shutil.copy(best_code_path, dest_file)
        else:
            print(f"{task_name} (id={task_number}): missing runtime")

    # Write JSON in requested format
    output_data = {
        "title": "Ours (OE, agents)",
        "results": sorted(results, key=lambda x: x["problem_id"])
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"\nRuntimes written to {output_path}")

    # Print geometric mean of speedups
    if speedup_list:
        geo_mean = gmean(speedup_list)
        print(f"\nGeometric mean speedup across {len(speedup_list)} tasks = {geo_mean:.3f}")
