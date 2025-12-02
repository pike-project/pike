
import os
import json
import shutil
from pathlib import Path

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

run_dir = (curr_dir / "../../../ropenevolve/examples/kernelbench/openevolve_output_lrc/h100_level_3-metr_trial_2/tasks").resolve()

sol_dest_dir = (curr_dir / "../../best_agent_solutions/h100/level3-metr/openevolve_pop_10_no_agents_300/best_solutions").resolve()
os.makedirs(sol_dest_dir, exist_ok=True)

tasks = []

for task_dirname in os.listdir(run_dir):
    task = int(task_dirname.split("task")[1])
    tasks.append(task)

tasks = sorted(tasks)

results = []

total_input_tokens = 0
total_output_tokens = 0

for task in tasks:
    task_dirname = f"task{task}"
    task_dir = run_dir / task_dirname

    best_info_path = task_dir / "output/best/best_program_info.json"

    with open(best_info_path) as f:
        best_info = json.load(f)
    
    runtime = best_info["metrics"]["runtime"]

    print(task, runtime)

    results.append({
        "problem_id": task,
        "runtime": runtime,
        # "results": {
        #     "eval_results": {
        #         "runtime": runtime,
        #     }
        # }
    })

    sol_src = task_dir / "output/best/best_program.py"
    sol_dest = sol_dest_dir / f"task_{task}.py"

    shutil.copy(sol_src, sol_dest)

    raw_responses_dir = task_dir / "output/raw_responses"

    for filename in os.listdir(raw_responses_dir):
        res_path = raw_responses_dir / filename
        with open(res_path) as f:
            raw_resp = json.load(f)

        usage_data = raw_resp["usage"]
        total_tokens = usage_data["total_tokens"]
        prompt_tokens = usage_data["prompt_tokens"]
        # completion_tokens = usage_data["completion_tokens"]

        # it seems that Google's OpenAI-style completion API does not show
        # reasoning tokens. It includes "completion_tokens", but prompt_tokens and
        # completion_tokens added together DOES NOT sum to total_tokens:
        # -----------> prompt_tokens + completion_tokens < total_tokens

        # this leads to the conservative assumption that the missing tokens must be counted
        # as output tokens, which are more expensive than input tokens, likely
        # the reasoning tokens
        resp_tokens = total_tokens - prompt_tokens

        total_input_tokens += prompt_tokens
        total_output_tokens += resp_tokens

print(f"Total input tokens used: {total_input_tokens}, total output tokens used: {total_output_tokens}")

# Gemini 2.5 Pro current prices per 1M tokens
input_price = 1.25
output_price = 10

input_cost = total_input_tokens * 1e-6 * input_price
output_cost = total_output_tokens * 1e-6 * output_price
total_cost = input_cost + output_cost

print(f"Input cost: ${input_cost:.2f}, Output cost: ${output_cost:.2f}")

print(f"Total cost: ${total_cost:.2f}")

avg_cost_per_task = total_cost / len(tasks)

print(f"Avg cost per task: ${avg_cost_per_task:.2f}")

out = {
    "title": "Ours (OpenEvolve)",
    "results": results,
}

# output_path = (curr_dir / "../../results/ours/h100_level3-metr/data/runtimes/ours_openevolve.json").resolve()

# with open(output_path, "w") as f:
#     json.dump(out, f, indent=4)
