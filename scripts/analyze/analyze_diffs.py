import os
import json
import difflib
import shutil
import tiktoken
import subprocess
from pathlib import Path
from openai import OpenAI
import numpy as np

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

target_attempt_count = 300

# run_name = "h100_level_3-metr_prev_agents_trial_1"
run_name = "h100_level_3-metr_openevolve_agents_trial_0"

root_dir = (curr_dir / "../../data/parallel_runs" / run_name / "runs/runs/run_0/run/tasks").resolve()

output_dir = (curr_dir / "../../data/diffs" / run_name).resolve()
samples_dir = output_dir / "samples"
embeddings_dir = output_dir / "embeddings"

os.makedirs(samples_dir, exist_ok=True)

# iterate tasks in the root dir, and for each task,
# go through attempt_0 at each iter, up to the cumulative attempt number of target_attempt
# (which should include attempts outside of just attempt_0)
# For each attempt_0, save for this task the pair of (prompt.md, code.py)
# where these pieces of data are the raw string read in from that particular file in the
# attempt_0 directory at that iter

# --- Helper Function ---

# def diff_counts(a, b):
#     if isinstance(a, str):
#         a = a.splitlines()
#     if isinstance(b, str):
#         b = b.splitlines()

#     diff = list(difflib.unified_diff(a, b, lineterm=''))
#     added = sum(1 for line in diff if line.startswith('+ ') and not line.startswith('+++'))
#     removed = sum(1 for line in diff if line.startswith('- ') and not line.startswith('---'))
#     return added, removed

def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def save_embedding_response(text, output_path):
    emb_res = client.embeddings.create(input=text, model="text-embedding-3-large")
    emb = emb_res.data[0].embedding
    np_arr = np.array(emb)
    # with open(output_path, "w") as f:
    #     json.dump(emb_res.to_dict(), f, indent=4)
    
    np.save(output_path, np_arr)

def diff_counts(p1, p2):
    res_removed = subprocess.run(
        f'diff -u {p1} {p2} | grep -E "^\-" | wc -l',
        shell=True,
        capture_output=True,
        text=True
    )

    removed = int(res_removed.stdout.strip()) - 1

    res_added = subprocess.run(
        f'diff -u {p1} {p2} | grep -E "^\+" | wc -l',
        shell=True,
        capture_output=True,
        text=True
    )

    added = int(res_added.stdout.strip()) - 1

    return added, removed

def strip_whitespace_and_comments(code: str) -> str:
    lines = code.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # skip blank or comment-only lines
        if not stripped or stripped.startswith("#"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def numeric_suffix(name: str, prefix: str) -> int:
    """Extract integer suffix from names like 'task12', 'iter_3'."""
    try:
        return int(name.replace(prefix, "").replace("_", ""))
    except ValueError:
        raise ValueError(f"Could not extract numeric suffix from '{name}' with prefix '{prefix}'")

# --- Main Traversal and Data Extraction Logic ---

# Dictionary to hold the final results.
# Structure: { "task_0": [(prompt_content_iter_0, kernel_content_iter_0), (prompt_content_iter_1, kernel_content_iter_1), ...], ... }
task_data = {}

print(f"Starting traversal of: {root_dir}\n")

try:
    # Get a list of all directories in root_dir starting with "task"
    task_dirs = [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("task")]
    # Sort the task directories numerically
    sorted_task_dirs = sorted(task_dirs, key=lambda d: numeric_suffix(d.name, "task"))

except FileNotFoundError:
    print(f"Error: Root directory not found at '{root_dir}'. Please check the path.")
    sorted_task_dirs = []

# Iterate through each sorted task directory
for task_path in sorted_task_dirs:
    task_name = task_path.name
    task_data[task_name] = []

    iter_output_dir = task_path / "output" / "iter_output"

    if not iter_output_dir.exists():
        print(f"  -> No 'iter_output' directory found. Skipping.")
        continue

    # Get and sort the iteration directories within the task
    try:
        iter_dirs = [d for d in iter_output_dir.iterdir() if d.is_dir() and d.name.startswith("iter_")]
        sorted_iter_dirs = sorted(iter_dirs, key=lambda d: numeric_suffix(d.name, "iter_"))
    except FileNotFoundError:
        # This case is unlikely if iter_output_dir exists, but good for safety
        print(f"  -> 'iter_output' directory disappeared or is unreadable. Skipping.")
        continue
    
    total_attempt_count = 0

    # Iterate through each sorted iteration directory
    for iter_path in sorted_iter_dirs:
        iter_number = int(iter_path.name.split("_")[1])

        attempts_dir = iter_path / "attempts"

        attempt_count = len(os.listdir(attempts_dir))
        total_attempt_count += attempt_count
        if total_attempt_count > target_attempt_count:
            break

        attempt_0_path = attempts_dir / "attempt_0"

        # We are only interested in attempt_0 for each iteration
        if not attempt_0_path.exists():
            continue

        prompt_file = attempt_0_path / "prompt.md"
        kernel_file = attempt_0_path / "code.py"

        # Check if both required files exist
        if prompt_file.exists() and kernel_file.exists():
            try:
                # Read the raw string content from each file
                prompt_content = prompt_file.read_text(encoding='utf-8')
                kernel_content = kernel_file.read_text(encoding='utf-8')

                # Save the pair of strings for this task
                task_data[task_name].append((iter_number, prompt_content, kernel_content))

            except Exception as e:
                print(f"  -> Error reading files in {attempt_0_path}: {e}")

    print(f"Processed {task_name}, total attempts: {total_attempt_count}")

# --- Verification ---
print("\n--- Traversal Complete ---")
print("Summary of collected data:")
total_pairs = 0
total_tokens = 0

means = []

for task_raw, data_list in task_data.items():
    task_num = int(task_raw.split("task")[1])
    task_dirname = f"task_{task_num}"

    num_pairs = len(data_list)
    total_pairs += num_pairs
    print(f"- Task '{task_dirname}': Found {num_pairs} pairs of (prompt.md, code.py) from attempt_0.")

    total_lines_changed = 0

    for idx, (iter_number, prompt, code) in enumerate(data_list):
        seed = prompt.split("```python\n")[-1].split("```")[0]
        sample_dir = samples_dir / task_dirname / f"sample_{idx}"

        code_stripped = strip_whitespace_and_comments(code)
        seed_stripped = strip_whitespace_and_comments(seed)

        os.makedirs(sample_dir, exist_ok=True)

        code_tokens = num_tokens_from_string(code_stripped)
        seed_tokens = num_tokens_from_string(seed_stripped)

        total_tokens += code_tokens + seed_tokens

        seed_path = sample_dir / "seed.py"
        with open(seed_path, "w") as f:
            f.write(seed_stripped)

        code_path = sample_dir / "code.py"
        with open(code_path, "w") as f:
            f.write(code_stripped)
        
        diff_added, diff_removed = diff_counts(seed_path, code_path)
        lines_changed = diff_added + diff_removed
        total_lines_changed += lines_changed

        # max_tokens = 8192
        # if code_tokens > max_tokens or seed_tokens > max_tokens:
        #     print(f"\tSkipping sample {idx}, too many tokens")
        #     continue

        # emb_dir = embeddings_dir / task_dirname / f"sample_{idx}"

        # if not os.path.isdir(emb_dir):
        #     os.makedirs(emb_dir, exist_ok=True)

        #     save_embedding_response(seed_stripped, emb_dir / "seed.npy")
        #     save_embedding_response(code_stripped, emb_dir / "code.npy")

    mean_lines_changed = total_lines_changed / len(data_list)
    means.append(mean_lines_changed)

    print(f"{task_raw} mean lines changed: {mean_lines_changed}")

with open(output_dir / "means.json", "w") as f:
    json.dump(means, f, indent=4)

print(f"Total tokens: {total_tokens}")
print(f"\nTotal pairs collected across all tasks: {total_pairs}")
