# parse_and_sort_stacks.py
from pathlib import Path

# Path to your log file
file_path = Path("log/torch/stacks_gpu.out")

stacks = []

# Read and parse each line
with file_path.open("r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Split stack trace and runtime (last whitespace-separated token)
        *trace_parts, runtime_str = line.rsplit(maxsplit=1)
        runtime = int(runtime_str)
        stack_trace = " ".join(trace_parts)
        stacks.append((runtime, stack_trace))

# Sort by runtime descending
stacks.sort(reverse=True, key=lambda x: x[0])

# Print sorted stack traces with runtimes
for runtime, stack_trace in stacks:
    print(f"{stack_trace} {runtime}")
