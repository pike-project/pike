Cost modeling notes and results can be found here. The objective is to determine the cost of repairing a program to the point of compiling without errors and being correct, after an LLM initially makes a radical optimization change that we want to explore.

If we give the agent a fixed budget of 5 attempts to try and correct the errors and correctness issues of the radically changed code, what is the total query cost to the LLM (in dollars)?

### Experiment 1

Run: /pscratch/sd/k/kir/llm/KernelBench-data/runs/2025_06_27_11_54_14

**This experiment uses o3, budget of 5 attempts, and task 1-1**

Note that the cost of the initial radical optimization is also included in this total cost.

start: 0.0
end: 0.11919

Notes:
- Loaded and correct on first try

### Experiment 2

**This experiment uses o3, budget of 5 attempts, and task 3-44**

start: 0.11919
end: 0.55458
diff: $0.43

Notes:
- Loaded and correct after 1 error fix attempt
- runtime: 66.202 ms

```bash
python eval.py --level 3 --task 44 --code_path /pscratch/sd/k/kir/llm/KernelBench-data/runs/2025_06_27_12_04_44/step_1/level_3_problem_44_sample_0/kernel.py
```

### Experiment 3

**model: o3, budget: 5 attempts, and task 3-26**

start: 0.55458
end: 0.79827
diff: $0.25

Notes:
- Loaded and correct on first attempt
- runtime: 34.273 ms

```bash
python eval.py --level 3 --task 26 --code_path /pscratch/sd/k/kir/llm/KernelBench-data/runs/2025_06_27_13_38_21/step_0/level_3_problem_26_sample_0/kernel.py
```

Compare against:

```bash
python eval.py --level 3 --task 26 --code_path /pscratch/sd/k/kir/llm/KernelBenchFiltered/best_agent_solutions/level_3/task_26.py
```

Input token count: 9317
Output token count: 9048

Cost assuming $10/$40 input and output cost

.009317 * $10 = $0.09
.009048 * $40 = $0.36

Cost assuming $2/$8

.009317 * $2 = $0.018
.009048 * $40 = $0.072

### Experiment 4

**This experiment uses Gemini 2.5 Pro, budget of 5 attempts, and task 3-44**

start: 0.79827
end: 
diff: $

Notes:
- 
- runtime: 

```bash
```

### Experiment 5

**This experiment uses Gemini 2.5 Pro, budget of 5 attempts, and task 3-26**

start: 
end: 
diff: $

Notes:
- 
- runtime: 

```bash
```
