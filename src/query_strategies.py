import numpy as np

def get_simple_branching_query(code, runtime, problem_id):
    return f"""Please improve the following code:

```python
{code}
```
"""

# returns a set of prompts
def simple_branching_strategy(sorted_solutions, num_samples, problem_id) -> list[str]:
    # TODO: should we create a next round based on just 1-2 solutions, if there are only that many?
    if len(sorted_solutions) < 3:
        return []

    sample_idxs = np.array(list(range(num_samples)))

    bins_np = np.array_split(sample_idxs, 3)
    # bins = [b.tolist() for b in bins_np]
    bin_sizes = [len(b) for b in bins_np]

    queries = []

    for sol_idx, bin_size in enumerate(bin_sizes):
        sol = sorted_solutions[sol_idx]
        code = sol["code"]
        runtime = sol["runtime"]

        for _ in range(bin_size):
            q = get_simple_branching_query(code, runtime, problem_id)
            queries.append(q)

    return queries

def test_simple_branching_strategy():
    sorted_sols = [
        {
            "code": "code1",
            "runtime": 0.1
        },
        {
            "code": "code2",
            "runtime": 0.2
        },
        {
            "code": "code3",
            "runtime": 0.3
        },
    ]

    qs = simple_branching_strategy(sorted_sols, 10, 1)

    assert len(qs) == 10, "Length of queries should be 10"

    for q in qs:
        print(q)

if __name__ == "__main__":
    test_simple_branching_strategy()
