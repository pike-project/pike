import numpy as np
import src.prompt_constructor_new as prompt

# the simple branching strategy requires at least 3 correct solutions, and it
# returns queries which focus on one of the top 3 solutions (in terms of runtime)
# splitting work evenly between the 3 solutions
# - returns a list of prompts
def simple_branching_strategy(sorted_solutions, num_samples, problem_code) -> list[str]:
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
            query = prompt.prompt_improve_solution(problem_code, code)
            queries.append(query)

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

    qs = simple_branching_strategy(sorted_sols, 10, "problem_code")

    assert len(qs) == 10, "Length of queries should be 10"

    for q in qs:
        print(q)

if __name__ == "__main__":
    test_simple_branching_strategy()
