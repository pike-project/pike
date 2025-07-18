import numpy as np
import src.prompt_constructor_new as prompt

# bin_count should be the minimum of some value (e.g. 4) and the number of solutions
# that we have available
def get_bin_sizes(num_samples: int, bin_count: int) -> list[int]:
    """
    Takes as input a target number of samples, along with the number of "bins" we want to create
    to divide up the number of samples

    Returns the size of each bin such that num_samples can be divided approximately evenly into bin_count

    Example:
    -> num_samples = 10, bin_count = 3, returns [4, 3, 3]
    """
    sample_idxs = np.array(list(range(num_samples)))
    bins_np = np.array_split(sample_idxs, bin_count)
    bin_sizes = [len(b) for b in bins_np]

    return bin_sizes

# the simple branching strategy requires at least 3 correct solutions, and it
# returns queries which focus on one of the top 3 solutions (in terms of runtime)
# splitting work evenly between the 3 solutions
# - returns a list of prompts
def simple_branching_strategy(sorted_solutions, num_samples, problem_code, phase=0) -> list[str]:
    # TODO: should we create a next round based on just 1-2 solutions, if there are only that many?
    if len(sorted_solutions) < 3:
        return []

    bin_sizes = get_bin_sizes(num_samples, 3)

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
            "runtime": 0.1,
            "phase": 0,
            "branch": 0
        },
        {
            "code": "code2",
            "runtime": 0.2,
            "phase": 1,
            "branch": 0
        },
        {
            "code": "code3",
            "runtime": 0.3,
            "phase": 1,
            "branch": 1
        },
    ]

    qs = simple_branching_strategy(sorted_sols, 10, "problem_code", phase=2)

    assert len(qs) == 10, "Length of queries should be 10"

    # for q in qs:
    #     print(q)

    print("✅ All tests passed")

if __name__ == "__main__":
    test_simple_branching_strategy()
