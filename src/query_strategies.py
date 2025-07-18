import src.prompt_constructor_new as prompt
import src.util.query_util as query_util
from dataclasses import dataclass

@dataclass
class StrategyQuery:
    query: str
    branch: int

# the simple branching strategy requires at least num_branches correct solutions, and it
# returns queries which focus on one of the top num_branches solutions (in terms of runtime)
# splitting work evenly between the num_branches solutions
# - returns a list of prompts
def simple_branching_strategy(sorted_solutions, num_samples, problem_code, num_branches=3, phase=0) -> list[StrategyQuery]:
    # TODO: should we create a next round based on just 1-2 solutions, if there are only that many?
    if len(sorted_solutions) < num_branches:
        return []

    bin_sizes = query_util.get_bin_sizes(num_samples, num_branches)

    queries = []

    for sol_idx, bin_size in enumerate(bin_sizes):
        sol = sorted_solutions[sol_idx]
        code = sol["code"]
        runtime = sol["runtime"]

        for _ in range(bin_size):
            query_str = prompt.prompt_improve_solution(problem_code, code)
            strat_query = StrategyQuery(query=query_str, branch=sol_idx)
            queries.append(strat_query)

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

if __name__ == "__main__":
    test_simple_branching_strategy()

    print("✅ All tests passed")
