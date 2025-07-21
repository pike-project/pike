import src.prompt_constructor_new as prompt
import src.util.query_util as query_util
from dataclasses import dataclass
import random
from typing import List


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


def elite_diverse_branching_strategy(
        sorted_solutions: List[dict],
        num_samples: int,
        problem_code: str,
        *,
        num_branches: int = 6,
        elite_ratio: float = 0.3,      # 30 % elite, 70 % diverse
        phase: int = 0
        ) -> list[StrategyQuery]:
    
    """
    Evolutionary style selection:
    - First take the top `elite_ratio` * num_branches solutions (best runtime)
    - Fill the remaining branches by randomly sampling from the rest
    """

    if len(sorted_solutions) < 1 or num_branches < 1:
        return []

    # --- 1) determine how many elite vs. diverse branches are needed
    elite_count = max(1, int(num_branches * elite_ratio))
    elite_count = min(elite_count, len(sorted_solutions))   

    diverse_count = num_branches - elite_count
    diverse_pool  = sorted_solutions[elite_count:]          # everything non-elite

    # --- 2) select the branches
    parents: List[dict] = []

    #   a) deterministic elite
    parents.extend(sorted_solutions[:elite_count])

    #   b) random diverse
    if diverse_count > 0 and diverse_pool:
        diverse_pick = random.sample(
            diverse_pool,
            k=min(diverse_count, len(diverse_pool))
        )
        parents.extend(diverse_pick)

    #   (If we still do not have enough parents – e.g. pool too small –
    #    fall back to simply repeating elites)
    while len(parents) < num_branches:
        parents.append(random.choice(sorted_solutions))

    # --- 3) distribute queries among the selected parents
    bin_sizes = query_util.get_bin_sizes(num_samples, len(parents))
    queries   = []

    for sol_idx, (sol, bin_size) in enumerate(zip(parents, bin_sizes)):
        for _ in range(bin_size):
            q = prompt.prompt_improve_solution(problem_code, sol["code"])
            queries.append(StrategyQuery(query=q, branch=sol_idx))

    return queries

def tournament_branching_strategy(
        sorted_solutions: List[dict],
        num_samples: int,
        problem_code: str,
        *,
        num_branches: int = 4,
        k: int = 3,         
        phase: int = 0
        ) -> list[StrategyQuery]:
    """
    k-way tournament selection: repeatedly pick k random candidates,
    best runtime wins the tournament, becomes one parent branch.
    """

    if len(sorted_solutions) < 1:
        return []

    parents: List[dict] = []

    while len(parents) < num_branches:
        # draw without replacement *within* the current tournament
        contenders = random.sample(sorted_solutions,
                                   k=min(k, len(sorted_solutions)))
        # best runtime (= min) wins
        winner = min(contenders, key=lambda s: s["runtime"])
        parents.append(winner)

    bin_sizes = query_util.get_bin_sizes(num_samples, len(parents))
    queries   = []

    for sol_idx, (sol, bin_size) in enumerate(zip(parents, bin_sizes)):
        for _ in range(bin_size):
            q = prompt.prompt_improve_solution(problem_code, sol["code"])
            queries.append(StrategyQuery(query=q, branch=sol_idx))

    return queries

def roulette_branching_strategy(
        sorted_solutions: List[dict],
        num_samples: int,
        problem_code: str,
        *,
        num_branches: int = 5,
        epsilon: float = 1e-8,   # to avoid div-by-zero for perfect runtime
        phase: int = 0
) -> list[StrategyQuery]:
    """
    Fitness-proportional (roulette-wheel) selection.
    Fitness = 1 / (runtime + ε)
    """

    if len(sorted_solutions) < 1:
        return []

    # compute fitness values
    fitness = [1.0 / (s["runtime"] + epsilon) for s in sorted_solutions]
    total_f = sum(fitness)
    if total_f == 0:                       # fallback: uniform probabilities
        probs = [1 / len(fitness)] * len(fitness)
    else:
        probs = [f / total_f for f in fitness]

    # choose parents WITH replacement → duplicates allowed
    parents_indices = random.choices(
        population=list(range(len(sorted_solutions))),
        weights=probs,
        k=num_branches
    )
    parents = [sorted_solutions[i] for i in parents_indices]

    bin_sizes = query_util.get_bin_sizes(num_samples, len(parents))
    queries   = []

    for sol_idx, (sol, bin_size) in enumerate(zip(parents, bin_sizes)):
        for _ in range(bin_size):
            q = prompt.prompt_improve_solution(problem_code, sol["code"])
            queries.append(StrategyQuery(query=q, branch=sol_idx))

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
