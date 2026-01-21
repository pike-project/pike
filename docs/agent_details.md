# Agent Details

More details about agent operation can be found here.

## Prompt Construction

Prompt construction is done in `src/prompt_construction.py`

There are a few key prompts that are used by the agent:

- ideas prompt: prompts an LLM to brainstorm tasks that should be assigned to the performance engineering agents
- phase 0 performance optimization prompt: gives a performance engineering agent an example of what a model may look like after applying optimizations, then prompts the agent to optimize a specific pytorch model, seeding with an idea from the previous brainstorming agent
- phase 1-n performance optimization prompt: similar to the previous, but instead of giving a brainstorming idea, it gives a previous, well-performing solution, and prompts the agent to improve upon this previous solution
- error fixing prompt: tells the error fixing agent to fix a compilation/correctness issue in the model code that was produced by a previous performance optimization agent

## Query Strategies

Query strategies to drive the agent framework search are implemented in: `src/query_strategies.py`

TODO: more notes about query strategies
