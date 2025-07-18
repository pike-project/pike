import os
from .utils import read_file

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")

BRAINSTORMING_PROBLEM_STATEMENT = """You are a brainstorming agent, in charge of generating a list of tasks which will be assigned to performance engineers. The performance engineers write custom CUDA kernels to replace pytorch operators in the given architecture to get speedups.
Your objective is to generate a list of diverse tasks which the performance engineers can work on in parallel.
You have complete freedom to choose the set of operators that may be replaced in the given architecture. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may tell performance engineers to replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax).\n
"""

PROBLEM_STATEMENT = """You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups.\n
You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax).\n
"""

INIT_INSTRUCTION = """
Optimize the architecture named Model with custom CUDA operators. Name your optimized output architecture ModelNew. Output the new code in codeblocks.
Generate real code, NOT pseudocode. Just output the new model code, no other text.
Try to make changes which will significantly improve performance over the given architecture.
You should try to make sure the code compiles and is fully functional, but we will attempt to fix errors and correctness issues if it does not.
"""

IMPROVE_INSTRUCTION = """
Try to make changes which will significantly improve performance by modifying your previous solution.
Generate real code, NOT pseudocode. Just output the new model code, no other text.
You should try to make sure the code compiles and is fully functional, but we will attempt to fix errors and correctness issues if it does not.
"""

BRAINSTORMING_INSTRUCTION = """
Create a list of ideas for tasks that performance engineers should work on to optimize the given architecture.
Just output the list of tasks, no other text. A maximum of 10 ideas are allowed, so prioritize the best ideas first.
Fewer ideas are also acceptable if there are not many unique tasks to work on.
"""

def _full_problem_instruction_with_example(
    arch_src: str, example_arch_src: str, example_new_arch_src: str
) -> str:
    prompt = ""

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
Here's an example to show you the syntax of inline embedding custom CUDA operators in torch. The example given architecture is:\n
```python
{example_arch_src}
```\n
The new architecture for the previous example with custom CUDA kernels looks like this:
```python
{example_new_arch_src}
```\n
"""

    prompt += f"""
You are given the following architecture:\n
```python
{arch_src}
```
"""
    return prompt

def get_example_arch():
    example_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_ex_add.py"
    )
    example_new_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_new_ex_add.py"
    )

    if not os.path.exists(example_arch_path):
        raise FileNotFoundError(
            f"Example architecture file not found: {example_arch_path}"
        )
    if not os.path.exists(example_new_arch_path):
        raise FileNotFoundError(
            f"Example new architecture file not found: {example_new_arch_path}"
        )

    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)

    return example_arch, example_new_arch

def brainstorming_problem_instruction(arch_src: str) -> str:
    example_arch_src, _ = get_example_arch()

    prompt = ""

    if example_arch_src != "":
        prompt += f"""
Here's an example to show you how to output ideas for separate optimization tasks. The example given architecture is:\n
```python
{example_arch_src}
```\n
The list of example optimization ideas is:
```
- split the computation into thread blocks to maximize parallelism
- coalesce loads from global memory
- avoid shared memory bank conflicts, if shared memory is being used
- attempt to apply tiling techniques
```\n
"""

    prompt += f"""
You are given the following architecture:\n
```python
{arch_src}
```
"""
    return prompt


def full_problem_instruction(ref_arch_src: str) -> str:
    """
    Using prompt example (an element-wise addition) for prompt templates
    The most basic form of example just to show LLM the task and the expected output format
    """
    arch = ref_arch_src
    # These are strictly defined for now

    # path to prompt template, show an example of Model (torch specifications) and ModelNew (torch + custom CUDA kernels)
    example_arch, example_new_arch = get_example_arch()

    return _full_problem_instruction_with_example(arch, example_arch, example_new_arch)

def prompt_generate_custom_cuda_from_prompt_template(ref_arch_src: str, idea: str) -> str:
    """
    Using prompt example (an element-wise addition) for prompt templates
    The most basic form of example just to show LLM the task and the expected output format
    """
    prompt = PROBLEM_STATEMENT
    prompt += full_problem_instruction(ref_arch_src)

    prompt += INIT_INSTRUCTION

    prompt += f"\nFocus specifically on the following optimization task: {idea}"

    return prompt

def prompt_summarize_error(custom_cuda, stdout, stderr):
    prompt = f"""
The following code failed to compile:
```python
{custom_cuda}
```
Here's the stdout:
```
{stdout}
```
Here's the stderr:
```
{stderr}
```
    
Give a concise description of the error, indicating exactly what the issue is to someone who has not seen the stdout/stderr output.
You do not need to explain how to fix the issue, you must only relay the information that is relevant to solving the issue
"""
    return prompt

# max_stdio_chars are needed to ensure we do not spend excessively on the input to the LLM
# if the stdout/stderr exceeds this max, we only keep the ending portion of it
def prompt_fix_compile_stdout_stderr(ref_arch_src, custom_cuda, results, max_stdio_chars=10000):
    stdout = results["stdout"]
    stderr = results["stderr"]

    stdout_trimmed = ""
    stderr_trimmed = ""

    # trimmed stdout and stderr
    if stdout is not None:
        stdout_trimmed = stdout[-max_stdio_chars:]
    
    if stderr is not None:
        stderr_trimmed = stderr[-max_stdio_chars:]

    timed_out = results["timed_out"]

    prompt = PROBLEM_STATEMENT
    prompt += f"""
With the following original architecture:
```python
{ref_arch_src}
```

You generated the following solution and it failed to compile or timed out:
```python
{custom_cuda}
```

End of stdout:
```
{stdout_trimmed}
```

End of stderr:
```
{stderr_trimmed}
```

Timed out: {timed_out}
    
Fix the compilation error in the new model code. Output the corrected code in codeblocks.
Just output the new model code, no other text.
"""
    return prompt

def prompt_fix_correctness(ref_arch_src, custom_cuda, max_diff):
    prompt = PROBLEM_STATEMENT
    prompt += full_problem_instruction(ref_arch_src)
    prompt += f"""
You generated the following solution architecture previously:
```python
{custom_cuda}
```

It compiled and ran, but failed to pass correctness checks. Your code exceeded the error tolerance from the ground-truth result.

The max value diffs for each of the outputs are: {max_diff}

Fix the correctness issue in the new model code. Output the corrected code in codeblocks.
Just output the new model code, no other text.
"""
    return prompt

def prompt_improve_solution(ref_arch_src, custom_cuda):
    prompt = PROBLEM_STATEMENT
    prompt += full_problem_instruction(ref_arch_src)
    prompt += f"""
You generated the following working solution previously:
```python
{custom_cuda}
```
"""
    
    prompt += IMPROVE_INSTRUCTION

    return prompt

def prompt_generate_ideas(ref_arch_src, custom_cuda=None):
    prompt = BRAINSTORMING_PROBLEM_STATEMENT
    prompt += brainstorming_problem_instruction(ref_arch_src)
#     prompt += f"""
# You generated the following working solution previously:
# ```python
# {custom_cuda}
# ```
# """
    
    prompt += BRAINSTORMING_INSTRUCTION

    return prompt
