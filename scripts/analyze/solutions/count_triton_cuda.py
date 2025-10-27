import os

def count_files_with_any_string(directory, match_strings):
    """
    Count how many files in a directory (recursively) contain
    ANY of the strings in `match_strings`.

    Parameters:
        directory (str): Path to the target directory.
        match_strings (list[str]): List of strings to search for.

    Returns:
        int: Number of files that contain at least one match.
    """
    count = 0

    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if any(s in content for s in match_strings):
                        count += 1
            except Exception as e:
                print(f"Skipping {filepath} (error: {e})")

    return count


# Example usage:
if __name__ == "__main__":
    # target_dir = "best_agent_solutions_new/h100/h100_level3-metr/prev_agents/best_solutions"
    target_dir = "best_agent_solutions_new/h100/h100_level3-metr/openevolve_agents/best_solutions"
    # search_terms = ["@triton"]
    search_terms = ["torch/extension.h", "cuda_runtime.h", "__syncthreads"]
    num_files = count_files_with_any_string(target_dir, search_terms)
    print(f"Number of files containing any of {search_terms}: {num_files}")
