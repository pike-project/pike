import os

def strip_whitespace_and_comments(code: str) -> str:
    lines = code.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # skip blank or comment-only lines
        if not stripped or stripped.startswith("#"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def mean_loc_in_directory(directory, extensions=None):
    """
    Calculate the mean Lines of Code (LoC) across all files in a directory,
    ignoring blank lines and comment-only lines.
    
    Args:
        directory (str): Path to the directory to scan.
        extensions (list[str], optional): List of file extensions to include (e.g., ['.py', '.js']).
    
    Returns:
        float: Mean LoC value, or 0 if no files matched.
    """
    loc_counts = []
    
    for root, _, files in os.walk(directory):
        for filename in files:
            if extensions and not any(filename.endswith(ext) for ext in extensions):
                continue
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                    cleaned_code = strip_whitespace_and_comments(code)
                    loc = len(cleaned_code.splitlines())
                    loc_counts.append(loc)
            except Exception as e:
                print(f"Could not read {filepath}: {e}")

    if not loc_counts:
        return 0.0
    
    return sum(loc_counts) / len(loc_counts)


# Example usage:
# directory_path = "./my_project"
# directory_path = "KernelBench/level3-pike"
directory_path = "KernelBench/level5"
mean_loc = mean_loc_in_directory(directory_path, extensions=['.py'])
print(f"Mean LoC (excluding blanks/comments): {mean_loc:.2f}")
