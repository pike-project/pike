import numpy as np

# bin_count should be the minimum of some value (e.g. 4) and the number of solutions
# that we have available
def get_bin_sizes(target_sample_count: int, bin_count: int) -> list[int]:
    """
    Takes as input a target number of samples, along with the number of "bins" we want to create
    to divide up the number of samples

    Returns the size of each bin such that num_samples can be divided approximately evenly into bin_count

    Example:
    -> num_samples = 10, bin_count = 3, returns [4, 3, 3]
    """
    sample_idxs = np.array(list(range(target_sample_count)))
    bins_np = np.array_split(sample_idxs, bin_count)
    bin_sizes = [len(b) for b in bins_np]

    return bin_sizes

def resize_list(l: list, size: int) -> list:
    """
    Resizes a list to a desired size by duplicating certain elements if the list is not long enough
    and trimming the list if it is too long

    Returns a new list
    """

    if len(l) == 0:
        raise Exception("List has no elements, no way to resize")

    # trim the list if it is too long
    if len(l) >= size:
        return l[:size]

    bin_sizes = get_bin_sizes(size, len(l))

    res = []

    for sol_idx, bin_size in enumerate(bin_sizes):
        v = l[sol_idx]

        for _ in range(bin_size):
            res.append(v)

    return res

def test_get_bin_sizes():
    assert get_bin_sizes(10, 3) == [4, 3, 3], "10 samples, bin count 3"
    assert get_bin_sizes(10, 4) == [3, 3, 2, 2], "10 samples, bin count 4"

def test_resize_list():
    assert resize_list([0, 1, 2], 10) == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2], "10 samples, bin count 3"
    assert resize_list([0, 1, 2, 3], 10) == [0, 0, 0, 1, 1, 1, 2, 2, 3, 3], "10 samples, bin count 4"
    assert resize_list([0, 1, 2, 3, 4, 5, 6], 5) == [0, 1, 2, 3, 4], "trim list"
    assert resize_list([0, 1, 2, 3, 4], 5) == [0, 1, 2, 3, 4], "same size"

    assert resize_list([0, 1, 2, 3, 4, 5, 6], 10) == [0, 0, 1, 1, 2, 2, 3, 4, 5, 6], "7 samples, bin count 10"

if __name__ == "__main__":
    test_get_bin_sizes()
    test_resize_list()

    print("✅ All tests passed")
