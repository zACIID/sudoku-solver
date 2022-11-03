import numpy as np


def check_no_repetitions(array: np.ndarray) -> bool:
    """
    Checks the provided array for no repetitions.
    The array should have cardinality equal to 9, and should contain only numbers from 1 to 9.
    Returns True if all the numbers are present exactly once.

    :param array: array with exactly 9 elements
    :return: True if the array contains all the numbers from 1 to 9 exactly once
    """

    # This function checks collections of exactly 9 elements,
    # which could be 3x3 squares, 1x9 rows or 1x9 columns
    # Cardinality MUST be 9
    assert array.size == 9, "The provided array has to be exactly of length 9"

    # Since the array is of size 9, np.all evaluates to True iff
    # each number 1 to 9 is contained exactly once
    # TODO this works to check if a complete sudoku grid is valid, but it doesn't work
    #   for partial rows/columns/squares
    return np.all(np.isin(array, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
