import numpy as np

import sudoku as sud


def is_solution_correct(solution: sud.SudokuGrid) -> bool:
    """
    Checks if a sudoku solution is correct, which means that the provided
    sudoku grid must be both full and valid.

    :param solution: complete sudoku grid
    :return: True if the solution is correct
    """

    return solution.is_full() and is_grid_valid(solution)


def is_grid_valid(grid: sud.SudokuGrid) -> bool:
    """
    Checks that each column, row and 3x3 square don't contain duplicates
    of numbers from 1 to 9. Returns True if no repetitions.

    :param grid: sudoku grid to check
    :return: True if no repetitions found
    """

    valid = True

    # Check Rows and Columns
    for i in range(0, 8+1):
        valid &= check_no_repetitions(grid.get_row(i))
        valid &= check_no_repetitions(grid.get_column(i))

        if not valid:
            return False

    # Check all the nine 3x3 squares
    for i in range(0, 8+1, 3):
        for j in range(0, 8+1, 3):
            if not check_no_repetitions(grid.get_square(
                cell=sud.CellCoordinates(row=i, col=j)
            )):
                return False

    return True


def check_no_repetitions(array: np.ndarray) -> bool:
    """
    Checks the provided array for no repetitions in the numbers between 1 and 9.
    This means that both an incomplete or complete sudoku column/row/box can be valid,
    iff there are no repetitions except for the values used for empty cells, which
    certainly do not belong in the set of integers from 1 to 9.

    :param array: array with exactly 9 elements that represents either
        a sudoku column, row or 3x3 box
    :return: True if any number between 1 and 9 is contained exactly once
    """

    assert array.size == 9, "The provided array has to be exactly of length 9"

    # Important is to check that duplicate elements are not 1 to 9 numbers
    # Remember, in fact, that there could also be duplicate filler
    #   values for empty cells
    unique, counts = np.unique(array, return_counts=True)
    duplicates = unique[counts > 1]

    return not np.any(np.isin(duplicates, range(1, 9+1)))
