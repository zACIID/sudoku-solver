import numpy as np
import pytest

import src.constraints as cn
from src.sudoku import SudokuGrid


def test_is_valid_solution(solved_sudoku: SudokuGrid):
    assert cn.is_solution_correct(solution=solved_sudoku)


def test_is_valid_row_column_square(valid_sudoku_row_column_square: np.ndarray):
    assert cn.check_no_repetitions(valid_sudoku_row_column_square)


def test_not_valid_row_column_square(not_valid_sudoku_row_column_square: np.ndarray):
    assert not cn.check_no_repetitions(not_valid_sudoku_row_column_square)
