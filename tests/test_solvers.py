import pytest
from loguru import logger

import src.solvers.bt_cp as bt
import src.solvers.annealing as ann
from src.model.sudoku_base import SudokuGrid
from src.solvers.utils import is_solution_correct


@pytest.mark.bt_cp
def test_bt_solver(sudoku_grid: SudokuGrid):
    solution = bt.bt_cp_sudoku_solver(sudoku=sudoku_grid)

    assert_solution(solution)


@pytest.mark.annealing
def test_annealing_solver(sudoku_grid: SudokuGrid):
    solution = ann.simulated_annealing_solver(sudoku=sudoku_grid)

    assert_solution(solution)


def assert_solution(solution: SudokuGrid):
    logger.debug("Solution to assert:\n"
                 f"{solution}")

    assert is_solution_correct(solution), "Solution should be correct"
