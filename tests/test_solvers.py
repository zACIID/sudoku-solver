import pytest

import src.solvers.bt_cp as bt
import src.solvers.annealing as ann
from src.model.sudoku_base import SudokuGrid
from src.model.sudoku_cp import ConstraintPropagationSudokuGrid
from src.model.sudoku_annealing import SimulatedAnnealingSudokuGrid
from src.solvers.utils import is_solution_correct


@pytest.mark.bt_cp
def test_bt_solver(sudoku_grid: SudokuGrid):
    cp_grid = ConstraintPropagationSudokuGrid.from_sudoku_grid(grid=sudoku_grid)

    solution = bt.bt_cp_sudoku_solver(sudoku=cp_grid)
    assert is_solution_correct(solution), "Solution should be correct"


@pytest.mark.annealing
def test_bt_solver(sudoku_grid: SudokuGrid):
    ann_grid = SimulatedAnnealingSudokuGrid.from_sudoku_grid(grid=sudoku_grid)

    solution = ann.simulated_annealing_solver(sudoku=ann_grid)
    assert is_solution_correct(solution), "Solution should be correct"
