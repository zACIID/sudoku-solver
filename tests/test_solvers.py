import pytest

import src.solvers.bt_cp as bt
from src.model.sudoku_base import SudokuGrid
from src.model.sudoku_cp import ConstraintPropagationSudokuGrid


@pytest.mark.bt_cp
def test_bt_solver(sudoku_grid: SudokuGrid):
    cp_grid = ConstraintPropagationSudokuGrid.from_sudoku_grid(grid=sudoku_grid)

    bt.bt_cp_sudoku_solver(sudoku=cp_grid)
