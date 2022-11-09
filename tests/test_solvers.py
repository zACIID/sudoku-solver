import pytest

import src.bt_cp_solver as bt
from src.sudoku import SudokuGrid, ConstraintPropagationSudokuGrid


@pytest.mark.bt_cp
def test_bt_solver(sudoku_grid: SudokuGrid):
    cp_grid = ConstraintPropagationSudokuGrid.from_sudoku_grid(grid=sudoku_grid)

    bt.bt_cp_sudoku_solver(sudoku=cp_grid)
