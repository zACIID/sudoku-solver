import pytest

import src.bt_solver as bt
from src.sudoku import SudokuGrid


@pytest.mark.bt_cp
def test_bt_solver(sudoku_grid: SudokuGrid):
    bt.sudoku_solver(sudoku=sudoku_grid)
