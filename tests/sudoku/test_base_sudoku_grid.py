import numpy as np
import pytest

from src.sudoku import SudokuGrid, CellCoordinates


@pytest.mark.parametrize("cell, value", [
    (CellCoordinates(row=1, col=1), 8),
    (CellCoordinates(row=3, col=8), 1),

    # This test should fail because rows/cols are out of bounds
    pytest.param(CellCoordinates(row=12, col=12), -1, marks=pytest.mark.xfail)
])
@pytest.mark.base_grid
def test_get_value(cell: CellCoordinates, value: int):
    # Arrange
    filler_value = -1
    test_grid = np.full(shape=(9, 9), fill_value=filler_value)

    test_grid[cell.row, cell.col] = value

    sudoku_grid = SudokuGrid(starting_grid=test_grid, empty_cell_marker=filler_value)

    # Act
    actual_value = sudoku_grid.get_value(cell=cell)

    # Assert
    assert actual_value == value


@pytest.mark.parametrize("cell, value", [
    (CellCoordinates(row=1, col=1), 8),
    (CellCoordinates(row=3, col=8), 1),

    # This test should fail because rows/cols are out of bounds
    pytest.param(CellCoordinates(row=12, col=12), 5, marks=pytest.mark.xfail),

    # This should fail because value to set is not valid
    pytest.param(CellCoordinates(row=12, col=12), 154, marks=pytest.mark.xfail),
])
@pytest.mark.base_grid
def test_set_value(cell: CellCoordinates, value: int):
    # Arrange
    filler_value = -1
    test_grid = np.full(shape=(9, 9), fill_value=filler_value)
    sudoku_grid = SudokuGrid(starting_grid=test_grid, empty_cell_marker=filler_value)

    # Act
    sudoku_grid.set_value(cell=cell, val=value)

    # Assert
    inner_grid = sudoku_grid.get_inner_grid_copy()
    assert inner_grid[cell.row, cell.col] == value


@pytest.mark.parametrize("cell, filler", [
    (CellCoordinates(row=1, col=1), -1),
    (CellCoordinates(row=3, col=8), -999),

    # This test should fail because rows/cols are out of bounds
    pytest.param(CellCoordinates(row=12, col=12), -777, marks=pytest.mark.xfail),
])
@pytest.mark.base_grid
def test_empty_cell(cell: CellCoordinates, filler: int):
    # Arrange
    test_grid = np.full(shape=(9, 9), fill_value=filler)
    sudoku_grid = SudokuGrid(starting_grid=test_grid, empty_cell_marker=filler)

    # Act
    sudoku_grid.empty_cell(cell=cell)

    # Assert
    inner_grid = sudoku_grid.get_inner_grid_copy()
    assert inner_grid[cell.row, cell.col] == filler
