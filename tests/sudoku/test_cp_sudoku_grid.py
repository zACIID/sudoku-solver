from typing import Set

import numpy as np
import pytest

from src.model.sudoku_base import SudokuGrid, CellCoordinates
from src.model.sudoku_cp import ConstraintPropagationSudokuGrid, NotInDomainException


@pytest.mark.parametrize("cell, value", [
    (
            CellCoordinates(
                row=np.random.randint(low=0, high=8 + 1),
                col=np.random.randint(low=0, high=8 + 1)
            ),
            8
    ),

    # This test should fail because rows/cols are out of bounds
    pytest.param(CellCoordinates(row=12, col=12), -1, marks=pytest.mark.xfail),

    # This should fail because value to set is not valid
    pytest.param(CellCoordinates(row=12, col=12), 154, marks=pytest.mark.xfail)
])
@pytest.mark.cp_grid
def test_set_value_and_domain_reduction(cell: CellCoordinates, value: int):
    # Arrange
    filler_value = -777
    test_grid = np.full(shape=(9, 9), fill_value=filler_value)
    sudoku_grid = ConstraintPropagationSudokuGrid(
        starting_grid=test_grid,
        empty_cell_marker=filler_value
    )

    # Act
    sudoku_grid.set_value(cell=cell, val=value)

    # Assert
    inner_grid = sudoku_grid.get_inner_grid_copy()
    assert inner_grid[cell.row, cell.col] == value

    # The value that was set should not be in the domain of any cell
    #   in the row/col/square of the current cell
    affected_cells = get_affected_cells(cell_to_update=cell)
    for c in affected_cells:
        assert value not in sudoku_grid.get_domain(cell=c)


def get_affected_cells(cell_to_update: CellCoordinates) -> Set[CellCoordinates]:
    affected_cells = set()
    for i in range(0, 8 + 1):
        affected_cells.add(CellCoordinates(row=cell_to_update.row, col=i))
        affected_cells.add(CellCoordinates(row=i, col=cell_to_update.col))

    square_starting_row = (cell_to_update.row // 3) * 3
    square_starting_col = (cell_to_update.col // 3) * 3
    for i in range(square_starting_row, square_starting_row + 3):
        for j in range(square_starting_col, square_starting_col + 3):
            affected_cells.add(CellCoordinates(row=i, col=j))

    return affected_cells


@pytest.mark.cp_grid
def test_set_value_not_in_domain():
    # Arrange
    filler_value = -777
    cell = CellCoordinates(
        row=np.random.randint(low=0, high=8 + 1),
        col=np.random.randint(low=0, high=8 + 1)
    )
    value = 9
    test_grid = np.full(shape=(9, 9), fill_value=filler_value)
    sudoku_grid = ConstraintPropagationSudokuGrid(
        starting_grid=test_grid,
        empty_cell_marker=filler_value
    )

    # Act
    sudoku_grid.set_value(cell=cell, val=value)

    # Assert
    inner_grid = sudoku_grid.get_inner_grid_copy()
    assert inner_grid[cell.row, cell.col] == value

    with pytest.raises(NotInDomainException):
        # This should raise an exception because the same row (or col or square)
        #   cannot contain the same value more than once
        same_row = CellCoordinates(row=cell.row, col=(cell.col + 1) % 9)
        sudoku_grid.set_value(cell=same_row, val=value)


@pytest.mark.parametrize("cell, filler", [
    (
            CellCoordinates(
                row=np.random.randint(low=0, high=8 + 1),
                col=np.random.randint(low=0, high=8 + 1)
            ),
            -777
    ),

    # This test should fail because rows/cols are out of bounds
    pytest.param(CellCoordinates(row=12, col=12), -1, marks=pytest.mark.xfail),

    # This should fail because value to set is not valid (must be != 1 to 9)
    pytest.param(CellCoordinates(row=5, col=5), 7, marks=pytest.mark.xfail)
])
@pytest.mark.cp_grid
def test_empty_cell(cell: CellCoordinates, filler: int, solved_sudoku: SudokuGrid):
    # Arrange
    solved_grid = solved_sudoku.get_inner_grid_copy()
    value_before_empty = solved_grid[cell.row, cell.col]
    solved_grid[cell.row, cell.col] = value_before_empty
    sudoku_grid = ConstraintPropagationSudokuGrid(
        starting_grid=solved_grid,
        empty_cell_marker=filler
    )

    # Act
    sudoku_grid.empty_cell(cell=cell)

    # Assert
    inner_grid = sudoku_grid.get_inner_grid_copy()
    assert inner_grid[cell.row, cell.col] == filler


# TODO test domain expansion on empty_cell: it is not so simple as in checking that affected
#   cells have the previous value added back to their domains, proper checks need to be made
#   because not all cells might be affected. Need to create an ad-hoc grid and test the expected cells


@pytest.mark.cp_grid
def test_domain_initialization(sudoku_grid: SudokuGrid):
    cp_grid = ConstraintPropagationSudokuGrid.from_sudoku_grid(grid=sudoku_grid)

    # Check that the value of each cell isn't in the domain
    #   of the cells in the same row/col/square
    for row in range(0, 8 + 1):
        for col in range(0, 8 + 1):
            current_cell = CellCoordinates(row=row, col=col)
            current_val = cp_grid.get_value(cell=current_cell)

            affected_cells = get_affected_cells(cell_to_update=current_cell)
            for c in affected_cells:
                domain = cp_grid.get_domain(cell=c)
                assert current_val not in domain


@pytest.mark.parametrize("min_domain_cell", [
    CellCoordinates(
        row=np.random.randint(low=0, high=8 + 1),
        col=np.random.randint(low=0, high=8 + 1)
    ),
    CellCoordinates(
        row=np.random.randint(low=0, high=8 + 1),
        col=np.random.randint(low=0, high=8 + 1)
    )
])
@pytest.mark.cp_grid
def test_get_minimum_domain_empty_cell(min_domain_cell: CellCoordinates, solved_sudoku: SudokuGrid):
    # Arrange
    solved_grid = solved_sudoku.get_inner_grid_copy()

    # Since the grid is full, emptying the provided cell means that it should
    #   become the only one with the minimum (non-empty) domain
    prev_value = solved_grid[min_domain_cell.row, min_domain_cell.col]
    solved_grid[min_domain_cell.row, min_domain_cell.col] = solved_sudoku.empty_cell_marker
    cp_grid = ConstraintPropagationSudokuGrid(
        starting_grid=solved_grid,
        empty_cell_marker=solved_sudoku.empty_cell_marker
    )

    # Act
    actual_min_domain_cell, domain = cp_grid.get_minimum_domain_empty_cell()

    # Assert
    # Check that cell is the expected one and that
    #   the domain contains only one element: the value prior to emptying the cell
    assert min_domain_cell == actual_min_domain_cell
    assert len(domain) == 1 and prev_value in domain
