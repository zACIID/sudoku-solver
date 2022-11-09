import numpy as np
import pytest

from src.sudoku import SudokuGrid, ConstraintPropagationSudokuGrid, CellCoordinates, NotInDomainException


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

    with pytest.raises(NotInDomainException):
        # This should raise an exception because the same row cannot contain
        #   the same value more than once
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
def test_empty_cell_and_domain_expansion(cell: CellCoordinates, filler: int):
    # Arrange
    value_before_empty = 9
    test_grid = np.full(shape=(9, 9), fill_value=filler)
    test_grid[cell.row, cell.col] = value_before_empty
    sudoku_grid = ConstraintPropagationSudokuGrid(
        starting_grid=test_grid,
        empty_cell_marker=filler
    )

    # Act
    sudoku_grid.empty_cell(cell=cell)

    # Assert
    inner_grid = sudoku_grid.get_inner_grid_copy()
    assert inner_grid[cell.row, cell.col] == filler

    # This should not raise an exception because the same row cannot contain
    #   the same value more than once
    same_row = CellCoordinates(row=cell.row, col=(cell.col + 1) % 9)
    sudoku_grid.set_value(cell=same_row, val=value_before_empty)
    assert True


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
def test_get_minimum_domain_cell(min_domain_cell: CellCoordinates, solved_sudoku: SudokuGrid):
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
    actual_min_domain_cell, domain = cp_grid.get_minimum_domain_cell()

    # Assert
    # Check that cell is the expected one and that
    #   the domain contains only one element: the value prior to emptying the cell
    assert min_domain_cell == actual_min_domain_cell
    assert len(domain) == 1 and prev_value in domain
