from copy import deepcopy

import numpy as np
import pytest

from src.model.sudoku_annealing import SimulatedAnnealingSudokuGrid
from src.model.sudoku_base import SudokuGrid, CellCoordinates


@pytest.fixture
def ann_grid(sudoku_grid: SudokuGrid) -> SimulatedAnnealingSudokuGrid:
    return SimulatedAnnealingSudokuGrid.from_sudoku_grid(grid=sudoku_grid)


def test_row_uniqueness_on_init(ann_grid: SimulatedAnnealingSudokuGrid):
    for i in range(0, 8+1):
        row = ann_grid.get_row(i)
        elements, counts = np.unique(row, return_counts=True)

        duplicates = elements[counts > 1]
        assert len(duplicates) == 0, "Each row should be initialized such that 1-9 numbers occur exactly once"


def test_same_row_swap(ann_grid: SimulatedAnnealingSudokuGrid):
    pre_swap_grid = deepcopy(ann_grid)
    cell_1, cell_2 = ann_grid.swap_two_cells_same_row()

    # Verify that cells belong to same row and that swap occurred correctly
    assert cell_1.row == cell_2.row
    assert pre_swap_grid.get_value(cell_1) == ann_grid.get_value(cell_2)
    assert pre_swap_grid.get_value(cell_2) == ann_grid.get_value(cell_1)


def test_score_equals_to_col_and_square_duplicates(ann_grid: SimulatedAnnealingSudokuGrid):
    tot_score = 0
    for i in range(0, 8 + 1):
        elems, counts = np.unique(ann_grid.get_column(i), return_counts=True)
        duplicates, dup_counts = counts[counts > 1], counts[counts > 1]
        col_score = np.sum(dup_counts) if len(dup_counts) > 0 else 0

        tot_score += col_score

    for i in range(0, 2 + 1):
        for j in range(0, 2 + 1):
            top_left_cell = CellCoordinates(row=i * 3, col=j * 3)
            elems, counts = np.unique(ann_grid.get_square(top_left_cell), return_counts=True)
            duplicates, dup_counts = counts[counts > 1], counts[counts > 1]
            square_score = np.sum(dup_counts) if len(dup_counts) > 0 else 0

            tot_score += square_score

    assert tot_score == ann_grid.get_score(), "Score is not equal to number of duplicates in columns + squares"
