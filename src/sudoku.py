from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EMPTY_CELL_VALUE = -777


class SudokuGrid:
    grid: np.ndarray

    def __init__(self, starting_grid: np.ndarray = None, empty_cell_value: int = EMPTY_CELL_VALUE):
        """
        :param starting_grid: starting sudoku grid.
            Empty cells must be marked with the provided value.
        :param empty_cell_value: value that empty cells are marked with. Must be a number
            out of the 1 to 9 integer interval (which is that of valid sudoku numbers)
        """

        assert starting_grid.shape[0] == 9 and starting_grid.shape[1] == 9, "Provided sudoku grid is not 9x9"

        self.grid = np.full((9, 9), empty_cell_value) if starting_grid is None else starting_grid

    def get_value(self, cell: CellCoordinates) -> int:
        return self.grid[cell.row, cell.col]

    def set_value(self, cell: CellCoordinates, val: int):
        assert val in [1, 2, 3, 4, 5, 6, 7, 8, 9, EMPTY_CELL_VALUE], "Provided value is not valid"

        self.grid[cell.row, cell.col] = val

    def get_row(self, i: int) -> np.ndarray:
        return self.grid[i, :]

    def get_column(self, i: int) -> np.ndarray:
        return self.grid[:, i]

    def get_square(self, starting_row: int, starting_col: int) -> np.ndarray:
        return self.grid[starting_row:starting_row+3, starting_col:starting_col+3]

    def is_full(self) -> bool:
        # Full when there are no more empty cells
        return EMPTY_CELL_VALUE not in self.grid


@dataclass(frozen=True)
class CellCoordinates:  # TODO cell coordinates?
    row: int
    col: int

    def get_next(self) -> CellCoordinates:
        """
        Returns a SudokuCell instance that represents the next cell, relative to the current one

        :return: next cell instance
        """

        # Here next is determined as right-adjacent with respect to columns
        # If the current cell is on the last column (col == 8), then move to
        #   the first cell (col == 0) of the next row

        next_col = (self.col + 1) % 9

        is_next_row = self.col + 1 == 9
        next_row = self.row + 1 if is_next_row else self.row

        return CellCoordinates(
            row=next_row, col=next_col
        )
