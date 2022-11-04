from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EMPTY_CELL_VALUE = -777


class SudokuGrid:
    grid: np.ndarray
    """
    9x9 Sudoku grid
    """

    empty_cell_value: int
    """
    Value used to mark empty cells in the Sudoku grid
    """

    def __init__(self, starting_grid: np.ndarray = None, empty_cell_value: int = EMPTY_CELL_VALUE):
        """
        :param starting_grid: starting sudoku grid.
            Empty cells must be marked with the provided value.
        :param empty_cell_value: value that empty cells are marked with. Must be a number
            out of the 1 to 9 integer interval (which is that of valid sudoku numbers)
        """

        assert starting_grid.shape[0] == 9 and starting_grid.shape[1] == 9, "Provided sudoku grid is not 9x9"
        assert empty_cell_value not in [1, 2, 3, 4, 5, 6, 7, 8, 9], \
            "Empty cells cannot be marked with integers from 1 to 9"

        self.grid = np.full((9, 9), empty_cell_value) if starting_grid is None else starting_grid

    def get_value(self, cell: CellCoordinates) -> int:
        return self.grid[cell.row, cell.col]

    def set_value(self, cell: CellCoordinates, val: int, only_if_empty: bool = True):
        """
        Sets the value of the specified cell. There is an option to specify that only empty cells
        can be set, causing an exception to be raised otherwise (default behavior).

        :param cell: coordinates of the cell to set
        :param val: value to set to the cell
        :param only_if_empty: if True (default), only empty cells can be set,
            otherwise an exception is raised
        """

        assert val in [1, 2, 3, 4, 5, 6, 7, 8, 9, self.empty_cell_value], "Provided value is not valid"

        if only_if_empty:
            if self.grid[cell.row, cell.col] != self.empty_cell_value:
                raise ValueError(f"Cannot set a value to a non-empty cell: {cell}")

        self.grid[cell.row, cell.col] = val

    def get_row(self, i: int) -> np.ndarray:
        return self.grid[i, :]

    def get_column(self, i: int) -> np.ndarray:
        return self.grid[:, i]

    def get_square(self, starting_row: int, starting_col: int) -> np.ndarray:
        return self.grid[starting_row:starting_row + 3, starting_col:starting_col + 3]

    def is_full(self) -> bool:
        # Full when there are no more empty cells
        return self.empty_cell_value not in self.grid


@dataclass(frozen=True)
class CellCoordinates:
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

    def __str__(self):
        return f"(row: {self.row}, col: {self.col})"