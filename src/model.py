from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

EMPTY_CELL_VALUE = -777


class SudokuGrid:
    grid: np.ndarray

    def __init__(self, starting_grid: np.ndarray = None):
        """"""

        self.grid = np.full((9, 9), EMPTY_CELL_VALUE) if starting_grid is None else starting_grid

    def get_value(self, cell: SudokuCell) -> int:
        return self.grid[cell.row, cell.col]

    def set_value(self, cell: SudokuCell, val: int):
        assert val in [1, 2, 3, 4, 5, 6, 7, 8, 9, EMPTY_CELL_VALUE], "Provided value is not valid"

        self.grid[cell.row, cell.col] = val

    def get_row(self, i: int) -> np.ndarray:
        return self.grid[i, :]

    def get_column(self, i: int) -> np.ndarray:
        return self.grid[:, i]

    def get_square(self, starting_row: int, starting_col: int) -> np.ndarray:
        return self.grid[starting_row:starting_row+3, starting_col:starting_col+3]

    def is_valid(self) -> bool:


@dataclass(frozen=True)
class SudokuCell:
    row: int
    col: int

    def get_next(self) -> SudokuCell:
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

        return SudokuCell(
            row=next_row, col=next_col
        )
