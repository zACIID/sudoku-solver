from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Set, List
from copy import deepcopy, copy

import numpy as np


# TODO refactoring ideas:
#   - Sudoku becomes a folder with base.py and constraint_propagation.py
#   - bt_cp_solver becomes solvers/bt_cp.py
#   - constraints.py becomes solvers/solution_checks.py or solvers/utils.py


@dataclass(frozen=True)
class CellCoordinates:
    row: int
    col: int

    def get_next(self) -> CellCoordinates | None:
        """
        Returns a SudokuCell instance that represents the next cell, relative to the current one.
        If the current instance is the last cell of the grid, returns None.

        :return: next cell instance, None if this is the last cell of the grid.
        """

        # Here next is determined as right-adjacent with respect to columns
        # If the current cell is on the last column (col == 8), then move to
        #   the first cell (col == 0) of the next row

        # Last cell of the grid: return None
        if self.col == 8 and self.row == 8:
            return None

        next_col = (self.col + 1) % 9

        is_next_row = self.col + 1 == 9
        next_row = self.row + 1 if is_next_row else self.row

        return CellCoordinates(
            row=next_row, col=next_col
        )

    def __str__(self):
        return f"(row: {self.row}, col: {self.col})"


class SudokuGrid:
    _inner_grid: np.ndarray
    """
    9x9 Sudoku grid
    """

    empty_cell_marker: int
    """
    Value used to mark empty cells in the Sudoku grid
    """

    def __init__(self, starting_grid: np.ndarray = None, empty_cell_marker: int = -777):
        """
        :param starting_grid: starting sudoku grid.
            Empty cells must be marked with the provided value.
        :param empty_cell_marker: value that empty cells are marked with. Must be a number
            out of the 1 to 9 integer interval (which is that of valid sudoku numbers)
        """

        if not (starting_grid.shape[0] == 9 and starting_grid.shape[1] == 9):
            raise ValueError("Provided sudoku grid is not 9x9")

        if empty_cell_marker in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            raise ValueError("Empty cells cannot be marked with integers from 1 to 9")

        self._inner_grid = np.full((9, 9), empty_cell_marker) if starting_grid is None else starting_grid
        self.empty_cell_marker = empty_cell_marker

    def __str__(self):
        # TODO prettify
        return str(self._inner_grid)

    def get_inner_grid_copy(self) -> np.ndarray:
        """
        Returns a copy (deep) of the underlying sudoku grid representation
        :return: grid deep copy
        """

        return deepcopy(self._inner_grid)

    def get_value(self, cell: CellCoordinates) -> int:
        return self._inner_grid[cell.row, cell.col]

    def set_value(self, cell: CellCoordinates, val: int, overwrite: bool = False):
        """
        Sets the value of the specified cell.
        The value can be a number from 1 to 9, or the empty cell marker.
        There is an option to specify that only empty cells can be set,
        causing an exception to be raised otherwise (default behavior).

        :param cell: coordinates of the cell to set
        :param val: value to set to the cell
        :param overwrite: if False, setting the value of a non-empty cell causes
            an exception to be raised
        """

        if overwrite:
            if self._inner_grid[cell.row, cell.col] != self.empty_cell_marker:
                raise ValueError(f"Cannot set a value to non-empty cell: {cell}")

        if not (1 <= val <= 9):
            raise ValueError(f"The provided value '{val}' is not a valid sudoku number")

        self._inner_grid[cell.row, cell.col] = val

    def empty_cell(self, cell: CellCoordinates):
        """
        Marks the specified cell to empty. The value used to mark it will be the one
        specified during the creation of this instance.

        :param cell: cell to mark es empty
        """

        self._inner_grid[cell.row, cell.col] = self.empty_cell_marker

    def get_row(self, i: int) -> np.ndarray:
        """
        Returns the sudoku row with the provided index
        :param i: index of the row, 0 based
        :return: row at the specified index
        """

        if not (0 <= i <= 8):
            raise ValueError("Index out of range")

        return self._inner_grid[i, :]

    def get_column(self, i: int) -> np.ndarray:
        """
        Returns the sudoku column with the provided index
        :param i: index of the column, 0 based
        :return: column at the specified index
        """

        if not (0 <= i <= 8):
            raise ValueError("Index out of range")

        return self._inner_grid[:, i]

    def get_square(self, cell: CellCoordinates) -> np.ndarray:
        """
        Returns the sudoku square that the specified cell belongs to.
        Such a square refers to the 9 squares characteristic of a sudoku grid,
        in which there can't be repetitions of numbers from 1 to 9.

        :param cell: cell to return the sudoku square of
        :return: sudoku square that the cell belongs to
        """

        # Square starting indexes, for both rows and cols, are either 0, 3 or 6
        starting_row = (cell.row // 3) * 3
        starting_col = (cell.col // 3) * 3

        return self._inner_grid[starting_row:starting_row + 3, starting_col:starting_col + 3]

    def is_full(self) -> bool:
        # Full when there are no more empty cells
        return self.empty_cell_marker not in self._inner_grid
