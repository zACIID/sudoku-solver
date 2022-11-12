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


class NotInDomainException(Exception):
    def __init__(
            self, cell: CellCoordinates,
            actual_domain: Set[int],
            value: int,
            grid: np.ndarray = None
    ):
        """
        Exception raised in case a value doesn't belong to the domain
        of some sudoku cell

        :param cell: sudoku cell
        :param actual_domain: domain of the cell
        :param value: value to be set that isn't part of the domain
        :param grid: state of the sudoku grid
        """

        super(NotInDomainException, self).__init__(
            f"The cell {cell} cannot be set to value {value} "
            f"because it doesn't belong to its domain.\n"
            f"Actual domain: {actual_domain}\n"
            f"Sudoku grid:\n"
            f"{grid if grid is not None else 'N/A'}"
        )


class ConstraintPropagationSudokuGrid(SudokuGrid):

    _cell_domains: Dict[CellCoordinates, Set[int]]
    """
    Dictionary that pairs each cell coordinate with its domain, which consists
    in the set of 1 to 9 integers that can be assigned to that cell without
    violating any constraint
    """

    _affected_cells_cache: Dict[CellCoordinates, Set[CellCoordinates]]
    """
    Dictionary where the set of cells affected by the update of the one 
    used as key is cached. Cache starts as empty and key-value pairs are added
    as this instance is used.
    """

    def __init__(self, starting_grid: np.ndarray, empty_cell_marker: int = -777):
        super(ConstraintPropagationSudokuGrid, self).__init__(
            starting_grid=starting_grid,
            empty_cell_marker=empty_cell_marker
        )

        # Initialize cell domains by scanning the board and accounting
        # for starting non-empty cells
        self._cell_domains = {}
        for row in range(0, 8+1):
            for col in range(0, 8+1):
                current_cell = CellCoordinates(row=row, col=col)
                cell_domain = self._calculate_cell_domain(cell=current_cell)

                self._cell_domains[current_cell] = cell_domain

        # Cache initialized empty
        self._affected_cells_cache = {}

    def _calculate_cell_domain(self, cell: CellCoordinates) -> Set[int]:
        # All available numbers (moves)
        maximum_domain = set(range(1, 9+1))

        row_domain = set(self.get_row(i=cell.row))
        available_on_row = maximum_domain.difference(row_domain)

        col_domain = set(self.get_column(i=cell.col))
        available_on_col = maximum_domain.difference(col_domain)

        square_domain = set(self.get_square(cell=cell).flatten())
        available_on_square = maximum_domain.difference(square_domain)

        # Cell domain = intersection of available numbers in its row/col/square
        return set.intersection(available_on_row, available_on_square, available_on_col)

    @staticmethod
    def from_sudoku_grid(grid: SudokuGrid) -> ConstraintPropagationSudokuGrid:
        """
        Creates a new instance based on the provided sudoku grid.
        Everything is deep-copied to avoid side-effects.

        :param grid: existing sudoku grid to base the current one on
        :return: new instance based on the provided grid
        """

        return ConstraintPropagationSudokuGrid(
            starting_grid=grid.get_inner_grid_copy(),
            empty_cell_marker=grid.empty_cell_marker
        )

    def set_value(self, cell: CellCoordinates, val: int, overwrite: bool = True):
        if val not in self._cell_domains[cell]:
            raise NotInDomainException(
                cell=cell,
                actual_domain=self._cell_domains[cell],
                value=val,
                grid=self._inner_grid
            )

        if not overwrite:
            # Empty the cell first to add the previous value
            #   back to the domains of affected cells
            self.empty_cell(cell=cell)

        super(ConstraintPropagationSudokuGrid, self).set_value(
            cell=cell,
            val=val,
            overwrite=overwrite
        )

        #self._recalculate_affected_domains(cell_to_update=cell)
        self._remove_from_affected_domains(val=val, cell_to_update=cell) # TODO old

    def empty_cell(self, cell: CellCoordinates):
        previous_value = self.get_value(cell)

        super(ConstraintPropagationSudokuGrid, self).empty_cell(cell=cell)

        # If cell was previously non-empty, recalculate the affected domains,
        #   AFTER EMPTYING THE CELL
        #   i.e. domains of cells in same row/col/square, in order to see
        #   if the previous value of this cell becomes available
        if previous_value != self.empty_cell_marker:
            self._recalculate_affected_domains(cell_to_update=cell)

    def _get_affected_cells(self, cell_to_update: CellCoordinates) -> Set[CellCoordinates]:
        # Check if the set of affected cells has been calculated before
        if cell_to_update in self._affected_cells_cache:
            return self._affected_cells_cache[cell_to_update]
        else:
            row_cells = {
                CellCoordinates(row=cell_to_update.row, col=i)
                for i in range(0, 8+1)
            }
            col_cells = {
                CellCoordinates(row=i, col=cell_to_update.col)
                for i in range(0, 8+1)
            }

            square_starting_row = (cell_to_update.row // 3) * 3
            square_starting_col = (cell_to_update.col // 3) * 3
            square_cells = {
                CellCoordinates(row=i, col=j)
                for i in range(square_starting_row, square_starting_row+3)
                for j in range(square_starting_col, square_starting_col+3)
            }

            affected_cells = set.union(row_cells, col_cells, square_cells)
            self._affected_cells_cache[cell_to_update] = affected_cells

            return affected_cells

    def _remove_from_affected_domains(self, val: int, cell_to_update: CellCoordinates):
        affected_cells = self._get_affected_cells(cell_to_update=cell_to_update)
        for c in affected_cells:
            try:
                # Note: it is possible that not all the affected cells contain
                #   such a value in their domain
                if val in self._cell_domains[c]:
                    self._cell_domains[c].remove(val)

                # TODO discard should be the preferred method instead of if+set,
                #   but it somehow causes seg-fault in debugger???
                # self._cell_domains[c].discard(val)

            except KeyError:  # TODO debug remove
                raise NotInDomainException(
                    cell=c,
                    actual_domain=self._cell_domains[c],
                    value=val,
                    grid=self._inner_grid
                )

    def _recalculate_affected_domains(self, cell_to_update: CellCoordinates):
        affected_cells = self._get_affected_cells(cell_to_update=cell_to_update)
        for c in affected_cells:
            new_domain = self._calculate_cell_domain(cell=c)
            self._cell_domains[c] = new_domain

    def get_domain(self, cell: CellCoordinates) -> Set[int]:
        """
        Returns the domain (a copy of) the specified cell

        :param cell: cell to retrieve the domain of
        :return: domain of the specified cell
        """

        return self._cell_domains[cell].copy()

    def get_minimum_domain_empty_cell(self) -> Tuple[CellCoordinates, Set[int]] | Tuple[None, None]:
        """
        Returns the coordinates of the empty cell with the minimum (smallest) domain,
        along with said domain.

        :return: cell with smallest non-empty domain and its domain,
            or the pair (None, None) if there are no empty cells left
        """

        # Get empty cells
        # TODO fix type hinting
        empty_cells = list(
            filter(
                lambda kv_pair: self.get_value(kv_pair[0]) == self.empty_cell_marker,
                self._cell_domains.items()
            )
        )

        if len(empty_cells) == 0:
            return None, None

        # Get the empty cell with the smallest domain
        min_cell, min_domain = min(empty_cells, key=lambda kv_pair: len(kv_pair[1]))

        return min_cell, copy(min_domain)

    def get_maximum_domain_empty_cell(self) -> Tuple[CellCoordinates, Set[int]] | Tuple[None, None]:
        """
        Returns the coordinates of the empty cell with the maximum (biggest) domain,
        along with said domain.

        :return: cell with biggest non-empty domain and its domain,
            or the pair (None, None) if there are no empty cells left
        """

        # Get empty cells
        # TODO fix type hinting
        empty_cells: List[Tuple[CellCoordinates, Set[int]]] = list(
            filter(
                lambda kv_pair: self.get_value(kv_pair[0]) == self.empty_cell_marker,
                self._cell_domains.items()
            )
        )

        if len(empty_cells) == 0:
            return None, None

        # Get the empty cell with the smallest domain
        max_cell, max_domain = max(empty_cells, key=lambda kv_pair: len(kv_pair[1]))

        return max_cell, copy(max_domain)
