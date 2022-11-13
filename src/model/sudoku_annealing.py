from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Set, Callable

import numpy as np

from src.model.sudoku_base import SudokuGrid, CellCoordinates


class SimulatedAnnealingSudokuGrid(SudokuGrid):
    _fixed_cells: Set[CellCoordinates]
    """
    Set containing all the cells that are non-empty at the start.
    Such cells cannot be randomly extracted during the simulated
    annealing procedure.
    """

    def __init__(self, starting_grid: np.ndarray, empty_cell_marker: int):
        super(SimulatedAnnealingSudokuGrid, self).__init__(
            starting_grid=starting_grid,
            empty_cell_marker=empty_cell_marker
        )

        self._fixed_cells = set()
        for row in range(0, 8+1):
            for col in range(0, 8+1):
                if starting_grid[row, col] != empty_cell_marker:
                    self._fixed_cells.add(CellCoordinates(row=row, col=col))

    @staticmethod
    def from_sudoku_grid(grid: SudokuGrid) -> SimulatedAnnealingSudokuGrid:
        """
        Creates a new instance based on the provided sudoku grid.
        Everything is deep-copied to avoid side-effects.

        :param grid: existing sudoku grid to base the current one on
        :return: new instance based on the provided grid
        """

        return SimulatedAnnealingSudokuGrid(
            starting_grid=grid.get_inner_grid_copy(),
            empty_cell_marker=grid.empty_cell_marker
        )

    def get_random_neighbor(self, cell: CellCoordinates) -> CellCoordinates:
        # Neighbors are retrieved from the 3x3 square adjacent to the provided cell
        # Such a square is allowed to "overflow" in case the cell is an edge cell
        neighbors: Set[CellCoordinates] = {
            # Row above
            CellCoordinates(row=(cell.row - 1) % 9, col=(cell.col - 1) % 9),
            CellCoordinates(row=(cell.row - 1) % 9, col=cell.col),
            CellCoordinates(row=(cell.row - 1) % 9, col=(cell.col + 1) % 9),
            # Same row
            CellCoordinates(row=cell.row, col=(cell.col - 1) % 9),
            CellCoordinates(row=cell.row, col=cell.col),
            CellCoordinates(row=cell.row, col=(cell.col + 1) % 9),
            # Col below
            CellCoordinates(row=(cell.row + 1) % 9, col=(cell.col - 1) % 9),
            CellCoordinates(row=(cell.row + 1) % 9, col=cell.col),
            CellCoordinates(row=(cell.row + 1) % 9, col=(cell.col + 1) % 9),
        }

        # Neighbors can't be cells from the starting grid
        valid_neighbors = list(neighbors.difference(self._fixed_cells))

        # Extract one randomly
        return valid_neighbors[np.random.randint(low=0, high=8+1)]


@dataclass
class SimulatedAnnealingState:
    """
    Class that represents the state of an independent Simulated Annealing
    solving run. This means that the class defines its own grid,
    scoring function and current cell.
    """

    current_cell: CellCoordinates
    grid: SimulatedAnnealingSudokuGrid
    scoring_function: Callable[[SudokuGrid], float]

    def __str__(self):
        return (f"Current cell: {self.current_cell}\n"
                f"Grid:\n"
                f"{self.grid}")

    @property
    def score(self) -> float:
        return self.scoring_function(self.grid)

    def update(self, next_cell: CellCoordinates, guess: int):
        # Important to copy, else the state pre- and post- update
        #   is virtually the same
        new_grid = deepcopy(self.grid)
        new_grid.set_value(cell=next_cell, val=guess)

        return SimulatedAnnealingState(
            current_cell=next_cell,
            grid=new_grid,
            scoring_function=self.scoring_function
        )

    # TODO do I need to keep track of # of bounces back and forth?
    #   might be useful to understand if I am stuck, in which case
    #   temperature should be increased by just the right amount
    #   to get out of the local optimum
