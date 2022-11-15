from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Set, Tuple

import numpy as np

import src.solvers.utils as ut
from src.model.sudoku_base import SudokuGrid, CellCoordinates


# TODO do this again by using the other strategy:
#   rows contain the correct values, and the random neighbor step consists
#   in swapping two random positions of a random row
#   cost function is simply the number of duplicates of cols + squares
#   Note: useful link: https://www.adrian.idv.hk/2019-01-30-simanneal/
#       at section "Parameter estimation" is written how to set the optimal
#       temperature, which could be useful for the report


class SimulatedAnnealingSudokuGrid(SudokuGrid):
    starting_cells: Set[CellCoordinates]
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

        self.starting_cells = set()
        for row in range(0, 8+1):
            for col in range(0, 8+1):
                current_cell = CellCoordinates(row=row, col=col)
                if starting_grid[row, col] != empty_cell_marker:
                    self.starting_cells.add(current_cell)

        self._initialize_rows_as_unique()

    def _initialize_rows_as_unique(self):
        """
        Fill all the empty cells in such a way that each row contains exactly one occurrence
        of each valid sudoku number (1 to 9)
        """

        all_sudoku_numbers = set(range(1, 9+1))
        for row in range(0, 8+1):
            remaining_numbers = all_sudoku_numbers.difference(set(self.get_row(i=row)))

            for col in range(0, 8+1):
                current_cell = CellCoordinates(row=row, col=col)
                if self.get_value(current_cell) == self.empty_cell_marker:
                    to_set = remaining_numbers.pop()
                    self.set_value(current_cell, val=to_set)

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

    def swap_two_cells_same_row(self):
        cell_1, cell_2 = self._get_random_row_neighbors()
        val_1, val_2 = self.get_value(cell_1), self.get_value(cell_2)

        self.set_value(cell_1, val_2)
        self.set_value(cell_2, val_1)

    def _get_random_row_neighbors(self) -> Tuple[CellCoordinates, CellCoordinates]:
        # Keep cycling until one suitable row is found
        # (one where there are two or more non-starting cells)
        valid_neighbors = None
        while valid_neighbors is None:
            rnd_row_idx = np.random.randint(0, 8+1)
            row_neighbors: Set[CellCoordinates] = {
                CellCoordinates(row=rnd_row_idx, col=i)
                for i in range(0, 8+1)
            }
            valid_neighbors = row_neighbors.difference(self.starting_cells)

            if len(valid_neighbors) < 2:
                valid_neighbors = None

        valid_neighbors = list(valid_neighbors)
        rnd_idxs = np.random.random_integers(low=0, high=len(valid_neighbors)-1, size=2)
        try:
            neighbor_1 = valid_neighbors[rnd_idxs[0]]
            neighbor_2 = valid_neighbors[rnd_idxs[1]]
        except IndexError as ex:
            raise ex

        return neighbor_1, neighbor_2

    def get_score(self) -> float:
        def get_collection_score(collection: np.ndarray) -> float:
            duplicates, counts = ut.get_duplicates(collection)
            duplicates_sum = int(np.sum(counts)) if len(counts) > 0 else 0

            return duplicates_sum

        tot_score = 0
        for i in range(0, 8+1):
            col_score = get_collection_score(self.get_column(i))
            row_score = get_collection_score(self.get_row(i))

            tot_score += col_score + row_score

        for i in range(0, 2+1):
            for j in range(0, 2+1):
                top_left_cell = CellCoordinates(row=i*3, col=j*3)
                square_score = get_collection_score(self.get_square(top_left_cell))

                tot_score += square_score

        return tot_score


@dataclass
class SimulatedAnnealingState:
    """
    Class that represents the state of an independent Simulated Annealing
    solving run
    """

    temperature: float
    grid: SimulatedAnnealingSudokuGrid
    _score: float = ut.LARGE_NUMBER

    def __str__(self):
        return (f"Score: {self.score}\n"
                f"Grid:\n"
                f"{self.grid}")

    @property
    def score(self) -> float:
        if self._score == ut.LARGE_NUMBER:
            self._score = self.grid.get_score()

        return self._score

    def get_next(self) -> SimulatedAnnealingState:
        # Important to copy, else the state pre- and post- update
        #   is virtually the same
        new_grid = deepcopy(self.grid)
        new_grid.swap_two_cells_same_row()

        return SimulatedAnnealingState(
            temperature=self.temperature,
            grid=new_grid
        )
