"""
Sudoku solver based on Simulated Annealing
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from loguru import logger

import src.solvers.utils as ut
from src.model.sudoku_base import SudokuGrid, CellCoordinates
from src.model.sudoku_annealing import SimulatedAnnealingSudokuGrid, SimulatedAnnealingState


def simulated_annealing_solver(
        sudoku: SudokuGrid,
        starting_temp: float = 10,
        temp_step: float = 0.0005
) -> SudokuGrid | None:
    """
    Solves the provided sudoku with the Simulated Annealing approach.
    Returns the solved sudoku grid, or None if no solution could be found.

    :param sudoku: sudoku grid to solve
    :param starting_temp: starting temperature value used in the annealing approach
    :param temp_step: value that the temperature decreases by after each "epoch"
        of the algorithm (i.e. each time a cell is set)
        (TODO also step at which temp increases in case the algo is stuck?)
    :return: solved sudoku grid, or None if no solution found
    """

    # Create grid instance that supports constraint propagation
    annealing_grid = SimulatedAnnealingSudokuGrid.from_sudoku_grid(grid=sudoku)

    starting_cell = CellCoordinates(row=0, col=0)
    states: Dict[int, SimulatedAnnealingState] = {
        # TODO implement beam search with k initial states (pass k as solver arg)
        #   in such a case, grid needs to be copied k times
        1: SimulatedAnnealingState(
            current_cell=starting_cell,
            grid=annealing_grid,
            scoring_function=scoring_function
        )
    }
    logger.debug(f"Starting to solve sudoku from cell {starting_cell}")
    solution = None
    current_temperature = starting_temp
    while solution is None and current_temperature > 0:
        for k, current_state in states.items():
            next_cell = current_state.grid.get_random_neighbor(cell=current_state.current_cell)
            guess = get_good_guess(grid=current_state.grid, cell=current_state.current_cell)
            new_state = current_state.update(next_cell=next_cell, guess=guess)

            logger.debug(f"Temperature: {current_temperature}")
            logger.debug(f"[{k}] Old grid (score: {current_state.score}):\n"
                         f"{current_state}")
            logger.debug(f"[{k}] Candidate grid (score: {new_state.score}):\n"
                         f"{new_state}")

            score_delta = new_state.score - current_state.score
            if score_delta <= 0:  # new state is better
                states[k] = new_state
            else:  # old state is better (delta > 0)
                # The right side of the inequality gets closer to 0 the lower the temperature,
                #   meaning that the worse state is increasingly less likely to get chosen
                #   the more the temperature decreases
                if np.random.rand() < 10**(-score_delta / current_temperature):
                    states[k] = new_state

        for k, current_state in states.items():
            if current_state.score == 0:
                solution = current_state.grid
                break

        current_temperature -= temp_step

    if solution is not None and ut.is_solution_correct(solution=solution):
        logger.debug(f"Found solution:\n"
                     f"{solution}")
        return solution
    else:
        logger.debug(f"Couldn't find solution for grid:\n"
                     f"{sudoku}")
        return None


def scoring_function(grid: SudokuGrid) -> float:
    # Notes:
    # - empty cells all have the same value, so they
    #   are included as duplicates in the `counts` array
    # - need to consider each row (or col or square) to properly
    #   count duplicates, because if the whole grid is considered at once,
    #   each number will always have duplicates
    row_duplicates = 0
    for i in range(0, 8+1):
        duplicates, counts = ut.get_duplicates(grid.get_row(i=i))
        row_duplicates += np.sum(counts) if len(counts) > 0 else 0

    col_duplicates = 0
    for i in range(0, 8+1):
        duplicates, counts = ut.get_duplicates(grid.get_column(i=i))
        col_duplicates += np.sum(counts) if len(counts) > 0 else 0

    square_duplicates = 0
    for i in range(0, 2+1):
        for j in range(0, 2+1):
            first_cell = CellCoordinates(row=i, col=j)
            duplicates, counts = ut.get_duplicates(grid.get_square(cell=first_cell))
            square_duplicates += np.sum(counts) if len(counts) > 0 else 0

    # Returns the max number of duplicates between cols, rows and squares
    # This way, the scoring function provides feedback for every "dimension" (col/row/square)
    return max(row_duplicates, col_duplicates, square_duplicates)


def get_good_guess(grid: SudokuGrid, cell: CellCoordinates) -> int:
    all_numbers = set(range(1, 9+1))
    row_numbers = set(grid.get_row(i=cell.row))
    col_numbers = set(grid.get_column(i=cell.col))
    square_numbers = set(grid.get_square(cell=cell).flatten())

    valid_guesses = all_numbers.difference(set.intersection(row_numbers, col_numbers, square_numbers))

    if len(valid_guesses) > 0:
        rnd_idx = np.random.randint(low=0, high=len(valid_guesses))
        return list(valid_guesses)[rnd_idx]
    else:
        return grid.empty_cell_marker
