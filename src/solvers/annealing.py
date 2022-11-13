"""
Sudoku solver based on Simulated Annealing
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from loguru import logger

import src.solvers.utils as ut
from src.model.sudoku_base import SudokuGrid, CellCoordinates
from src.model.sudoku_annealing import SimulatedAnnealingSudokuGrid, SimulatedAnnealingState


def simulated_annealing_solver(
        sudoku: SudokuGrid,
        starting_temp: float = 100,
        temp_step: float = 0.005,
        scoring_function: Callable[[SudokuGrid], float] = None
) -> SudokuGrid | None:
    """
    Solves the provided sudoku with the Simulated Annealing approach.
    Returns the solved sudoku grid, or None if no solution could be found.

    :param sudoku: sudoku grid to solve
    :param starting_temp: starting temperature value used in the annealing approach
    :param temp_step: value that the temperature decreases by after each "epoch"
        of the algorithm (i.e. each time a cell is set)
        (TODO also step at which temp increases in case the algo is stuck?)
    :param scoring_function: scoring function to execute the annealing procedure with.
        Default score is calculated as the number of empty cells + duplicates in the grid
    :return: solved sudoku grid, or None if no solution found
    """

    def default_scoring_function(grid: SudokuGrid) -> float:
        total_duplicates = 0
        for i in range(0, 8+1):
            # Notes:
            # - empty cells all have the same value, so they
            #   are included as duplicates in the `counts` array
            # - need to consider each row (or col or square) to properly
            #   count duplicates, because if the whole grid is considered at once,
            #   each number will always have duplicates
            duplicates, counts = ut.get_duplicates(grid.get_row(i=i))

            total_duplicates += np.sum(counts) if len(counts) > 0 else 0

        return total_duplicates

    # Create grid instance that supports constraint propagation
    annealing_grid = SimulatedAnnealingSudokuGrid.from_sudoku_grid(grid=sudoku)
    scoring_function = default_scoring_function if scoring_function is None else scoring_function

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
    while solution is not None and current_temperature > 0:
        for k, current_state in states.items():
            next_cell = current_state.grid.get_random_neighbor(cell=current_state.current_cell)
            guess = np.random.randint(low=0, high=8+1)
            new_state = current_state.update(next_cell=next_cell, guess=guess)

            logger.debug(f"[{k}] Old grid:\n"
                         f"{current_state}")
            logger.debug(f"[{k}] Candidate grid:\n"
                         f"{new_state}")

            score_delta = new_state.score - current_state.score
            if score_delta < 0:  # new state is better
                states[k] = new_state
            else:  # old state is better
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
