"""
Sudoku solver based on Simulated Annealing
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from loguru import logger

import src.solvers.utils as ut
from src.model.sudoku_annealing import SimulatedAnnealingSudokuGrid, SimulatedAnnealingState
from src.model.sudoku_base import SudokuGrid


@dataclass
class SimulatedAnnealingParams:
    starting_temp: float
    """
    Starting temperature value used in the annealing approach
    """

    temp_decrease_rate: float
    """
    Rate, less than 1, that the temperature decreases at.
    i.e. T_X+1 = T_X * rate
    """

    max_epochs: int
    """
    Max number of epochs to execute the solver for 
    """


def simulated_annealing_solver(
        sudoku: SudokuGrid,
        params: SimulatedAnnealingParams = None
) -> SudokuGrid | None:
    """
    Solves the provided sudoku with the Simulated Annealing approach.
    Returns the solved sudoku grid, or None if no solution could be found.

    :param sudoku: sudoku grid to solve
    :param params: simulated annealing params
    :return: solved sudoku grid, or None if no solution found
    """

    if params is None:
        params = SimulatedAnnealingParams(
            starting_temp=3,
            temp_decrease_rate=0.95,
            max_epochs=1000000
        )

    if params.temp_decrease_rate >= 1:
        raise ValueError("Temperature decrease rate must be < 1")

    # Create grid instance that supports constraint propagation
    annealing_grid = SimulatedAnnealingSudokuGrid.from_sudoku_grid(grid=sudoku)
    current_state = SimulatedAnnealingState(temperature=params.starting_temp, grid=annealing_grid)
    prev_state_counter: Tuple[SimulatedAnnealingState, int] = (current_state, 0)

    solution = None
    epoch = 0
    while solution is None and epoch <= params.max_epochs:
        new_state = current_state.get_next()
        score_delta = new_state.score - current_state.score
        if score_delta <= 0:  # new state is better
            current_state = new_state
        else:  # old state is better (delta > 0)
            # The right side of the inequality gets closer to 0 the lower the temperature
            if np.random.rand() < 10 ** (-score_delta / current_state.temperature):
                current_state = new_state

        # Terminate if solution found (score == 0)
        if current_state.score == 0:
            solution = current_state.grid
            break

        current_state.temperature *= params.temp_decrease_rate
        epoch += 1

        # If current score (state) equals previous score (state), increase counter by 1
        # Counter is needed to understand when the algorithm gets stuck
        prev_state, prev_counter = prev_state_counter
        if current_state.score == prev_state.score:
            prev_state_counter = (prev_state, prev_counter + 1)
        else:
            prev_state_counter = (current_state, 0)

        # If stuck in the same state for some time, increase temperature again to get out
        prev_state, prev_counter = prev_state_counter
        stuckness_threshold = 100  # Multiplier+1 every 250 times the score doesn't change
        stuckness_multiplier = prev_counter / (stuckness_threshold / 2)

        # Temp reset is lower the more time passes
        epoch_multiplier = max((1 - (epoch / params.max_epochs)), 0.01)
        if prev_counter > stuckness_threshold:
            current_state.temperature = min(
                params.starting_temp * epoch_multiplier * stuckness_multiplier,
                params.starting_temp
            )

    if solution is None:
        logger.debug(f"Couldn't find solution for grid:\n"
                     f"{sudoku}")
        return None

    if ut.is_solution_correct(solution=solution):
        logger.debug(f"Found solution at epoch {epoch}:\n"
                     f"{solution}")
        return solution

    assert False, (f"Found solution, but incorrect. Should never get here:\n"
                   f"{sudoku}")
