"""
Sudoku solver based on Recursive Back-tracking and Constraint Propagation
"""
from __future__ import annotations

from typing import Tuple, Set

from loguru import logger

import src.solvers.utils as ut
from src.model.sudoku_base import SudokuGrid, CellCoordinates
from src.model.sudoku_cp import ConstraintPropagationSudokuGrid


def bt_cp_sudoku_solver(sudoku: SudokuGrid) -> SudokuGrid | None:
    """
    Solves the provided sudoku with the Back-tracking + Constraint Propagation approach.
    Returns the solved sudoku grid, or None if no solution could be found.

    :param sudoku: sudoku grid to solve
    :return: solved sudoku grid, or None if no solution found
    """

    def solver_aux(
            grid: ConstraintPropagationSudokuGrid,
            current_cell: CellCoordinates,
            current_domain: Set[int]
    ) -> Tuple[bool, ConstraintPropagationSudokuGrid]:
        # If None it means that all cells have been exhausted and the sudoku was solved
        if current_cell is None:
            return ut.is_solution_correct(grid), grid

        if len(current_domain) == 0:
            # If the current cell has an empty domain return False
            #   because it can't happen unless the grid is wrong
            return False, grid

        # This should never happen because only empty cells should be fed to the solver
        assert grid.get_value(cell=current_cell) == grid.empty_cell_marker, "Cell should be empty"

        logger.debug(f"Attempting cell {current_cell}. Domain: {current_domain}")
        logger.debug(f"Current Grid:\n"
                     f"{grid}")

        for attempt in current_domain:
            # Allow current cell to be overwritten
            grid.set_value(cell=current_cell, val=attempt, overwrite=False)

            # Go to the next cell after having tried with the current attempt
            # Note: the next cell and its domain are recalculated every time
            #   a new attempt is made, because such an action affects the
            #   domains of other cells
            next_cell, next_domain = grid.get_minimum_domain_empty_cell()
            solved, solution = solver_aux(
                grid=grid,
                current_cell=next_cell,
                current_domain=next_domain
            )

            if solved:
                return True, solution

        # If no solution found at current cell, set as empty again and go back
        logger.debug(f"Attempts for cell {current_cell} exhausted, returning...")
        grid.empty_cell(cell=current_cell)
        return False, grid

    # Create grid instance that supports constraint propagation
    cp_grid = ConstraintPropagationSudokuGrid.from_sudoku_grid(grid=sudoku)
    starting_cell, starting_domain = cp_grid.get_minimum_domain_empty_cell()

    logger.debug(f"Starting to solve sudoku from cell {starting_cell}")
    solved, solution = solver_aux(
        grid=cp_grid,
        current_cell=starting_cell,
        current_domain=starting_domain
    )

    if ut.is_solution_correct(solution=solution) and solved:
        logger.debug(f"Found solution:\n"
                     f"{solution}")
        return solution
    else:
        logger.debug(f"Couldn't find solution for grid:\n"
                     f"{sudoku}")
        return None
