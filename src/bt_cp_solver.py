"""
Sudoku solver based on Recursive Back-tracking and Constraint Propagation
"""
from __future__ import annotations

from typing import Tuple, Set

from loguru import logger

import src.constraints as cn
from src.sudoku import SudokuGrid, ConstraintPropagationSudokuGrid, CellCoordinates


def bt_cp_sudoku_solver(sudoku: SudokuGrid) -> SudokuGrid:
    """
    Solves the provided sudoku with the Back-tracking + Constraint Propagation approach.
    Returns the solved sudoku grid.

    :param sudoku: sudoku grid to solve
    :return: solved sudoku grid
    """

    def solver_aux(
            grid: ConstraintPropagationSudokuGrid,
            current_cell: CellCoordinates,
            current_domain: Set[int]
    ) -> Tuple[bool, ConstraintPropagationSudokuGrid]:
        # If None it means that all cells have been exhausted and the sudoku was solved
        if current_cell is None:
            return cn.is_solution_correct(grid), grid

        # If current cell is filled, go to next
        if grid.get_value(cell=current_cell) != grid.empty_cell_marker:
            next_cell, next_domain = grid.get_minimum_domain_empty_cell()

            return solver_aux(grid=grid, current_cell=next_cell, current_domain=next_domain)

        logger.info(f"Attempting cell {current_cell}")
        logger.debug(f"Current cell domain: {current_domain}")
        logger.debug(f"Current Grid:\n"
                     f"{grid.get_inner_grid_copy()}")

        if len(current_domain) == 0:
            # If the current cell has an empty domain return False
            #   because it can't happen unless the grid is wrong
            return False, grid

        for attempt in current_domain:
            # Allow current cell to be overwritten
            grid.set_value(cell=current_cell, val=attempt, only_if_empty=False)

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

    logger.info(f"Starting to solve sudoku from cell {starting_cell}")
    solved, solution = solver_aux(
        grid=cp_grid,
        current_cell=starting_cell,
        current_domain=starting_domain
    )

    assert cn.is_solution_correct(solution=solution) and solved, (
        "The provided solution should be correct.\n"
        f"Return values:\n"
        f"Solved = {solved}\n"
        f"Solution =\n"
        f"{solution}"
    )

    return solution
