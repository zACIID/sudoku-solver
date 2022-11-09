"""
Sudoku solver based on Recursive Back-tracking and Constraint Propagation
"""
from __future__ import annotations

from typing import Tuple
from copy import deepcopy

import constraints as cn
from sudoku import SudokuGrid, ConstraintPropagationSudokuGrid, CellCoordinates


def bt_cp_sudoku_solver(sudoku: SudokuGrid) -> SudokuGrid:
    """
    Solves the provided sudoku with the Back-tracking + Constraint Propagation approach.
    Returns the solved sudoku grid.

    :param sudoku: sudoku grid to solve
    :return: solved sudoku grid
    """

    def solver_aux(
            grid: ConstraintPropagationSudokuGrid,
            current_cell: CellCoordinates
    ) -> Tuple[bool, ConstraintPropagationSudokuGrid]:
        next_cell, min_domain = grid.get_minimum_domain_cell()

        # If current cell is filled, go to next
        if grid.get_value(cell=current_cell) != grid.empty_cell_marker:
            if next_cell is None:
                return cn.is_solution_correct(grid), grid

            return solver_aux(grid=grid, current_cell=current_cell.get_next())

        for attempt in range(1, 9+1):
            # Allow current cell to be overwritten
            grid.set_value(cell=current_cell, val=attempt, only_if_empty=False)

            # Continue only if not at the end
            if next_cell is not None:
                solved, solution = solver_aux(grid=grid, current_cell=next_cell)
            else:
                solved = cn.is_solution_correct(solution=grid)
                solution = grid

            if solved:
                return True, solution

        # If no solution found at current cell, set as empty again and go back
        grid.set_value(cell=current_cell, val=grid.empty_cell_marker)
        return False, grid

    # Pass copy to avoid side effects
    solved, solution = solver_aux(
        grid=ConstraintPropagationSudokuGrid.from_sudoku_grid(grid=sudoku),
        current_cell=CellCoordinates(row=0, col=0)
    )

    assert cn.is_solution_correct(solution=solution) and solved, (
        "The provided solution should be correct.\n"
        f"Return values:\n"
        f"Solved = {solved}\n"
        f"Solution =\n"
        f"{solution}"
    )

    return solution
