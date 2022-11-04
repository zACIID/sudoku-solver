"""
Sudoku solver based on Recursive Back-tracking and Constraint Propagation
"""
from __future__ import annotations

from typing import Tuple
from copy import deepcopy

import constraints as cn
from sudoku import SudokuGrid, CellCoordinates


# TODO metodo per non settare celle originali (controllare se all'inizio sono settate con EMPTY_VALUE?)


def sudoku_solver(sudoku: SudokuGrid) -> SudokuGrid:
    """
    Solves the provided sudoku with the Back-tracking + Constraint Propagation approach.
    Returns the solved sudoku grid.

    :param sudoku: sudoku grid to solve
    :return: solved sudoku grid
    """

    def solver_aux(grid: SudokuGrid, current_cell: CellCoordinates) -> Tuple[bool, SudokuGrid]:
        next_cell = current_cell.get_next()

        # If current cell is filled, go to next
        if sudoku.get_value(cell=current_cell) != sudoku.empty_cell_marker:
            if next_cell is None:
                return cn.is_solution_correct(sudoku), sudoku

            return solver_aux(grid=grid, current_cell=current_cell.get_next())

        for attempt in range(1, 9+1):
            # Allow current cell to be overwritten
            sudoku.set_value(cell=current_cell, val=attempt, only_if_empty=False)

            # Continue only if not at the end
            if next_cell is not None:
                solved, solution = solver_aux(grid=grid, current_cell=next_cell)
            else:
                solved = cn.is_solution_correct(solution=sudoku)
                solution = sudoku

            if solved:
                return True, solution

        # If no solution found at current cell, set as empty again and go back
        sudoku.set_value(cell=current_cell, val=sudoku.empty_cell_marker)
        return False, sudoku

    # Pass copy to avoid side effects
    sudoku_copy = deepcopy(sudoku)
    solved, solution = solver_aux(
        grid=sudoku_copy,
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
