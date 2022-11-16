from __future__ import annotations

import argparse
import os
import time
from enum import Enum

import numpy as np
from loguru import logger

from src.model.sudoku_base import SudokuGrid
from src.solvers.annealing import simulated_annealing_solver, SimulatedAnnealingParams
from src.solvers.bt_cp import bt_cp_sudoku_solver


EMPTY_CELL_MARKER = -777


class SolverStrategy(Enum):
    SimulatedAnnealing = "sim_ann"
    ConstraintPropagation = "cp"

    def __str__(self):
        return self.value


def solve_sudoku(grid: np.ndarray, strategy: SolverStrategy) -> SudokuGrid | None:
    sudoku_grid = SudokuGrid(starting_grid=grid, empty_cell_marker=EMPTY_CELL_MARKER)

    def handle_simulated_annealing() -> SudokuGrid | None:
        return simulated_annealing_solver(
            sudoku=sudoku_grid,
            params=SimulatedAnnealingParams(
                starting_temp=3,
                temp_decrease_rate=0.95,
                max_epochs=3000000
            )
        )

    def handle_constraint_propagation() -> SudokuGrid | None:
        return bt_cp_sudoku_solver(sudoku=sudoku_grid)

    strategy_switch = {
        SolverStrategy.SimulatedAnnealing: handle_simulated_annealing,
        SolverStrategy.ConstraintPropagation: handle_constraint_propagation
    }
    solver = strategy_switch[strategy]
    return solver()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--sudoku_path",
        type=str,
        default="sudokus/example1.txt",
        help="Path to a valid sudoku file that represents the grid to solve",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        choices=[str(strat) for strat in SolverStrategy],
        default=SolverStrategy.ConstraintPropagation.value,
        help="Strategy that the solver will adopt. Available strategies:\n"
             f"{[str(strat) for strat in SolverStrategy]}",
    )

    arguments = parser.parse_args()
    return arguments


def parse_sudoku_file(path: str | os.PathLike | bytes) -> np.ndarray:
    """
    Parse files that represent a sudoku grid. The format is the following:

    | 12x456789
    | 123456x89
    | 123456789
    | 1x3456789
    | 123456789
    | 1234x6789
    | 1234x6789
    | 123456x89
    | 12345x789

    Where ```x``` represents the empty character, which can be any non-null character.

    :param path: path to the file
    :return: numpy array representing the grid
    """

    with open(path) as sudoku_file:
        lines = sudoku_file.read().split()
        cells = [c for l in lines for c in l]

        rows = []
        for row in range(0, 8+1):
            current_row = []
            for col in range(0, 8+1):
                # Cells is the "linearized" 9x9 grid
                current_cell: str = cells[row*9 + col]
                if current_cell.isnumeric() and 1 <= int(current_cell) <= 9:
                    current_row.append(int(current_cell))
                else:
                    # Any number != 1 to 9 is good to represent empty cells
                    current_row.append(EMPTY_CELL_MARKER)

            rows.append(current_row)

        return np.array(rows)


if __name__ == "__main__":
    start_time = time.perf_counter()

    args = parse_args()

    logger.info(f"Parsing file at '{args.sudoku_path}'")
    parsed_grid = parse_sudoku_file(args.sudoku_path)

    logger.info(f"Executing solver with strategy '{args.strategy}'")
    solution = solve_sudoku(grid=parsed_grid, strategy=SolverStrategy(args.strategy))

    if solution is None:
        logger.warning(f"Couldn't find solution for grid:\n"
                       f"{parsed_grid}")
    else:
        logger.info(f"Found solution for grid:\n"
                    f"{parsed_grid}\n"
                    f"\n"
                    f"Solution:\n"
                    f"{solution}")

    end_time = time.perf_counter()
    logger.info(f"Total execution time: {end_time  - start_time} [s]")


# TODO provare a parte, test troppo lenti

## Medium grid
# np.array([
#    [EMPTY_CELL, 8, EMPTY_CELL, 6, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 1, EMPTY_CELL],
#    [EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 8, 2, 5, 6],
#    [EMPTY_CELL, EMPTY_CELL, 1, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL],
#    [EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 9, EMPTY_CELL, 4, 6, EMPTY_CELL, 3],
#    [EMPTY_CELL, EMPTY_CELL, 9, EMPTY_CELL, 7, EMPTY_CELL, 5, EMPTY_CELL, EMPTY_CELL],
#    [4, EMPTY_CELL, 7, 5, EMPTY_CELL, 2, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL],
#    [EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 8, EMPTY_CELL, EMPTY_CELL],
#    [7, 1, 3, 4, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL],
#    [EMPTY_CELL, 5, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 9, EMPTY_CELL, 3, EMPTY_CELL]
# ]),
#
## Hard grid
# np.array([
#    [EMPTY_CELL, EMPTY_CELL, 6, 3, EMPTY_CELL, 7, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL],
#    [EMPTY_CELL, EMPTY_CELL, 4, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 5],
#    [1, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 6, EMPTY_CELL, 8, 2],
#    [2, EMPTY_CELL, 5, EMPTY_CELL, 3, EMPTY_CELL, 1, EMPTY_CELL, 6],
#    [EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 2, EMPTY_CELL, EMPTY_CELL, 3, EMPTY_CELL, EMPTY_CELL],
#    [9, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 7, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 4],
#    [EMPTY_CELL, 5, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL],
#    [EMPTY_CELL, 1, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL],
#    [EMPTY_CELL, EMPTY_CELL, 8, 1, EMPTY_CELL, 9, EMPTY_CELL, 4, EMPTY_CELL]
# ]),
