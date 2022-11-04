"""
Sudoku solver based on Recursive Back-tracking and Constraint Propagation
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

import constraints as cn
from sudoku import SudokuGrid, CellCoordinates


# TODO metodo per non settare celle originali (controllare se all'inizio sono settate con EMPTY_VALUE?)


def sudoku_solver(sudoku: SudokuGrid, starting_cell: CellCoordinates) -> SudokuGrid:
    """

    :param sudoku:
    :param starting_cell:
    :return:
    """

    for attempt in range(1, 9+1):
        sudoku.set_value(starting_cell, val=attempt)


