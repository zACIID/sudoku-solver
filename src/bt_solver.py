"""
Sudoku solver based on Recursive Back-tracking and Constraint Propagation
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

import model as mod
import constraints as cn


# TODO metodo per non settare celle originali (controllare se all'inizio sono settate con EMPTY_VALUE?)


def sudoku_solver(sudoku: mod.SudokuGrid, starting_cell: mod.SudokuCell) -> mod.SudokuGrid:
    """

    :param sudoku:
    :param starting_cell:
    :return:
    """

    for attempt in range(1, 9+1):
        sudoku.set_value(starting_cell, val=attempt)


