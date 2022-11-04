import numpy as np
import pytest

from src.sudoku import SudokuGrid

EMPTY_CELL = -777


@pytest.fixture(
    ids=[
        "easy",
        "medium",
        "hard"
    ],
    params=[
        # Easy grid
        np.ndarray([
            [EMPTY_CELL, 8, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 2, EMPTY_CELL, EMPTY_CELL],
            [EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 8, 4, EMPTY_CELL, 9, EMPTY_CELL],
            [EMPTY_CELL, EMPTY_CELL, 6, 3, 2, EMPTY_CELL, EMPTY_CELL, 1, EMPTY_CELL],
            [EMPTY_CELL, 9, 7, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 8, EMPTY_CELL],
            [8, EMPTY_CELL, EMPTY_CELL, 9, EMPTY_CELL, 3, EMPTY_CELL, EMPTY_CELL, 2],
            [EMPTY_CELL, 1, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 9, 5, EMPTY_CELL],
            [EMPTY_CELL, 7, EMPTY_CELL, EMPTY_CELL, 4, 5, 8, EMPTY_CELL, EMPTY_CELL],
            [EMPTY_CELL, 3, EMPTY_CELL, 7, 1, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL],
            [EMPTY_CELL, EMPTY_CELL, 8, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 4, EMPTY_CELL]
        ]),

        # Medium grid
        np.ndarray([
            [EMPTY_CELL, 8, EMPTY_CELL, 6, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 1, EMPTY_CELL],
            [EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 8, 2, 5, 6],
            [EMPTY_CELL, EMPTY_CELL, 1, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL],
            [EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 9, EMPTY_CELL, 4, 6, EMPTY_CELL, 3],
            [EMPTY_CELL, EMPTY_CELL, 9, EMPTY_CELL, 7, EMPTY_CELL, 5, EMPTY_CELL, EMPTY_CELL],
            [4, EMPTY_CELL, 7, 5, EMPTY_CELL, 2, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL],
            [EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 8, EMPTY_CELL, EMPTY_CELL],
            [7, 1, 3, 4, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL],
            [EMPTY_CELL, 5, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 9, EMPTY_CELL, 3, EMPTY_CELL]
        ]),

        # Hard grid
        np.ndarray([
            [EMPTY_CELL, EMPTY_CELL, 6, 3, EMPTY_CELL, 7, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL],
            [EMPTY_CELL, EMPTY_CELL, 4, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 5],
            [1, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 6, EMPTY_CELL, 8, 2],
            [2, EMPTY_CELL, 5, EMPTY_CELL, 3, EMPTY_CELL, 1, EMPTY_CELL, 6],
            [EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 2, EMPTY_CELL, EMPTY_CELL, 3, EMPTY_CELL, EMPTY_CELL],
            [9, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 7, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, 4],
            [EMPTY_CELL, 5, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL],
            [EMPTY_CELL, 1, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL],
            [EMPTY_CELL, EMPTY_CELL, 8, 1, EMPTY_CELL, 9, EMPTY_CELL, 4, EMPTY_CELL]
        ]),
    ])
def sudoku_grid(request) -> SudokuGrid:
    """
    Fixture parameterized to produce 3 sudoku grids: easy, medium, hard

    :param request: pytest fixture params
    :return: sudoku grid instance
    """
    return SudokuGrid(
        starting_grid=request.param,
        empty_cell_value=EMPTY_CELL
    )


@pytest.fixture
def solved_sudoku() -> SudokuGrid:
    solved_grid = np.ndarray([
        [8, 2, 7, 1, 5, 4, 3, 9, 6],
        [9, 6, 5, 3, 2, 7, 1, 4, 8],
        [3, 4, 1, 6, 8, 9, 7, 5, 2],
        [5, 9, 3, 4, 6, 8, 2, 7, 1],
        [4, 7, 2, 5, 1, 3, 6, 8, 9],
        [6, 1, 8, 9, 7, 2, 4, 3, 5],
        [7, 8, 6, 2, 3, 5, 9, 1, 4],
        [1, 5, 4, 7, 9, 6, 8, 2, 3],
        [2, 3, 9, 8, 4, 1, 5, 6, 7]
    ])

    return SudokuGrid(
        starting_grid=solved_grid,
        empty_cell_value=EMPTY_CELL
    )


@pytest.fixture(
    ids=[
        "row/column",
        "square"
    ],
    params=[
        np.ndarray([8, 2, EMPTY_CELL, 1, 5, 4, 3, EMPTY_CELL, EMPTY_CELL]),
        np.ndarray(
            [
                [8, 2, EMPTY_CELL],
                [9, 6, EMPTY_CELL],
                [3, EMPTY_CELL, EMPTY_CELL]
        ])
    ]
)
def valid_sudoku_row_column_square(request) -> np.ndarray:
    return request.param


@pytest.fixture(
    ids=[
        "row/column",
        "square"
    ],
    params=[
        np.ndarray([2, 2, EMPTY_CELL, 1, 5, 4, 2, EMPTY_CELL, EMPTY_CELL]),
        np.ndarray(
            [
                [8, 2, EMPTY_CELL],
                [9, 9, EMPTY_CELL],
                [3, EMPTY_CELL, 3]
        ])
    ]
)
def not_valid_sudoku_row_column_square(request) -> np.ndarray:
    return request.param
