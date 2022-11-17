"""
This script provides an interface to test RAPTOR performances
It computes journeys with different:
    - origin and destination stops and time departure
    - raptor settings
"""
from __future__ import annotations

import argparse
import json
import os.path
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Dict

import pandas as pd
from loguru import logger

from src.main import SolverStrategy, solve_sudoku, parse_sudoku_file


OUT_FILENAME = "runner_out.csv"  # output file with performance info


@dataclass
class Query:
    sudoku_path: str | os.PathLike | bytes
    strategy: SolverStrategy

    def __str__(self) -> str:
        return f"{{sudoku: {self.sudoku_path}, strategy: {self.strategy}}}"


@dataclass
class QueryResult(Query):
    execution_time: float

    @staticmethod
    def from_query(query: Query, execution_time: float):
        return QueryResult(
            sudoku_path=query.sudoku_path,
            strategy=query.strategy,
            execution_time=execution_time
        )


def runner(config_path: str, output_dir: str):
    logger.debug("Configuration file          : {}", config_path)
    logger.debug("Output directory            : {}", output_dir)

    if not os.path.exists(output_dir):
        os.mkdir(path=output_dir)

    logger.debug("Loading runner configuration...")
    runner_config: Dict = _json_to_dict(config_path)

    logger.info("Executing queries...")
    queries = generate_queries(runner_config["queries"])
    results = []
    for q in queries:
        parsed_grid = parse_sudoku_file(q.sudoku_path)

        start_time = perf_counter()
        solve_sudoku(parsed_grid, q.strategy)
        end_time = perf_counter()

        results.append(QueryResult.from_query(query=q, execution_time=end_time - start_time))

    logger.info(f"Saving execution results at '{output_dir}'...")
    calculate_and_store_performance(results=results, out_dir=output_dir)


def _json_to_dict(file: str) -> Dict:
    """
    Convert a json to a dictionary
    :param file: path to json file
    :return: data as a dictionary
    """

    return json.load(open(file))


def generate_queries(query_objs: Mapping) -> Sequence[Query]:
    """
    Returns a sequence of queries based on the provided settings.

    :param query_objs: value of the "queries" field of the runner configuration
    :return: sequence of queries
    """

    logger.info("Reading sudoku queries...")
    generated_queries = []
    for q_obj in query_objs:
        q = Query(sudoku_path=q_obj["sudoku_path"], strategy=SolverStrategy(q_obj["strategy"]))

        generated_queries.append(q)

    return generated_queries


def calculate_and_store_performance(results: Sequence[QueryResult], out_dir: str | os.PathLike | bytes):
    perf_records = []
    for res in results:
        record = {
            "sudoku": res.sudoku_path,
            "strategy": str(res.strategy),
            "execution_time": res.execution_time
        }
        perf_records.append(record)

    results_df = pd.DataFrame.from_records(perf_records)
    results_df.to_csv(os.path.join(out_dir, OUT_FILENAME))


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="performance/runner_config.json",
        help="Runner configuration file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="performance/",
        help="Directory where the output of the runner is saved",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = _parse_arguments()

    runner(
        config_path=args.config,
        output_dir=args.output,
    )
