# sudoku-solver

## Summary

Simple sudoku solver made for educational purposes, as part of a university course project.
At the moment, the following solving strategies are supported:

- _Backtracking with Constraint Propagation_
- _Simulated Annealing_

Not all the supported strategies are optimal to solve the sudoku problem, but, as mentioned before, 
the main purpose of this work is educational.

## Executing the solver

The script `src/main.py` can be launched to execute the solver on a sudoku grid provided via text file.
Such a file has the following format:

```
12x456789
123456x89
123456789
1x3456789
123456789
1234x6789
1234x6789
123456x89
12345x789
```

with each `x` marking an empty cell on the sudoku grid. `x` can be any non-null character that is not a 1 to 9 digit.

### Examples

> `python src/main.py -p "sudokus/example1.txt" -s "cp"`

> `python src/main.py -p "sudokus/example1.txt" -s "sim_ann"`

