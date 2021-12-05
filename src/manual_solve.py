#!/usr/bin/python

import os, sys
import json
import numpy as np
import re

from colorama import Fore, Style, init

init()  # this colorama init helps Windows


### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.

""" Summary
Across these three tasks, I had to use the knowledge I gainied in this module
to solve these problems:
    - solve_0d3d703e() required me to use my knowledge of vectorization to solve.
    - solve_6f8cd79b() required me to use my knowledge slicing to solve it in a elegant way.
    - solve_9172f3a0() required me to use my knowledge of concatenation to solve it.
Each of these tasks tested my knowledge of both Python and Numpy.

"""


def solve_0d3d703e(x):
    """Map old values to new values.
    Take the original values, look up a dictionary (mapper)
    and replace it with its new value. The original value is the key
    of the dictionary and the new value is the value
    e.g. <Original Value> : <New Value>
    """
    mapper = {1: 5,
              2: 6,
              3: 4,
              4: 3,
              5: 1,
              6: 2,
              8: 9,
              9: 8, }
    x = np.vectorize(mapper.get)(x)
    return x


def solve_6f8cd79b(x):
    """This task requires you to turn all border blocks to the value of 8.
    I worked this out by taking the first and last each column and row and
    updating their values from 0 to 8.
    """
    # Change the first and last element of each row
    for i in range(len(x)):
        x[i][0] = 8
        x[i][-1] = 8
    # Change the first and last element of each column
    for j in range(len(x[0])):
        x[0][j] = 8
        x[-1][j] = 8
    return x


def solve_9172f3a0(x):
    """
    This task requires you magnify/ blow up a 3x3 array into a 9x9 array
    To solve this I first create an 3x3 array of each element
    Then creates 3 lists of 3x3 arrays. This would be the rows
    I then concatenate them to be 3 columns of 3x9 arrays
    After that I had to concatentate them again to be a single
    9x9 array.
    """
    new_shape = [3, 3]

    # Melt the 2D array (x) into a list
    melt = [ele for row in x for ele in row]

    # Create a new 3x3 array from each element in the list melt
    elem_magnified = [np.full(new_shape, num) for num in melt]

    # Break the list into three separate 3x3 arrays
    rows = np.array_split(elem_magnified, 3)

    # Stitch the columns together to make a three 3x9 arrays
    # This is simple column 1 of the originally array
    # expand/ magnified or multiplied by 3 in both the x and y axis
    s_arr = np.concatenate(rows, axis=1)

    # Stitch each of the arrays (columns) together to form a 9x9 array
    x = np.concatenate(s_arr, axis=1)

    # Return answer
    return x


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})"
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals():
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1)  # just the task ID
            solve_fn = globals()[name]  # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)


# ref https://stackoverflow.com/a/54955094
# Enum-like class of different styles
# these are the styles for background
class style():
    BLACK = '\033[40m'
    RED = '\033[101m'
    GREEN = '\033[42m'
    YELLOW = '\033[103m'
    BLUE = '\033[44m'
    MAGENTA = '\033[45m'
    CYAN = '\033[46m'
    WHITE = '\033[47m'
    RESET = '\033[0m'
    DARKYELLOW = '\033[43m'
    DARKRED = '\033[41m'
    DARKYELLOW = '\033[2m' + '\033[33m'
    DARKRED = '\033[2m' + '\033[31m'
    DARKWHITE = '\033[2m' + '\033[37m'


# the order of colours used in ARC
# (notice DARKYELLOW is just an approximation)
cmap = [style.BLACK,
        style.BLUE,
        style.RED,
        style.GREEN,
        style.YELLOW,
        style.WHITE,
        style.MAGENTA,
        style.DARKYELLOW,
        style.CYAN,
        style.DARKRED]


def echo_colour(x):
    s = " "  # print a space with a coloured background

    for row in x:
        for i in row:
            # print a character twice as grids are too "thin" otherwise
            print(cmap[int(i)] + s + s + style.RESET, end="")
        print("")


## TODO write a more convenient diff function, either for grids
## or for json files (because each task is stored as a single line
## of json in GitHub).


def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""

    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)


def show_result(x, y, yhat):
    print("Input")
    echo_colour(x)  # if echo_colour(x) doesn't work, uncomment print(x) instead
    # print(x)
    print("Correct output")
    echo_colour(y)
    # print(y)
    print("Our output")
    echo_colour(yhat)
    # print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__":
    main()
