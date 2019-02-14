import numpy as np

# PV Note
# USE THE FOLLOWING FUNCTIONS LIKE CONSTANTS
# DO NOT CHANGE THE SIZES OF THIS FUNCTIONS, IF THEY ARE CHANGED, MANY TESTS THAT USE THIS TESTLIB
# MIGHT BREAK, Probable refactor when time permits


def get_hilbert_curve():
    return np.array([[[1, 1, 1],
                      [1, 0, 1],
                      [1, 0, 1]],
                     [[0, 0, 0],
                      [0, 0, 0],
                      [1, 0, 1]],
                     [[1, 1, 1],
                      [1, 0, 1],
                      [1, 0, 1]]], dtype=bool)


def get_tiny_loop():
    # array of loop
    tiny_loop = np.zeros((5, 5), dtype=bool)
    tiny_loop[1:4, 1:4] = np.array([[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]], dtype=bool)
    return tiny_loop


def get_tiny_loop_3d():
    tiny_loop = get_tiny_loop()
    tiny_loop_3d = np.zeros((3, 5, 5), dtype=bool)
    tiny_loop_3d[1] = tiny_loop
    return tiny_loop_3d


def get_tiny_loop_with_branches():
    # array of a cycle with branches
    tiny_loop = np.zeros((5, 5), dtype=bool)
    tiny_loop[1:4, 1:4] = np.array([[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]], dtype=bool)
    tiny_loop[0, 2] = 1
    tiny_loop[4, 2] = 1
    tiny_loop_with_branches = np.zeros((3, 5, 5), dtype=bool)
    tiny_loop_with_branches[1] = tiny_loop
    return tiny_loop_with_branches


def get_disjoint_trees_no_cycle_3d():
    # array of two disjoint trees
    cross_pair = np.zeros((10, 10, 10), dtype=bool)
    cross = np.zeros((5, 5), dtype=bool)
    cross[:, 2] = 1
    cross[2, :] = 1
    cross_pair[0, 0:5, 0:5] = cross
    cross_pair[5, 5:10, 5:10] = cross
    return cross_pair


def get_single_line():
    # array of no branches single straight line
    sample_line = np.zeros((5, 5, 5), dtype=bool)
    sample_line[1, :, 4] = 1
    return sample_line


def get_tiny_loops_with_branches():
    # array of 2 cycles with branches
    tiny_loop = np.zeros((10, 10), dtype=bool)
    loop = np.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]], dtype=bool)
    tiny_loop[1:4, 1:4] = loop
    tiny_loop[0, 2] = 1
    tiny_loop[4, 2] = 1
    tiny_loop[5:8, 1:4] = loop
    tiny_loop[4, 2] = 1
    tiny_loop[8, 2] = 1
    tiny_loops_with_branches = np.zeros((3, 10, 10), dtype=bool)
    tiny_loops_with_branches[1] = tiny_loop
    return tiny_loops_with_branches


def sum_of_powers(*args):
    return sum(2 ** power for power in args)
