import numpy as np

import skeleton.rotationalOperators as rotationalOperators

randArr = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                   [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                   [[0, 1, 0], [0, 0, 1], [0, 0, 0]]], dtype=bool)


def test_column():
    assert rotationalOperators.column(randArr, 0).sum() == 1
    assert rotationalOperators.column(randArr, 1).sum() == 2
    assert rotationalOperators.column(randArr, 2).sum() == 0


def test_flipLrInX():
    try:
        rotationalOperators.flipLrInX(randArr[0:1])
    except AssertionError:
        print("error raised correctly")
    finally:
        assert rotationalOperators.flipLrInX(randArr).sum() == randArr.sum()


def test_flipUdInY():
    try:
        rotationalOperators.flipLrInX(randArr[0:1])
    except AssertionError:
        print("error raised correctly")
    finally:
        assert rotationalOperators.flipUdInY(randArr).sum() == randArr.sum()


def test_flipFbInZ():
    try:
        rotationalOperators.flipLrInX(randArr[0:1])
    except AssertionError:
        print("error raised correctly")
    finally:
        assert rotationalOperators.flipFbInZ(randArr).sum() == randArr.sum()


def test_rot3D90():
    try:
        rotationalOperators.flipLrInX(randArr[0:1])
    except AssertionError:
        print("error raised correctly")
    finally:
        assert rotationalOperators.rot3D90(randArr).sum() == randArr.sum()


def test_getDirectionList():
    test_array = np.array([[[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=np.uint8)
    directions_list = rotationalOperators.getDirectionsList(test_array)
    expected_results = [2, 24, 16, 6, 10, 26, 4, 20, 18, 22, 12, 8]
    for expected_result, direction in zip(expected_results, directions_list):
        assert direction.reshape(27).tolist()[expected_result - 1]
        assert direction.sum() == 1
