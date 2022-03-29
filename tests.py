import pytest
import numpy as np

from seam_carving import vertical_backward_cost


def test_horizontal_backward_energy():
    grad = np.array([[5, 8, 12, 3],  # from the lecture slides
                     [4, 2,  3, 9],
                     [7, 3,  4, 2],
                     [5, 4,  7, 8]])

    expected_energy = np.array([[ 5,  8, 12,  3],
                                [ 9,  7,  6, 12],
                                [14,  9, 10,  8],
                                [14, 13, 15, 16]])
    cost, min_indices = vertical_backward_cost(grad)

    assert np.all(cost == expected_energy), {'got': cost, 'expected': expected_energy}
