import pytest
import numpy as np

from seam_carving import horizontal_backward_energy


def test_horizontal_backward_energy():
    grad = np.array([[5, 8, 12, 3],  # from the lecture slides
                     [4, 2,  3, 9],
                     [7, 3,  4, 2],
                     [5, 4,  7, 8]])

    expected_energy = np.array([[ 5,  8, 12,  3],
                                [ 9,  7,  6, 12],
                                [14,  9, 10,  8],
                                [14, 13, 15, 16]])
    energy = np.rot90(horizontal_backward_energy(np.rot90(grad)), -1) # rotated because we implemented it horizontally

    assert np.all(energy == expected_energy), {'got': energy, 'expected': expected_energy}
