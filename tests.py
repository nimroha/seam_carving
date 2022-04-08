import os
import pytest
import numpy as np

from seam_carving import vertical_cost, roll_matrix, remove_seam, resize, compute_new_edge_cost
from utils import open_image

TEST_IMAGE_PATH   = os.path.join(os.path.dirname(__file__), 'checkpoint.jpeg')
TEST_IMAGE_HEIGHT = 262
TEST_IMAGE_WIDTH  = 400

@pytest.fixture(scope='session')
def test_img():
    return open_image(TEST_IMAGE_PATH)

def test_vertical_backward_energy():
    grad = np.array([[5, 8, 12, 3],  # from the lecture slides
                     [4, 2,  3, 9],
                     [7, 3,  4, 2],
                     [5, 4,  7, 8]])

    expected_energy = np.array([[ 5,  8, 12,  3],
                                [ 9,  7,  6, 12],
                                [14,  9, 10,  8],
                                [14, 13, 15, 16]])
    cost, min_indices = vertical_cost(grad, intensity=None, use_forward=False) # only test gradient ("backwards")

    assert np.all(cost == expected_energy), {'got': cost, 'expected': expected_energy}


def test_roll_matrix():
    arr = np.eye(5)
    roll_by = np.arange(5)
    rolled = roll_matrix(arr, -roll_by)
    assert np.all(rolled[:,1:] == 0)
    assert np.all(rolled[:, 0] == 1)


def test_remove_seam():
    arr = np.eye(5)
    seam_cols = np.arange(5)
    removed = remove_seam(arr, seam_cols)
    assert arr.shape[1] == removed.shape[1] + 1 # removed one column
    assert arr.shape[0] == removed.shape[0]     # did not remove rows
    assert np.all(removed == 0)


@pytest.mark.parametrize('h', [TEST_IMAGE_HEIGHT, TEST_IMAGE_HEIGHT + 30, TEST_IMAGE_HEIGHT - 30])
@pytest.mark.parametrize('w', [TEST_IMAGE_WIDTH,  TEST_IMAGE_WIDTH  + 30, TEST_IMAGE_WIDTH  - 30])
@pytest.mark.parametrize('use_forward', [False, True])
def test_resize(test_img, h, w, use_forward):
    img = test_img.copy()
    results = resize(img, h ,w, use_forward)
    resized = results['resized']
    horizontal_seams_viz = results['horizontal_seams']
    vertical_seams_viz   = results['vertical_seams']
    assert resized.shape == (h, w, 3) # the requested size
    assert vertical_seams_viz.shape   == (TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, 3)
    assert horizontal_seams_viz.shape == (TEST_IMAGE_HEIGHT, w, 3)


def test_compute_new_edge_cost():
    intensity = np.array([[10, 10, 11,  9,  5,  5],
                          [ 9,  6, 10,  5, 12, 19],
                          [ 6, 19, 19,  8, 14,  9],
                          [ 9,  9, 12, 19,  9, 11],
                          [16, 13, 18, 19,  8, 19]], dtype=np.int32)

    padded  = np.row_stack([np.zeros(intensity.shape[1]), intensity])
    inf_col = np.full(padded.shape[1], fill_value=np.inf)
    padded  = np.column_stack([inf_col, padded, inf_col])
    x, left_y, right_y = compute_new_edge_cost(padded)

    # compute the explicit naive loops to compare too
    expected_x = np.zeros_like(padded)
    for i in range(1, padded.shape[0]):
        for j in range(1, padded.shape[1] - 1):
            expected_x[i, j] = np.abs(padded[i, j - 1] - padded[i, j + 1])

    expected_left_y = np.zeros_like(padded)
    for i in range(1, padded.shape[0]):
        for j in range(1, padded.shape[1] - 1):
            expected_left_y[i, j] = np.abs(padded[i - 1, j] - padded[i, j - 1])

    expected_right_y = np.zeros_like(padded)
    for i in range(1, padded.shape[0]):
        for j in range(1, padded.shape[1] - 1):
            expected_right_y[i, j] = np.abs(padded[i - 1, j] - padded[i, j + 1])

    expected_left_y [:2] = 0 # zero out padding and first row
    expected_right_y[:2] = 0

    assert np.all(x      [1:, 1:-1] == expected_x      [1:, 1:-1])
    assert np.all(left_y [1:, 1:-1] == expected_left_y [1:, 1:-1])
    assert np.all(right_y[1:, 1:-1] == expected_right_y[1:, 1:-1])
