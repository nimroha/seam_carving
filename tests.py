import os
import pytest
import numpy as np

from seam_carving import vertical_cost, roll_matrix, remove_seam, resize
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
@pytest.mark.parametrize('use_forward', [False]) # TODO add True afte implementing
def test_resize(test_img, h, w, use_forward):
    img = test_img.copy()
    results = resize(img, h ,w, use_forward)
    resized = results['resized']
    horizontal_seams_viz = results['horizontal_seams']
    vertical_seams_viz   = results['vertical_seams']
    assert resized.shape == (h, w, 3) # the requested size
    assert vertical_seams_viz.shape   == (TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, 3)
    assert horizontal_seams_viz.shape == (TEST_IMAGE_HEIGHT, w, 3)
