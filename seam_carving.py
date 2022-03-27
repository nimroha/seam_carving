from typing import Dict, Any

import numpy as np

import utils

NDArray = Any

RED   = [255, 0, 0]
BLACK = [  0, 0, 0]


def compute_gradient_magnitude(intensity):
    grad_y = np.diff(intensity, n=1, axis=0)
    grad_x = np.diff(intensity, n=1, axis=1)

    return np.sqrt((grad_x ** 2 + grad_y ** 2) / 2)


def get_top_neighbors_min_value(arr, i, j):
    """
    get the minimal value of the 3 top neighbors of the pixel (i,j)

    :param arr: numpy array of values
    :param i: row index
    :param j: col index

    :return: minimal value of top neighbors
    """
    if i == 0:
        return 0 # no left neighbors

    # get "in-bounds" rows of neighbors
    h, w = arr.shape[:2]
    neighbor_cols = [max(0, j - 1),
                     j,
                     min(w - 1, j + 1)]

    values = [arr[i - 1, col] for col in neighbor_cols] # values can repeat but it doesn't matter since we take the mean

    return np.min(values)


def horizontal_vertical_energy(grad):
    raise NotImplementedError()


def vertical_backward_energy(grad):
    energy = grad.copy()
    for i in range(energy.shape[0]):
        for j in range(energy.shape[1]):
            energy[i, j] += get_top_neighbors_min_value(energy, i, j)

    return energy


def select_vertical_seams(energy, k):
    """
    select horizontal seams

    :param energy: energy values to base the decision on
    :param k: number of seams to select

    :return: list of lists of seam indices
    """
    start = np.argmin(energy[:, -1]) # start from the last row
    # TODO select greedily or all at once


def resize_cols(image, k, use_forward):
    """
    find horizontal seams and remove/duplicate them

    :param image: numpy array of image to carve
    :param k: number of pixels to carve in each row, negative means remove
    :param use_forward: whether to use the forward energy for seam finding (False for simple gradient)

    :return: tuple - numpy array of carved image
                     list of indices of carved pixels
    """
    intensity = utils.to_grayscale(image)
    gradient  = compute_gradient_magnitude(intensity)
    energy    = horizontal_vertical_energy(gradient) if use_forward else vertical_backward_energy(gradient)

    return image, image

def resize(image: NDArray, out_height: int, out_width: int, forward_implementation: bool) -> Dict[str, NDArray]:
    """

    :param image: ِnp.array which represents an image.
    :param out_height: the resized image height
    :param out_width: the resized image width
    :param forward_implementation: a boolean flag that indicates whether forward or basic implementation is used.
                                    if forward_implementation is true then the forward-looking energy is used otherwise
                                    the basic implementation is used.
    :return: A dictionary with three elements, {'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3},
            where img1 is the resized image and img2/img3 are the visualization images
            (where the chosen seams are colored red and black for vertical and horizontal seams, respecitvely).
    """
    height_diff = out_height - image.shape[0]
    width_diff  = out_width  - image.shape[1]

    horizontal_carved,   horizontal_seams   = resize_cols(image, width_diff, forward_implementation)
    vertical_carved_rot, vertical_seams_rot = resize_cols(np.rot90(horizontal_carved), height_diff, forward_implementation)
    vertical_carved = np.rot90(vertical_carved_rot, -1)
    vertical_seams  = np.rot90(vertical_seams_rot,  -1)

    # add visualizations
    horizontal_seams_viz = image.copy() # take the image one step before the carving
    vertical_seams_viz   = horizontal_carved.copy() # same
    horizontal_seams_viz[horizontal_seams] = BLACK
    vertical_seams_viz[vertical_seams]     = RED

    return {'resized':          vertical_carved, # this is after both steps
            'horizontal_seams': horizontal_seams_viz,
            'vertical_seams':   vertical_seams_viz}
