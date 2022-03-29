from typing import Dict, Any

import numpy as np

import utils

NDArray = Any

RED   = [255, 0, 0]
BLACK = [  0, 0, 0]


def horizontal_vertical_cost(grad):
    raise NotImplementedError()


def vertical_backward_cost(grad):
    """
    calculate vertical cost matrix

    :param grad: gradient of the image in each point

    :return: cost matrix, min matrix that indicated where in the last row the minimum came from
    """
    energy = grad.copy()

    # pad zero row from above
    zero_row = np.zeros([1, energy.shape[1]])
    energy_pad = np.row_stack([zero_row, energy])

    # pad zero columns from the left and right
    zero_column = np.zeros([energy_pad.shape[0], 1])
    energy_pad = np.column_stack([zero_column, energy_pad, zero_column])

    image_range = np.arange(1,energy_pad.shape[1]-1)

    min_indices = np.zeros_like(grad, dtype=int)

    for i in range(1, energy_pad.shape[0]):
        # create a matrix of size [3 x image_width]
        # where the 3 rows correspond to the 3 options to go from left / center / right
        last_row_options = np.row_stack([energy_pad[i-1, :-2], energy_pad[i-1, 1:-1], energy_pad[i-1, 2:]])
        # min_index[i]: "-1" - came from left, "0" - came from center, "1" - came from right
        min_index = np.argmin(last_row_options, axis=0)-1
        energy_pad[i, 1:-1] += energy_pad[i-1, image_range+min_index]
        energy_pad[i, 0] = energy_pad[i, 1]
        energy_pad[i, -1] = energy_pad[i, -2]
        min_indices[i-1,:] = min_index

    return energy_pad[1:,1:-1], min_indices


def select_vertical_seam(cost, min_indices):
    """
    select vertical seams

    :param cost: cost function generated from the image
    :param min_indices: min indices matrix for backtracking

    :return: vector of seam indices (first index corresponds to the first row in the image
    """
    seam = np.zeros([cost.shape[0]],dtype=int)

    start = np.argmin(cost[:, -1]) # start from the last row

    seam[-1] = start

    for i in range(cost.shape[0]-1, 0, -1):
        new_index = seam[i] + min_indices[i, seam[i]]
        if new_index < 0:
            new_index = 0
        if new_index > cost.shape[1]:
            new_index = cost.shape[1]
        seam[i-1] = new_index

    return seam

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
    gradient = utils.get_gradients(intensity)
    cost, min_indices = horizontal_vertical_cost(gradient) if use_forward else vertical_backward_cost(gradient)
    seam = select_vertical_seam(cost, min_indices)

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

    # TODO:
    # horizontal_seams_viz[horizontal_seams] = BLACK
    # vertical_seams_viz[vertical_seams]     = RED

    return {'resized':          vertical_carved, # this is after both steps
            'horizontal_seams': horizontal_seams_viz,
            'vertical_seams':   vertical_seams_viz}
