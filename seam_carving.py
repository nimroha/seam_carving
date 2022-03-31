from typing import Dict, Any

import numpy as np

import utils

NDArray = Any

RED   = [255, 0, 0]
BLACK = [  0, 0, 0]


def vertical_cost(grad, intensity, use_forward):
    """
    calculate vertical cost matrix

    :param grad: gradient of the image in each point
    :param intensity: intensity values of the image (greyscale)
    :param use_forward: whether to compute forward energy cost or just gradient based

    :return: cost matrix, min matrix that indicated where in the last row the minimum came from
    """
    energy = grad.copy()

    # pad zero row from above
    zero_row = np.zeros([1, energy.shape[1]])
    energy_pad = np.row_stack([zero_row, energy])

    # pad infinity columns from the left and right
    inf_column = np.full([energy_pad.shape[0], 1], fill_value=np.inf)
    energy_pad = np.column_stack([inf_column, energy_pad, inf_column])

    if use_forward: # pad intensity as well
        # mirror the top row to avoid considering "phantom" edges at the top of the image # TODO necessary?
        intensity_pad = np.row_stack([intensity[0], intensity])

        # pad infinity columns from the left and right
        intensity_pad = np.column_stack([inf_column, intensity_pad, inf_column])

    image_range = np.arange(1,energy_pad.shape[1]-1)
    min_indices = np.zeros_like(grad, dtype=int)

    for i in range(1, energy_pad.shape[0]):
        # create a matrix of size [3 x image_width]
        # where the 3 rows correspond to the 3 options to go from left / center / right
        last_row_options = np.row_stack([energy_pad[i-1, :-2], energy_pad[i-1, 1:-1], energy_pad[i-1, 2:]])
        if use_forward: # add the cost of new edges
            last_row_new_edge_cost = None # TODO compute new edges here with intensity pad
            last_row_options += last_row_new_edge_cost

        # min_index[i]: "-1" - came from left, "0" - came from center, "1" - came from right
        min_index = np.argmin(last_row_options, axis=0)-1
        energy_pad[i, 1:-1] += energy_pad[i-1, image_range+min_index]
        energy_pad[i, 0] = energy_pad[i, 1]
        energy_pad[i, -1] = energy_pad[i, -2]
        min_indices[i-1,:] = min_index

    return energy_pad[1:,1:-1], min_indices


def select_vertical_seam(cost, min_indices):
    """
    select vertical seams (backtracking algorithm)

    :param cost: cost function generated from the image
    :param min_indices: min indices matrix for backtracking

    :return: vector of seam indices (first index corresponds to the first row in the image
    """
    seam = np.zeros([cost.shape[0]],dtype=int)

    start = np.argmin(cost[-1, :]) # start from the last row

    seam[-1] = start

    for i in range(cost.shape[0]-1, 0, -1):
        new_index = seam[i] + min_indices[i, seam[i]]
        if new_index < 0:
            new_index = 0
        if new_index > cost.shape[1]:
            new_index = cost.shape[1]
        seam[i-1] = new_index

    return seam


def roll_matrix(A, r):
    """
    Roll the rows of A by r.
    From: https://stackoverflow.com/questions/20360675/roll-rows-of-a-matrix-independently

    :param A: matrix to roll
    :param r: roll value
    :return: rolled matrix
    """
    rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]
    r[r < 0] += A.shape[1]
    column_indices = column_indices - r[:,np.newaxis]
    return A[rows, column_indices]


def remove_seam(matrix, seam):
    """
    remove seam from matrix

    :param matrix: 2d matrix
    :param seam: seam to be removed from matrix
    :return: matrix with 1 less column (according to the seam)
    """

    matrix = roll_matrix(matrix, -seam)
    matrix = matrix[:, 1:]
    matrix = roll_matrix(matrix, seam)
    return matrix


def resize_cols(image, k, use_forward):
    """
    find horizontal seams and remove/duplicate them

    :param image: numpy array of image to carve
    :param k: number of pixels to carve in each row, negative means remove
    :param use_forward: whether to use the forward energy for seam finding (False for simple gradient)

    :return: tuple - numpy array of carved image
                     matrix that indicates the selected pixels
    """
    intensity = utils.to_grayscale(image)
    gradient = utils.get_gradients(intensity)

    column_indices = np.tile(np.arange(gradient.shape[1]), [gradient.shape[0], 1])
    row_indices = np.tile(np.arange(gradient.shape[0])[:,np.newaxis], gradient.shape[1])
    output_seam = np.zeros(gradient.shape, dtype=bool)

    for current_k in range(np.abs(k)):
        cost, min_indices = vertical_cost(gradient, intensity, use_forward)
        seam = select_vertical_seam(cost, min_indices)

        # set output seam
        current_seam_original_image_columns = column_indices[np.arange(gradient.shape[0]), seam]
        current_seam_original_image_rows    = row_indices[np.arange(gradient.shape[0]), seam]
        output_seam[current_seam_original_image_rows, current_seam_original_image_columns] = True

        # remove the relevant items from each matrix
        column_indices = remove_seam(column_indices, seam)
        row_indices    = remove_seam(row_indices, seam)
        intensity      = remove_seam(intensity, seam)
        gradient       = utils.get_gradients(intensity)

    # reduce / enlarge the image
    new_image_shape = list(image.shape)
    new_image_shape[1] = new_image_shape[1] + k
    if k < 0:
        # remove elements
        flattened_image = np.delete(image.reshape([-1, 3]),output_seam.flatten(),axis=0)
        image = flattened_image.reshape(new_image_shape)
    else:
        # add elements
        indices_of_items_to_repeat = np.argwhere(output_seam.flatten()).flatten()
        flattened_image = image.reshape([-1, 3])
        items_to_repeat = flattened_image[indices_of_items_to_repeat, :]
        flattened_image = np.insert(flattened_image, indices_of_items_to_repeat, np.squeeze(items_to_repeat), axis=0)
        image = flattened_image.reshape(new_image_shape)

    return image, output_seam


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

    vertical_carved,       vertical_seams       = resize_cols(image, width_diff, forward_implementation)
    horizontal_carved_rot, horizontal_seams_rot = resize_cols(np.rot90(vertical_carved), height_diff, forward_implementation)
    carved = np.rot90(horizontal_carved_rot, -1)
    horizontal_seams  = np.rot90(horizontal_seams_rot,  -1)

    # add visualizations
    vertical_seams_viz   = image.copy() # take the image one step before the carving
    horizontal_seams_viz = vertical_carved.copy() # same

    vertical_seams_viz[vertical_seams]     = RED
    horizontal_seams_viz[horizontal_seams] = BLACK

    return {'resized':          carved, # this is after both steps
            'horizontal_seams': horizontal_seams_viz,
            'vertical_seams':   vertical_seams_viz}
