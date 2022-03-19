from typing import Dict, Any

import numpy as np

import utils


RED = [255, 0, 0]

BLACK = [0, 0, 0]

NDArray = Any

def horizontal_forward_energy(arr):
    pass


def horizontal_grad_energy(arr):
    pass


def horizontal_resize(image, k, use_forward):
    """
    find horizontal seams and remove/duplicate them

    :param image: numpy array of image to carve
    :param k: number of pixels to carve in each row, negative means remove
    :param use_forward: whether to use the forward energy for seam finding (False for simple gradient)

    :return: tuple - numpy array of carved image
                     list of indices of carved pixels
    """
    intensity = utils.to_grayscale(image)
    energy = horizontal_forward_energy(intensity) if use_forward else horizontal_grad_energy(intensity)

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

    horizontal_carved,   horizontal_seams   = horizontal_resize(image,                       width_diff,  forward_implementation)
    vertical_carved_rot, vertical_seams_rot = horizontal_resize(np.rot90(horizontal_carved), height_diff, forward_implementation)
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
