from lib.utils import kitti_utils
import numpy as np


def dist_to_box(box, points):
    """
    Calculates the unsigned distance for each point in a list of points to a box
    :param box
    """
    box_corners = kitti_utils.boxes3d_to_corners3d(box)
