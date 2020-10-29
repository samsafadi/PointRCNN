from lib.utils import kitti_utils
import numpy as np
import torch
import torch.nn as nn


def dist_to_box_centroid(points, box_centroid):
    """
    Calculates the unsigned distance for each point in a list of points to the centroid of a box
    setting value to zero if within box
    :param points: torch.Tensor(N, 3)
    :param box_centroid: torch.Tensor([x, y, z])
    :return: distances: torch.Tensor (1, N)
    """
    # Find the L2 distance between points and box_centroid using torch
    pdist = nn.PairwiseDistance(p=2)
    return pdist(points, box_centroid)


def dist_to_boxes(points, boxes):
    """
    Calculates combined distance for each point to all boxes
    :param points: (N, 3)
    :param boxes: (N, 7) [x, y, z, h, w, l, ry]
    :return: distances_array: (M) torch.Tensor of [(N), (N), ...] distances
    """
    distances_array = torch.Tensor([])
    box_corners = kitti_utils.boxes3d_to_corners3d(boxes)

    for box in box_corners:
        minX = min(box[:, 0])
        minY = min(box[:, 1])
        minZ = min(box[:, 2])
        maxX = max(box[:, 0])
        maxY = max(box[:, 1])
        maxZ = max(box[:, 2])
        centroid = np.array([(maxX + minX) / 2, (maxY + minY) / 2, (maxZ + minZ) / 2])
        dists_to_curr_box = dist_to_box_centroid(torch.from_numpy(points), torch.from_numpy(centroid)).reshape(1, len(points))
        distances_array = torch.cat((distances_array.float(), dists_to_curr_box.float()), 0)

    return distances_array
