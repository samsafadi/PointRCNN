from lib.utils import kitti_utils
import numpy as np


def dist_to_box_centroid(box, points):
    """
    Calculates the unsigned distance for each point in a list of points to the centroid of a box
    setting value to zero if within box
    :param box: [x, y, z, h, w, l, ry]
    :param points: (N, 3)
    """

    if box is None or len(points) == 0:
        return None

    box_corners = kitti_utils.boxes3d_to_corners3d(box)
    if box_corners.size == 0:
        return

    box_corners = box_corners[0]

    minX = min(box_corners[:, 0])
    minY = min(box_corners[:, 1])
    minZ = min(box_corners[:, 2])
    maxX = max(box_corners[:, 0])
    maxY = max(box_corners[:, 1])
    maxZ = max(box_corners[:, 2])

    centroid = np.array([(maxX + minX) / 2, (maxY + minY) / 2, (maxZ + minZ) / 2])
    distances = []
    for point in points:
        if point[0] >= minX and point[0] <= maxX and point[1] >= minY \
                and point[1] <= maxY and point[2] >= minZ and point[2] <= maxZ:
            distances.append(0)
        else:
            distances.append(np.sqrt((point[0] - centroid[0])**2 + (point[1] - centroid[1])**2 + (point[2] - centroid[2])**2))

    return np.array(distances)


def dist_to_boxes(boxes, points):
    """
    Calculates combined distance for each point to all boxes
    :param boxes: (N, 7) [x, y, z, h, w, l, ry]
    :param points: (N, 3)
    """
    assert len(boxes) > 0, "provide more than 0 boxes"

    dists = dist_to_box_centroid(boxes[0], points)
    if len(boxes) > 1:
        for box in boxes[1:]:
            dists += dist_to_box_centroid(box, points)

    return dists


def inverse_distance_policy(distances):
    """
    Calculates 1/distances putting inf if distance=0
    :param distances: (1, N)
    """
    scores = np.array([])
    for d in distances:
        if d == 0:
            scores.append(np.inf)
        else:
            scores.append(1 / d)

    return scores
