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


def pt_info_to_input(pts_rect, pts_intensity, npoints, use_pts_intensity):
    """
    Calculates pts_input from pts_rect and pts_intensity
    :param pts_rect: (N, 3)
    :param pts_intensity: (N, 1)
    :param npoints: int
    :param use_intensity: bool
    :return: pts_input, ret_pts_rect, ret_pts_features
    """
    if npoints < len(pts_rect):
        pts_depth = pts_rect[:, 2]
        pts_near_flag = pts_depth < 40.0
        far_idxs_choice = np.where(pts_near_flag == 0)[0]
        near_idxs = np.where(pts_near_flag == 1)[0]
        near_idxs_choice = np.random.choice(near_idxs, npoints - len(far_idxs_choice), replace=False)

        choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
            if len(far_idxs_choice) > 0 else near_idxs_choice
        np.random.shuffle(choice)
    else:
        choice = np.arange(0, len(pts_rect), dtype=np.int32)

        if npoints > len(pts_rect):
            while len(choice) < npoints:
                extra_choice = np.random.choice(choice, min([npoints - len(pts_rect), len(pts_rect)]), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
        np.random.shuffle(choice)

    ret_pts_rect = np.expand_dims(pts_rect[choice, :], axis=0)
    ret_pts_intensity = pts_intensity[choice] - 0.5  # translate intensity to [-0.5, 0.5]

    pts_features = [ret_pts_intensity.reshape(-1, 1)]
    ret_pts_features = np.concatenate(pts_features, axis=1) if pts_features.__len__() > 1 else pts_features[0]

    if use_pts_intensity:
        pts_input = np.concatenate((ret_pts_rect, ret_pts_features), axis=1)  # (N, C)
    else:
        pts_input = ret_pts_rect

    return pts_input, ret_pts_rect, ret_pts_features
