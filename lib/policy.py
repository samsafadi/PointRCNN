import numpy as np
import torch

from lib.utils import das_utils


def distance_from_gt_policy(pts_rect, pts_intensity, gt_boxes3d, ratio):
    """
    :param pts_rect: (N, 3)
    :param pts_intensity: (N, 1)
    :param gt_boxes: (M, 7) [x, y, z, h, w, l, ry]
    :param ratio: int ~ [0, 1]
    :return: pts_rect, pts_intensity
    """
    num_points = pts_rect.shape[0]
    # Calculates the distance from each point to the centroid of each box
    dists = das_utils.dist_to_boxes(pts_rect, gt_boxes3d)
    # minimum distance to a centroid for all points
    min_dists = torch.min(dists, 0)
    smallest_k_indices = torch.topk(min_dists[0], int(ratio * num_points), largest=False)[1]

    masked_lidar = []
    masked_intensity = []
    for idx in smallest_k_indices:
        masked_lidar.append(pts_rect[idx])
        masked_intensity.append(pts_intensity[idx])

    pts_rect = np.array(masked_lidar)
    pts_intensity = np.array(masked_intensity)

    return pts_rect, pts_intensity


def random_sample(pts_rect, pts_intensity, ratio):
    """
    Selects a random sample of points of size ratio*len(pts_rect)
    :param pts_rect: (N, 3)
    :param pts_intensity: (N, 1)
    :param ratio: int ~ [0, 1]
    :return: pts_rect, pts_intensity
    """
    sample = np.array(range(len(pts_rect)))
    sample = np.random.choice(sample, size=int(ratio * len(pts_rect)), replace=False)

    pts_rect = pts_rect[sample, :]
    pts_intensity = pts_intensity[sample, :]

    return pts_rect, pts_intensity


def roi_based_sample(pts_rect, pts_intensity, roi_scores, roi_boxes3d, ratio):
    """
    Samples points from pts_rect and pts_intensity based on the entropy gained
    as shown in roi_scores and roi_boxes3d
    :param pts_rect: (N, 3)
    :param pts_intensity: (N, 1)
    :param roi_scores: (B, M)
    :param roi_boxes3d: (B, M, 7)
    :return: pts_rect, pts_intensity
    """
    return