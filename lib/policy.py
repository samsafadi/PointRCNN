import numpy as np
import torch

from lib.utils import das_utils
from lib.utils.roipool3d.roipool3d_utils import pts_in_boxes3d_cpu


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


def roi_based_sample(pts_rect, pts_intensity, prev_pts_rect, prev_pts_intensity, roi_scores, roi_boxes3d, ratio):
    """
    Samples points from pts_rect and pts_intensity based on the entropy gained
    as shown in roi_scores and roi_boxes3d. Uses points found in last pass and adds to them
    :param pts_rect: (N, 3)
    :param pts_intensity: (N, 1)
    :param prev_pts_rect: (N, 3)
    :param prev_pts_intensity: (N, 1)
    :param roi_scores: (M)
    :param roi_boxes3d: (M, 7)
    :return: pts_rect, pts_intensity
    """
    # send everything to cpu
    roi_scores = roi_scores.cpu()
    roi_boxes3d = roi_boxes3d.cpu()

    pts_and_intensity = np.append(pts_rect, pts_intensity, axis=1)
    prev_pts_and_intensity = np.append(prev_pts_rect, prev_pts_intensity, axis=1)

    # Sets of tuples of points to calculate diff
    prev_pts_set = set(map(tuple, prev_pts_and_intensity))
    pts_set = set(map(tuple, pts_and_intensity))

    # Get pts not in previous pass
    pts_diff_set = pts_set - prev_pts_set
    pts_diff = np.array(list(pts_diff_set))

    # pts and intensity diff
    pts_rect_diff = pts_diff[:, :3]

    seen = set()
    point_sets = list(range(len(roi_scores)))
    confidence = (1 / (1 + np.exp(-roi_scores.numpy())))
    entropy = das_utils.confidence_to_entropy(confidence)
    normalized_entropy_distribution = (1 / sum(entropy)) * entropy

    highest_to_lowest_entropy = sorted(range(len(entropy)), key=entropy.__getitem__, reverse=True)

    # Points in boxes in order of decreasing entropy to prevent duplicates
    for i in highest_to_lowest_entropy:
        # Add new sampled points to set
        pts_in_box_mask = pts_in_boxes3d_cpu(torch.Tensor(pts_rect_diff), roi_boxes3d[i].unsqueeze(0))[0]
        pts_in_box = pts_diff[pts_in_box_mask]
        for pt in pts_in_box:
            if tuple(pt) in seen:
                np.delete(pts_in_box, np.argwhere(pts_in_box == pt))
            else:
                seen.add(tuple(pt))

        point_sets[i] = pts_in_box

    # Sampled points (add prev points)
    sampled_points = prev_pts_and_intensity

    # Fill with random points if less points than desired
    if (len(pts_rect) * ratio > sum([len(point_sets[i]) for i in range(len(point_sets))])):
        new_sampled_points = pts_and_intensity[pts_in_boxes3d_cpu(torch.Tensor(pts_rect), roi_boxes3d)[0]]
        shuffled_pts = np.random.permutation(pts_and_intensity)
        i = 0
        j = int(len(pts_rect) * ratio) - len(new_sampled_points)
        while i < j:
            if tuple(shuffled_pts[i]) in seen:
                j += 1
            else:
                new_sampled_points = np.concatenate((new_sampled_points, shuffled_pts[i].reshape(1, 4)), axis=0)
            i += 1

        sampled_points = np.concatenate((sampled_points, new_sampled_points), axis=0)
        return np.array(sampled_points)[:, :3], np.array(sampled_points)[:, 3]

    # Sampling stage
    # Flattened indices to sample from
    indices = np.array([(i, j) for i in range(len(point_sets)) for j in range(len(point_sets[i]))])
    # Probabilities for all of the indices
    probs = np.array([normalized_entropy_distribution[i] / len(point_sets[i])
                        for i in range(len(point_sets)) for j in range(len(point_sets[i]))])
    probs /= probs.sum()

    sampled_indices_indices = np.random.choice(range(len(indices)), size=int(ratio * len(pts_rect)), p=probs, replace=False)
    sampled_indices = indices[sampled_indices_indices]
    new_sampled_points = [point_sets[idx[0]][idx[1]] for idx in sampled_indices]

    sampled_points = np.concatenate((sampled_points, new_sampled_points), axis=0)

    # print(sampled_points)
    return np.array(sampled_points)[:, :3], np.array(sampled_points)[:, 3]
