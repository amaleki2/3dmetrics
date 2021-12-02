import numpy as np
from scipy.spatial import cKDTree


def compute_chamfer_distance(gt_points, predicted_points):
    gt_points_kd_tree = cKDTree(gt_points)
    dist, _ = gt_points_kd_tree.query(predicted_points)
    chamfer_dist = np.mean(np.square(dist))
    return chamfer_dist


def compute_symmetric_chamfer_distance(gt_points, predicted_points):
    gt_to_pr_chamfer = compute_chamfer_distance(gt_points, predicted_points)
    pr_to_gt_chamfer = compute_chamfer_distance(predicted_points, gt_points)
    return gt_to_pr_chamfer + pr_to_gt_chamfer
