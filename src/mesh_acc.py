from scipy.spatial import KDTree


def compute_mesh_accuracy(gt_points, pr_points, thresh=0.9):
    n_90percent = round(thresh * len(gt_points))
    gt_points_kd_tree = KDTree(gt_points)
    d1, _ = gt_points_kd_tree.query(pr_points)
    d1.sort()
    return d1[n_90percent]

