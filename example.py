import numpy as np
from src.chamfer import compute_symmetric_chamfer_distance
from src.emd import compute_emd
from src.mesh_acc import compute_mesh_accuracy


gt_points = np.array([[0.5, 3], [1, 4], [2, 4], [3, 5], [5, 5], [5, 2]])
predicted_points = np.array([[2, 3], [3, 3], [5, 4], [4, 2]])
chamfer_distance = compute_symmetric_chamfer_distance(gt_points, predicted_points)
print("chamfer distance = %0.3f" % chamfer_distance)

gt_points = gt_points[:3]
predicted_points = predicted_points[:3]
emd_distance = compute_emd(gt_points, predicted_points)

print("emd distance = %0.3f" % emd_distance)
