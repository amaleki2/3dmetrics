import numpy as np
import matplotlib
import matplotlib.pyplot as plt
font = {'family': 'normal', 'size': 16}
matplotlib.rc('font', **font)


def plot_figure(distance="chamfer"):
    gt_points = np.array([[0.5, 3], [1, 4], [2, 4], [3, 5], [5, 5], [5, 2]])
    predicted_points = np.array([[2, 3], [3, 3], [5, 4], [4, 2]])
    if distance == "emd":
        gt_points = gt_points[:3]
        predicted_points = predicted_points[:3]
        assignments = [[0, 0], [1, 1], [2, 2]]
    else:
        assignments = [[0, 2], [1, 2], [2, 4], [3, 5]]

    plt.scatter(gt_points[:, 0], gt_points[:, 1], marker='o', s=100, c='red')
    plt.scatter(predicted_points[:, 0], predicted_points[:, 1], marker='o', s=100, c='blue')

    for i, (x, y) in enumerate(gt_points):
        plt.annotate(str(i+1), (x+0.05, y+0.075), fontsize=16)

    for i, (x, y) in enumerate(predicted_points):
        plt.annotate(str(i + 1), (x+0.05, y+0.075), fontsize=16)

    for i, j in assignments:
        x1, y1 = predicted_points[i]
        x2, y2 = gt_points[j]
        plt.plot([x1, x2], [y1, y2], linestyle='--', linewidth=3, color='k')

    plt.xlim(0, 6)
    plt.ylim(0, 6)
    plt.grid(which="major", linestyle='--')

    plt.show()
