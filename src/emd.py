import pulp
import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import linprog


def compute_emd_scipy2(gt_points, predicted_points, return_assignment=False):
    if not len(gt_points) == len(predicted_points):
        raise ValueError("number of point samples should be equal")

    n = len(gt_points)
    n_eq = n + n + n * (n-1) // 2
    b = np.ones(n_eq)
    b[2 * n:] = 0  # last n equations have 0 right hand side
    A = np.zeros((n_eq, n * n))

    d = distance_matrix(gt_points, predicted_points)

    # sum of rows is 1
    for i in range(n):
        A[i, i*n:(i+1)*n] = 1

    # sum of columns is 1
    for i in range(n):
        A[i + n, i::n] = 1

    # A_ij = A_ji
    k = 2 * n
    for i in range(n):
        for j in range(i + 1, n):
            A[k, i*n + j] = 1
            A[k, i + j*n] = -1
            k += 1

    c = d.reshape(-1)
    bounds = (0, 1)
    sol = linprog(c, A_eq=A, b_eq=b, bounds=bounds)
    assignments = sol.x.reshape(n, n)
    loss = sol.fun
    if return_assignment:
        return loss, assignments
    else:
        return loss


def compute_emd(gt_points, predicted_points, integer_tag_on=True):
    if not len(gt_points) == len(predicted_points):
        raise ValueError("number of point samples should be equal")

    n = len(gt_points)
    d = distance_matrix(gt_points, predicted_points) ** 2
    prob = pulp.LpProblem("world_mover_distance", pulp.LpMinimize)
    if integer_tag_on:
        variables = [pulp.LpVariable("x" + str(i), 0, 1, pulp.LpInteger) for i in range(n * n)]
    else:
        variables = [pulp.LpVariable("x" + str(i), 0, 1) for i in range(n * n)]

    eq = 0
    for i in range(n):
        for j in range(n):
            eq += d[i, j] * variables[i * n + j]
    prob += eq

    for i in range(n):
        ieq = 0
        for j in range(n):
            ieq += variables[i * n + j]
        prob += ieq == 1

    for i in range(n):
        ieq = 0
        for j in range(n):
            ieq += variables[i + j * n]
        prob += ieq == 1

    for i in range(n):
        for j in range(i+1, n):
            ieq = variables[i + j * n] - variables[i * n + j]
            prob += ieq == 0

    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    loss = pulp.value(prob.objective) / n
    status = pulp.LpStatus[prob.status]
    if not (status == "Optimal"):
        raise RuntimeError("Optimal status not reached. status = %s, " % status)

    return loss
