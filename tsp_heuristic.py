import elkai
import numpy as np
import torch

CONST = 100000.0
def calc_dist(p, q):
    return np.sqrt(((p[1] - q[1])**2)+((p[0] - q[0]) **2)) * CONST

def get_ref_reward(pointset):

    if isinstance(pointset, torch.cuda.FloatTensor) or  isinstance(pointset, torch.FloatTensor):
        pointset = pointset.detach().numpy()

    num_points = len(pointset)
    ret_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i+1, num_points):
            ret_matrix[i,j] = ret_matrix[j,i] = calc_dist(pointset[i], pointset[j])
    q = elkai.solve_float_matrix(ret_matrix) # Output: [0, 2, 1]
    dist = 0
    for i in range(num_points):
        dist += ret_matrix[q[i], q[(i+1) % num_points]]
    return dist / CONST
