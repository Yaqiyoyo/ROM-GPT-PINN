import numpy as np

def A_z_analytic(grid_points, mu):
    return (mu / (3 * np.pi ** 2)) * np.sin(np.pi * grid_points[:, 0:1]) * np.sin(np.pi * grid_points[:, 1:2]) * np.sin(np.pi * grid_points[:, 2:3])