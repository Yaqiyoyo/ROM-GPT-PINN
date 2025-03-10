from exact_u import u
from GetPredictData import GetPredictData
import numpy as np



nbx = 100
nby = 100
xy_grid, x_grid, y_grid = GetPredictData(nbx, nby)

eps_max = 5
eps_num = 30
u_r = np.zeros((xy_grid.shape[0], eps_num))
u_i = np.zeros((xy_grid.shape[0], eps_num))
epsilon = np.linspace(1, eps_max, eps_num)
for i in range(eps_num):
    cartesian_u_values = u(np.sqrt(x_grid**2 + y_grid**2), np.arctan2(y_grid, x_grid), epsilon[i])
    u_r[:, i] = np.real(cartesian_u_values).ravel()
    u_i[:, i] = np.imag(cartesian_u_values).ravel()

