import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import math
import time
import deepxde as dde

from get_dde_PINN_data import get_dde_PINN_data
from dde_PINN import NN
from exact_u import u
from Plotting import plot_solution
from GetPredictData import GetPredictData
from bayesian_PINN_train import PINN_train

from bayes_opt import BayesianOptimization


L = 2.0  # Length of rectangle (R)
l = 2.0  # Width of rectangle (R)

# We choose l=L to have a square

# Bounds of x, y

x_lower = -L / 2.0
x_upper = L / 2.0
y_lower = -l / 2.0
y_upper = l / 2.0
R = 0.25  # Radius of circle (S)

# Center of the circle

Cx = 0.0
Cy = 0.0
mu_r = 1.0
freq = 0.3
v_0_1 = 0.3
omk = (2.0 * math.pi * freq) / v_0_1  # Pulsation
kap = 1.0 / (omk * mu_r)
eps1_r = 1.0
# eps2_r = 4.0
outer = dde.geometry.Rectangle(xmin=(x_lower, y_lower), xmax=(x_upper, y_upper))
inter = dde.geometry.Disk([Cx, Cy], R)
geom = outer


def is_on_boundary(x):
    return geom.on_boundary(x)


num_domain = 5000
num_boundary = 3500
num_test = 2500


def train_and_evaluate(hparams, eps):
    """
    根据 hparams 中的:
      - num_domain, num_boundary
      - w_pde, w_abc
      - lr
      - num_layers, num_neurons
    构建 & 训练 PINN，并返回 L2 relative error。
    """

    # 1. 提取并处理超参数

    w_pde = hparams.get("w_pde", 0.015)
    w_bc = hparams.get("w_bc", 0.125)

    # 构造 loss_weights，保持与原代码一致：[w_pde, w_pde, w_abc, w_abc, w_abc, w_abc]
    loss_weights = [w_pde, w_pde, w_bc, w_bc, w_bc, w_bc]

    data = get_dde_PINN_data(geom, eps, num_domain, num_boundary, num_test)
    model, losshistory, train_state = PINN_train(data, loss_weights)

    # 5. 在较粗网格上计算误差
    nbx, nby = 100, 100
    xc = np.linspace(x_lower, x_upper, nbx)
    yc = np.linspace(y_lower, y_upper, nby)
    x_grid, y_grid = np.meshgrid(xc, yc)
    xy_grid = np.vstack((np.ravel(x_grid), np.ravel(y_grid))).T

    predictions = model.predict(xy_grid)
    pred_Erz = predictions[:, 0]
    pred_Eiz = predictions[:, 1]

    # 真解
    cartesian_u_values = u(
        np.sqrt(x_grid ** 2 + y_grid ** 2),
        np.arctan2(y_grid, x_grid),eps
    )
    cartesian_u_values = np.nan_to_num(cartesian_u_values)
    real_u_values = np.real(cartesian_u_values).ravel()
    imaginary_u_values = np.imag(cartesian_u_values).ravel()

    l2_error_erz = dde.metrics.l2_relative_error(real_u_values, pred_Erz)
    l2_error_eiz = dde.metrics.l2_relative_error(imaginary_u_values, pred_Eiz)
    mean_l2_error = 0.5 * (l2_error_erz + l2_error_eiz)

    return mean_l2_error


def objective_for_bayes(w_pde, w_bc, eps):
    """
    贝叶斯优化要“最大化”此目标函数，而我们要“最小化”误差，所以返回 -error。
    """
    hparams = {
        "w_pde": w_pde,
        "w_bc": w_bc,
    }
    error = train_and_evaluate(hparams, eps)
    return -error  # 目标是最小化 error -> 返回值要最大化 -error


def run_bayesian_optimization(eps):
    """
    运行贝叶斯优化，搜索下列超参数:

      - w_pde in [0.001, 0.05]
      - w_abc in [0.05, 0.25]

    """

    pbounds = {

        'w_pde': (0.01, 0.1),
        'w_bc': (0.05, 0.25)

    }
    optimizer = BayesianOptimization(
        f=lambda w_pde, w_bc:objective_for_bayes(w_pde, w_bc, eps),
        pbounds=pbounds,
        verbose=2,
        random_state=1234
    )

    # ====================== 第一次 maximize ======================
    # 初始随机采样 10 次 + 第 1 批迭代 20 次
    optimizer.maximize(init_points=10, n_iter=10)

    # 记录并打印当前最优结果（迭代到 20 次）
    best_result = optimizer.max
    best_params = best_result["params"]
    best_error = -best_result["target"]  # 将 -error 转回 error
    print("\n===== 迭代 20 次后最优结果 =====")
    print("最优超参数:", best_params)
    print("对应的最小误差:", best_error)


    return optimizer, best_params, best_error


def get_param(eps):
    start_time = time.time()

    # 运行贝叶斯优化
    if  eps >=2:
        optimizer, best_params, best_error = run_bayesian_optimization(eps)
        w_pde = best_params["w_pde"]
        w_bc = best_params["w_bc"]
    else:
        w_pde = 0.015
        w_bc = 0.125

    end_time = time.time()
    print("总耗时:", end_time - start_time, "秒")

    return w_pde, w_bc

