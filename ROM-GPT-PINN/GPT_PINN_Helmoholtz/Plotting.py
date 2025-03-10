import matplotlib.pyplot as plt
import numpy as np

# plt.style.use(['science', 'notebook'])
import deepxde as dde
from Log.log import logger

def plot_solution(predictions, exact, x_grid, y_grid, eps, path):
    path = path
    eps = eps
    nbx = x_grid.shape[0]
    nby = y_grid.shape[0]
    pred_Erz = predictions[:, 0].reshape(nbx, nby)
    pred_Eiz = predictions[:, 1].reshape(nbx, nby)
    cartesian_u_values = np.nan_to_num(exact)
    # Extract real and imaginary parts of the values

    real_u_values = np.real(cartesian_u_values)
    imaginary_u_values = np.imag(cartesian_u_values)

    # absolute error
    err_Eiz = np.abs(imaginary_u_values - pred_Eiz)
    err_Erz = np.abs(real_u_values - pred_Erz)

    # Draw the colorations of Eiz and Erz on the plan

    fig,ax = plt.subplots(2,3, figsize=(15,8))

    # Draw Eiz_pred

    axp0= ax[0,0].pcolor(x_grid, y_grid, pred_Eiz, cmap='seismic', shading='auto')
    cbar0= fig.colorbar(axp0, ax=ax[0,0])
    ax[0,0].set_xlabel('x')
    ax[0,0].set_ylabel('y')
    ax[0,0].set_title('Predicted $E_{iz}$')
    ax[0,0].set_aspect('equal')

    # Draw Eiz_exact
    
    axp1= ax[0,1].pcolor(x_grid, y_grid, imaginary_u_values, cmap='seismic', shading='auto')
    cbar0= fig.colorbar(axp1, ax=ax[0,1])
    ax[0,1].set_xlabel('x')
    ax[0,1].set_ylabel('y')
    ax[0,1].set_title('Exact $E_{iz}$')
    ax[0,1].set_aspect('equal')

    # Draw err
    axp2= ax[0,2].pcolor(x_grid, y_grid, err_Eiz, cmap='seismic', shading='auto')
    cbar0= fig.colorbar(axp2, ax=ax[0,2])
    ax[0,2].set_xlabel('x')
    ax[0,2].set_ylabel('y')
    ax[0,2].set_title('Error $E_{iz}$')
    ax[0,2].set_aspect('equal')

    # Draw Erz_pred

    axp3= ax[1,0].pcolor(x_grid, y_grid, pred_Erz, cmap='seismic', shading='auto')
    cbar1= fig.colorbar(axp3, ax=ax[1,0])
    ax[1,0].set_xlabel('x')
    ax[1,0].set_ylabel('y')
    ax[1,0].set_title('Predicted $E_{rz}$')
    ax[1,0].set_aspect('equal')

    # Draw Erz_exact

    axp4= ax[1,1].pcolor(x_grid, y_grid, real_u_values, cmap='seismic', shading='auto')
    cbar1= fig.colorbar(axp4, ax=ax[1,1])
    ax[1,1].set_xlabel('x')
    ax[1,1].set_ylabel('y')
    ax[1,1].set_title('Exact $E_{rz}$')
    ax[1,1].set_aspect('equal')

    # Draw err
    axp5= ax[1,2].pcolor(x_grid, y_grid, err_Erz, cmap='seismic', shading='auto')
    cbar1= fig.colorbar(axp5, ax=ax[1,2])
    ax[1,2].set_xlabel('x')
    ax[1,2].set_ylabel('y')
    ax[1,2].set_title('Error $E_{rz}$')
    ax[1,2].set_aspect('equal')



    l2_difference_Erz = dde.metrics.l2_relative_error(real_u_values, pred_Erz)
    l2_difference_Eiz = dde.metrics.l2_relative_error(imaginary_u_values, pred_Eiz)

    print("L2 relative error in Erz:", l2_difference_Erz)
    print("L2 relative error in Eiz:", l2_difference_Eiz)
    logger.info(fr"L2 relative error in Erz:{l2_difference_Erz}")
    logger.info(fr"L2 relative error in Eiz:{l2_difference_Eiz}")
    # fig.suptitle(fr"$\epsilon_r = {eps:.3f}$", fontsize=20)
    plt.savefig(fr"{path}/solutions_eps_{eps}.pdf", format='pdf', bbox_inches='tight')

    plt.tight_layout()
    plt.show()

