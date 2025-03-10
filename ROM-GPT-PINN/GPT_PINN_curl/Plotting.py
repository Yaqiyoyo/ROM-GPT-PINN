import matplotlib.pyplot as plt
import numpy as np

# plt.style.use(['science', 'notebook'])
import deepxde as dde
from Log.log import logger

def plot_solution(predictions, exact, x_grid, y_grid, mu, path):
    path = path

    exact = np.nan_to_num(exact)
    # Extract real and imaginary parts of the values

    # absolute error
    err = np.abs(predictions - exact)


    # Draw the colorations of Eiz and Erz on the plan
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))



    # Draw Eiz_pred


    ax0=axes[0].pcolor(x_grid, y_grid, predictions.reshape(x_grid.shape), cmap='seismic', shading='auto')
    cbar0 = fig.colorbar(ax0, ax=axes[0])
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Predicted $A_z$ at $z=0.5$")
    axes[0].set_aspect('equal')


    # Draw Eiz_exact
    ax1=axes[1].pcolor(x_grid, y_grid, exact.reshape(x_grid.shape), cmap='seismic', shading='auto')
    cbar1 = fig.colorbar(ax1, ax=axes[1])
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Analytic $A_z$ at $z=0.5$")
    axes[1].set_aspect('equal')


    # Draw err
    ax2=axes[2].pcolor(x_grid, y_grid, err.reshape(x_grid.shape), cmap='seismic', shading='auto')
    cbar2 = fig.colorbar(ax2, ax=axes[2])
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_title("Error at $z=0.5$")
    axes[2].set_aspect('equal')



    l2_difference_Az = dde.metrics.l2_relative_error(exact, predictions)


    print("L2 relative error in Az:", l2_difference_Az)

    logger.info(fr"L2 relative error in Az:{l2_difference_Az}")


    plt.savefig(fr"{path}/solutions_mu_{mu:.3f}.pdf", format='pdf',bbox_inches='tight')

    plt.tight_layout()
    plt.show()

