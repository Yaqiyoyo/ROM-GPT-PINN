import numpy as np

def GetPredictData(nbx, nby):
    
    x_lower = -1.0
    x_upper =  1.0
    y_lower = -1.0
    y_upper =  1.0

 

    xc = np.linspace(x_lower, x_upper, nbx)
    yc = np.linspace(y_lower, y_upper, nby)

    x_grid, y_grid = np.meshgrid(xc, yc)

    # Stack x, y coordinates

    xy_grid = np.vstack((np.ravel(x_grid),np.ravel(y_grid))).T

    return xy_grid, x_grid, y_grid

