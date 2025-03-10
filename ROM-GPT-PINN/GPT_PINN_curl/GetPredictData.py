import numpy as np

def GetPredictData(nbx, nby):
    
    x_lower = 0
    x_upper =  1.0
    y_lower = 0
    y_upper =  1.0
    z = 0.5


 

    xc = np.linspace(x_lower, x_upper, nbx)
    yc = np.linspace(y_lower, y_upper, nby)

    X, Y = np.meshgrid(xc, yc)
    Z = np.full_like(X, z)
    grid_points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

    # Stack x, y coordinates



    return X,Y,grid_points

