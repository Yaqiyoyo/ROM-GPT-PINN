import numpy as np
import sys
import matplotlib.pyplot as plt
import pylab as pylabplt
from scipy.interpolate import griddata
import time
import math
import scipy.special as sp

# Choose DeepXde backend

import os
os.environ["DDEBACKEND"] = "pytorch"

# Import DeepXDE libraries

import deepxde as dde
from deepxde.backend import backend_name, tf, torch
from deepxde import utils
import torch.nn as nn

if backend_name == "tensorflow" or backend_name == "tensorflow.compat.v1":
   be = tf
elif backend_name == "pytorch":
   be = torch

# Default configuration for floating-point numbers

dde.config.set_default_float("float64")

class NN(nn.Module): 
    def __init__(self, mu):
        super().__init__()

        self.x_lower = 0
        self.x_upper =  1
        self.y_lower = 0
        self.y_upper = 1
        self.z_lower = 0
        self.z_upper = 1

        # Geometry definition

        outer = dde.geometry.Cuboid(xmin=[self.x_lower,self.y_lower, self.z_lower], xmax=[self.x_upper,self.y_upper, self.z_upper])
        self.geom = outer
        self.mu= mu


    def pde(self, x, A):
 
        """
            Parameters
        - The first argument to pde is a 3-dimensional vector where:
                - the  first component x[:,0] is the x coordinate
                - the second component x[:,1] is the y coordinate
                - the third component x[:,2] is the z coordinate
                
            - The second argument to pde is the network output and is 
            a 1-dimensional vector where:
                - v[:,0] is Az

            
        """

        A_z = A[:, 0:1]
        J_z = torch.sin(math.pi * x[:, 0:1]) * torch.sin(math.pi * x[:, 1:2]) * torch.sin(math.pi * x[:, 2:3])

        dA_z_xx = dde.grad.hessian(A_z, x, i=0, j=0)
        dA_z_yy = dde.grad.hessian(A_z, x, i=1, j=1)
        dA_z_zz = dde.grad.hessian(A_z, x, i=2, j=2)

        f_z = dA_z_xx + dA_z_yy + dA_z_zz + self.mu * J_z
    
        return f_z






    


       


    

