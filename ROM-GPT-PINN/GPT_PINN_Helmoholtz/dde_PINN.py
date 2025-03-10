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
    def __init__(self, epsilon):
        super().__init__()
        L = 2.0   # Length of rectangle (R) 
        l = 2.0   # Width of rectangle (R)
        self.x_lower = -L/2.0
        self.x_upper =  L/2.0
        self.y_lower = -l/2.0
        self.y_upper =  l/2.0
        self.R = 0.25  # Radius of circle (S)

        # Center of the circle

        self.Cx = 0.0
        self.Cy = 0.0

        # Geometry definition

        outer = dde.geometry.Rectangle(xmin=(self.x_lower,self.y_lower), xmax=(self.x_upper,self.y_upper))
        inter = dde.geometry.Disk([self.Cx, self.Cy], self.R)

        self.geom = outer

        self.eps1_r = 1.0    # Real part of electric permittivity outside the disk
        self.eps1_i = 0.0    # Imaginary part of electric permittivity outside the disk

        # self.eps2_r = 4.0     # Real part of electric permittivity inside the disk 
        self.eps2_r = epsilon
        self.eps2_i = 0.0     # Imaginary part of electric permittivity inside the disk

        mu_r = 1.0    

        self.Z_0 = 1.0    # In vacuum (the incident plane wave is injected in vacuum)

        v_0_1 = 0.3  # Velocity 1 (outside the disk) 
        v_0_2 = v_0_1/np.sqrt(mu_r*self.eps2_r) # Velocity 2 (inside the disk) if you change eps2_r, you must also change v_0_2 (in case eps2_r = 4.0, we have v_0_2 = 0.15)

        freq  = 0.3  # 0.9 GHz = 900 Mhz

        lam1  = v_0_1/freq # Wave length 1 (outside the disk)
        lam2  = v_0_2/freq # Wave length 2 (inside the disk)

        self.omk   = (2.0*math.pi*freq)/v_0_1 # Pulsation
        omk2  = (2.0*math.pi*freq)/v_0_2 # Pulsation

        self.kap   = 1.0/(self.omk*mu_r) # Constant used in the definition of the PDE 

        self.ampE  = 1.0 # Amplitude of the electric field
        self.ampH  = self.ampE/self.omk # Amplitude of the magnetic field
        self.kx1 = self.omk   # outside the disk
        self.kx2 = omk2  # inside the disk


    def pde(self, u, v):
 
        """
            Parameters
        - The first argument to pde is a 2-dimensional vector where:
                - the  first component u[:,0] is the x coordinate
                - the second component u[:,1] is the y coordinate 
                
            - The second argument to pde is the network output and is 
            a 2-dimensional vector where:
                - v[:,0] is Ez_r(x,y,z,omega) 
                - v[:,1] is Ez_i(x,y,z,omega) 
            
        """

        Ez_r, Ez_i = v[:,0:1], v[:,1:2]
        
        d2Ez_r_x2 = dde.grad.hessian(Ez_r, u, i=0, j=0) # d2Ez_r/dx2
        d2Ez_r_y2 = dde.grad.hessian(Ez_r, u, i=1, j=1) # d2Ez_r/dy2
        
        d2Ez_i_x2 = dde.grad.hessian(Ez_i, u, i=0, j=0) # d2Ez_i/dx2
        d2Ez_i_y2 = dde.grad.hessian(Ez_i, u, i=1, j=1) # d2Ez_i/dy2
    

        curl2E_z_r = - (d2Ez_r_x2 + d2Ez_r_y2)
        curl2E_z_i = - (d2Ez_i_x2 + d2Ez_i_y2)
        
        
        d2 = (u[:,0:1] - self.Cx)*(u[:,0:1] - self.Cx) + (u[:,1:2] - self.Cy)*(u[:,1:2] - self.Cy)
        d  = be.sqrt(d2) # Distance between a point X(x,y) and the center of the disk (Cx,Cy)
        cond = be.less(d[:], self.R)

    # A little reminder : kap = 1.0/(mu_r*omk)

    # Equation (31.1) with a leger modification in term of coefficients

        fEz_1 = self.omk*(self.eps1_r*Ez_r - self.eps1_i*Ez_i) - self.kap*curl2E_z_r  # outside the disk
        fEz_2 = self.omk*(self.eps2_r*Ez_r - self.eps2_i*Ez_i) - self.kap*curl2E_z_r  # inside the disk

        fEz = be.where(cond, fEz_2, fEz_1)

    # Equation (31.2) with a leger modification in term of coefficients

        gEz_1 =-self.omk*(self.eps1_r*Ez_i + self.eps1_i*Ez_r) + self.kap*curl2E_z_i  # outside the disk
        gEz_2 =-self.omk*(self.eps2_r*Ez_i + self.eps2_i*Ez_r) + self.kap*curl2E_z_i  # inside the disk
    
        gEz = be.where(cond, gEz_2, gEz_1)
    
        return [fEz, gEz]
    
    def boundary(self, x, on_boundary): 
        return on_boundary 
    

    def EHx_abc_r(self, x, y,_):

    # Calculate normal outgoing vector n=(nx,ny,0) depending on whatever the point (x,y) belongs to the boundary or not
    # x[:,0:1] refers to x coordinate and x[:,1;2] refers to y coordinate like in the defition of the PDE residual
    
        nx  = 0.0  
        ny  = 0.0
    
        nx  = be.where(x[:,0:1] == self.x_lower, -1.0,  nx)
        nx  = be.where(x[:,0:1] == self.x_upper,  1.0,  nx)

        ny  = be.where(x[:,1:2] == self.y_lower, -1.0,  ny)
        ny  = be.where(x[:,1:2] == self.y_upper,  1.0,  ny)

        # Calculate n x Er : nEx is the component along the x-axis, nEy is the component along the y-axis, nEz=0.0 (Annexe A.1)
        # y[:,0:1] refers to Erz and y[:,1:2] refers to Eiz

        nEx =  ny * y[:,0:1] 
        nEy = -nx * y[:,0:1]

        # Compute H_r from E_i

        dEz_i_x = dde.grad.jacobian(y, x, i=1, j=0) # Calculate dEz_i/dx
        dEz_i_y = dde.grad.jacobian(y, x, i=1, j=1) # Calculate dEz_i/dy
        
        # A little reminder : kap = 1/(mu_r*omk) 
        
        Hx =  -self.kap*dEz_i_y # Hrx function of Ezi Equation (36.1)
        Hy =   self.kap*dEz_i_x # Hry function of Ezi Equation (36.2)
        
        # Calculate Hr x n : nHz is the component along the z-axis, nHx=nHy=0.0 (Annexe A.1)
    
        nHz = -nx*Hy + ny*Hx

        # Calculate n x (Hr x n) : nHxn is the component along the x-axis, nHyn is the component along the y-axis, nHzn=0.0 (Annexe A.1)
        
        nHxn =  ny*nHz 
        nHyn = -nx*nHz 
        
        # Calculate the left side of the first equation of Equation (33)
        
        rEHx = nEx - self.Z_0*nHxn  # along the x-axis
        rEHy = nEy - self.Z_0*nHyn  # along the y-axis
        
        # Components of the incident plane wave, wave vector has an only component along the x-axis
        
        Ezinc   =   self.ampE*be.cos(self.kx1*x[:,0:1])
        Hyinc   =  -self.ampH*self.kx1*be.cos(self.kx1*x[:,0:1]) 
        
        # Calculate n x Erinc : nExinc is the component along the x-axis, nEyinc is the component along the y-axis, nEzinc=0.0 (Annexe A.1)

        nExinc  =  ny*Ezinc
        nEyinc  = -nx*Ezinc 
        
        # Calculate Hrinc x n : nHzinc is the component along the z-axis, nHxinc=nHyinc=0.0 (Annexe A.1)
        
        nHzinc  = -nx*Hyinc 

        # Calculate n x (Hrinc x n) : nHxninc is the component along the x-axis, nHyninc is the component along the y-axis, nHzninc=0.0 (Annexe A.1)
        
        nHxninc =  ny*nHzinc
        nHyninc = -nx*nHzinc
        
        # Calculate the right side of the first equation of Equation (33)

        rEHxinc = nExinc - self.Z_0*nHxninc  # along the x-axis
        rEHyinc = nEyinc - self.Z_0*nHyninc  # along the y-axis
        
        return rEHx - rEHxinc


    def EHy_abc_r(self, x, y,_):
        
        # Calculate normal outgoing vector n=(nx,ny,0) depending on whatever the point (x,y) belongs to the boundary or not
        # x[:,0:1] refers to x coordinate and x[:,1;2] refers to y coordinate like in the defition of the PDE residual
        
        nx  = 0.0  
        ny  = 0.0

        nx  = be.where(x[:,0:1] == self.x_lower, -1.0,  nx)
        nx  = be.where(x[:,0:1] == self.x_upper,  1.0,  nx)

        ny  = be.where(x[:,1:2] == self.y_lower, -1.0,  ny)
        ny  = be.where(x[:,1:2] == self.y_upper,  1.0,  ny)

        # Calculate n x Er : nEx is the component along the x-axis, nEy is the component along the y-axis, nEz=0.0 (Annexe A.1)
        # y[:,0:1] refers to Erz and y[:,1:2] refers to Eiz

        nEx =  ny * y[:,0:1] 
        nEy = -nx * y[:,0:1]

        # Compute H_r from E_i

        dEz_i_x = dde.grad.jacobian(y, x, i=1, j=0) # Calculate dEz_i/dx
        dEz_i_y = dde.grad.jacobian(y, x, i=1, j=1) # Calculate dEz_i/dy
        
        # A little reminder : kap = 1/(mu_r*omk) 
        
        Hx =  -self.kap*dEz_i_y # Hrx function of Ezi Equation (36.1)
        Hy =   self.kap*dEz_i_x # Hry function of Ezi Equation (36.2)
        
        # Calculate Hr x n : nHz is the component along the z-axis, nHx=nHy=0.0 (Annexe A.1)
    
        nHz = -nx*Hy + ny*Hx

        # Calculate n x (Hr x n) : nHxn is the component along the x-axis, nHyn is the component along the y-axis, nHzn=0.0 (Annexe A.1)
        
        nHxn =  ny*nHz 
        nHyn = -nx*nHz 
        
        # Calculate the left side of the first equation of Equation (33)
        
        rEHx = nEx - self.Z_0*nHxn  # along the x-axis
        rEHy = nEy - self.Z_0*nHyn  # along the y-axis
        
        # Components (real part) of the incident plane wave, wave vector has an only component along the x-axis
        
        Ezinc   =   self.ampE*be.cos(self.kx1*x[:,0:1])
        Hyinc   =  -self.ampH*self.kx1*be.cos(self.kx1*x[:,0:1]) 
        
        # Calculate n x Erinc : nExinc is the component along the x-axis, nEyinc is the component along the y-axis, nEzinc=0.0 (Annexe A.1)

        nExinc  =  ny*Ezinc
        nEyinc  = -nx*Ezinc 
        
        # Calculate Hrinc x n : nHzinc is the component along the z-axis, nHxinc=nHyinc=0.0 (Annexe A.1)
        
        nHzinc  = -nx*Hyinc 

        # Calculate n x (Hrinc x n) : nHxninc is the component along the x-axis, nHyninc is the component along the y-axis, nHzninc=0.0 (Annexe A.1)
        
        nHxninc =  ny*nHzinc
        nHyninc = -nx*nHzinc
        
        # Calculate the right side of the first equation of Equation (33)

        rEHxinc = nExinc - self.Z_0*nHxninc  # along the x-axis
        rEHyinc = nEyinc - self.Z_0*nHyninc  # along the y-axis
        
    
        return rEHy - rEHyinc


    #-----------------------#
    # Imaginary part of E and H
    #-----------------------#

    def EHx_abc_i(self, x, y,_):
        
        # Calculate normal outgoing vector n=(nx,ny,0) depending on whatever the point (x,y) belongs to the boundary or not
        # x[:,0:1] refers to x coordinate and x[:,1;2] refers to y coordinate like in the defition of the PDE residual

        nx = 0.0
        ny = 0.0
    
        nx = be.where(x[:,0:1] == self.x_lower, -1.0,  nx)
        nx = be.where(x[:,0:1] == self.x_upper,  1.0,  nx)

        ny = be.where(x[:,1:2] == self.y_lower, -1.0,  ny)
        ny = be.where(x[:,1:2] == self.y_upper,  1.0,  ny)

        # Calculate n x Er : nEx is the component along the x-axis, nEy is the component along the y-axis, nEz=0.0 (Annexe A.1)
        # y[:,0:1] refers to Erz and y[:,1:2] refers to Eiz

        nEx =  ny * y[:,1:2]
        nEy = -nx * y[:,1:2]


        # Compute H_i from E_r
        
        dEz_r_x = dde.grad.jacobian(y, x, i=0, j=0)  # Calculate dEz_r/dx
        dEz_r_y = dde.grad.jacobian(y, x, i=0, j=1)  # Calculate dEz_r/dy
        
        # A little reminder : kap = 1/(mu_r*omk)
    
        Hx =   self.kap*dEz_r_y   # Hix function of Ezr Equation (36.3)
        Hy =  -self.kap*dEz_r_x  # Hix function of Ezr Equation (36.4)
        
        # Calculate Hi x n : nHz is the component along the z-axis, nHx=nHy=0.0 (Annexe A.1)
    
        nHz = -nx*Hy + ny*Hx

        # Calculate n x (Hi x n) : nHxn is the component along the x-axis, nHyn is the component along the y-axis, nHzn=0.0 (Annexe A.1)
        
        nHxn =  ny*nHz 
        nHyn = -nx*nHz 
        
        # Calculate the left side of the second equation of Equation (33)
    
        rEHx = nEx - self.Z_0*nHxn   # along the x-axis
        rEHy = nEy - self.Z_0*nHyn   # along the y-axis
        
    
        # Components (imaginary part) of the incident plane wave, wave vector has an only component along the x-axis
    
        Ezinc =   self.ampE*be.sin(-self.kx1*x[:,0:1])
        Hyinc =  -self.ampH*self.kx1*be.sin(-self.kx1*x[:,0:1]) 
        
    
        # Calculate n x Erinc : nExinc is the component along the x-axis, nEyinc is the component along the y-axis, nEzinc=0.0 (Annexe A.1)
    
        nExinc =  ny*Ezinc
        nEyinc = -nx*Ezinc 
        
        # Calculate Hrinc x n : nHzinc is the component along the z-axis, nHxinc=nHyinc=0.0 (Annexe A.1)
        
        nHzinc = -nx*Hyinc 

        # Calculate n x (Hrinc x n) : nHxninc is the component along the x-axis, nHyninc is the component along the y-axis, nHzninc=0.0 (Annexe A.1)
        
        nHxninc =  ny*nHzinc
        nHyninc = -nx*nHzinc
        
        # Calculate the right side of the second equation of Equation (33)
        
        rEHxinc = nExinc - self.Z_0*nHxninc  # along the x-axis
        rEHyinc = nEyinc - self.Z_0*nHyninc  # along the y-axis
        
        return rEHx - rEHxinc


    def EHy_abc_i(self, x, y,_):
    
        
        # Calculate normal outgoing vector n=(nx,ny,0) depending on whatever the point (x,y) belongs to the boundary or not
        # x[:,0:1] refers to x coordinate and x[:,1;2] refers to y coordinate like in the defition of the PDE residual

        nx = 0.0
        ny = 0.0
    
        nx = be.where(x[:,0:1] == self.x_lower, -1.0,  nx)
        nx = be.where(x[:,0:1] == self.x_upper,  1.0,  nx)

        ny = be.where(x[:,1:2] == self.y_lower, -1.0,  ny)
        ny = be.where(x[:,1:2] == self.y_upper,  1.0,  ny)

        # Calculate n x Er : nEx is the component along the x-axis, nEy is the component along the y-axis, nEz=0.0 (Annexe A.1)
        # y[:,0:1] refers to Erz and y[:,1:2] refers to Eiz

        nEx =  ny * y[:,1:2]
        nEy = -nx * y[:,1:2]


        # Compute H_i from E_r
        
        dEz_r_x = dde.grad.jacobian(y, x, i=0, j=0)  # Calculate dEz_r/dx
        dEz_r_y = dde.grad.jacobian(y, x, i=0, j=1)  # Calculate dEz_r/dy
        
        # A little reminder : kap = 1/(mu_r*omk)
    
        Hx =   self.kap*dEz_r_y   # Hix function of Ezr Equation (36.3)
        Hy =  -self.kap*dEz_r_x  # Hix function of Ezr Equation (36.4)
        
        # Calculate Hi x n : nHz is the component along the z-axis, nHx=nHy=0.0 (Annexe A.1)
    
        nHz = -nx*Hy + ny*Hx

        # Calculate n x (Hi x n) : nHxn is the component along the x-axis, nHyn is the component along the y-axis, nHzn=0.0 (Annexe A.1)
        
        nHxn =  ny*nHz 
        nHyn = -nx*nHz 
        
        # Calculate the left side of the second equation of Equation (33)
    
        rEHx = nEx - self.Z_0*nHxn   # along the x-axis
        rEHy = nEy - self.Z_0*nHyn   # along the y-axis
        
    
        # Components (imaginary part) of the incident plane wave, wave vector has an only component along the x-axis
    
        Ezinc =   self.ampE*be.sin(-self.kx1*x[:,0:1])
        Hyinc =  -self.ampH*self.kx1*be.sin(-self.kx1*x[:,0:1]) 
        
    
        # Calculate n x Erinc : nExinc is the component along the x-axis, nEyinc is the component along the y-axis, nEzinc=0.0 (Annexe A.1)
    
        nExinc =  ny*Ezinc
        nEyinc = -nx*Ezinc 
        
        # Calculate Hrinc x n : nHzinc is the component along the z-axis, nHxinc=nHyinc=0.0 (Annexe A.1)
        
        nHzinc = -nx*Hyinc 

        # Calculate n x (Hrinc x n) : nHxninc is the component along the x-axis, nHyninc is the component along the y-axis, nHzninc=0.0 (Annexe A.1)
        
        nHxninc =  ny*nHzinc
        nHyninc = -nx*nHzinc
        
        # Calculate the right side of the second equation of Equation (33)
        
        rEHxinc = nExinc - self.Z_0*nHxninc  # along the x-axis
        rEHyinc = nEyinc - self.Z_0*nHyninc  # along the y-axis
        
        return rEHy - rEHyinc

    
    


       


    

