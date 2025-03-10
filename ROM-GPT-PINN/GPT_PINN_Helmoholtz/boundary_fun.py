import torch
import math
import numpy as np
torch.set_default_dtype(torch.float64)

L = 2.0   # Length of rectangle (R) 
l = 2.0   # Width of rectangle (R)

# We choose l=L to have a square 

# Bounds of x, y

x_lower = -L/2.0
x_upper =  L/2.0
y_lower = -l/2.0
y_upper =  l/2.0
mu_r = 1.0    
eps1_r = 1.0    # Real part of electric permittivity outside the disk
eps1_i = 0.0    # Imaginary part of electric permittivity outside the disk


Z_0 = 1.0    # In vacuum (the incident plane wave is injected in vacuum)

v_0_1 = 0.3  # Velocity 1 (outside the disk) 

freq  = 0.3  # 0.9 GHz = 900 Mhz

lam1  = v_0_1/freq # Wave length 1 (outside the disk)

omk   = (2.0*math.pi*freq)/v_0_1 # Pulsation

kap   = 1.0/(omk*mu_r) # Constant used in the definition of the PDE 

ampE  = 1.0 # Amplitude of the electric field
ampH  = ampE/omk # Amplitude of the magnetic field
kx1 = omk   # outside the disk

be = torch

#-----------------------#
# Real part of E and H
#-----------------------#

def EHx_abc_r_(x, y,dEz_i_x,dEz_i_y):

    # Calculate normal outgoing vector n=(nx,ny,0) depending on whatever the point (x,y) belongs to the boundary or not
    # x[:,0:1] refers to x coordinate and x[:,1;2] refers to y coordinate like in the defition of the PDE residual
    
    nx  = 0.0  
    ny  = 0.0
   
    nx  = be.where(x[:,0:1] == x_lower, -1.0,  nx)
    nx  = be.where(x[:,0:1] == x_upper,  1.0,  nx)

    ny  = be.where(x[:,1:2] == y_lower, -1.0,  ny)
    ny  = be.where(x[:,1:2] == y_upper,  1.0,  ny)

    # Calculate n x Er : nEx is the component along the x-axis, nEy is the component along the y-axis, nEz=0.0 (Annexe A.1)
    # y[:,0:1] refers to Erz and y[:,1:2] refers to Eiz

    nEx =  ny * y[:,0:1] 
    nEy = -nx * y[:,0:1]

    # Compute H_r from E_i

    # dEz_i_x = dde.grad.jacobian(y, x, i=1, j=0) # Calculate dEz_i/dx
    # dEz_i_y = dde.grad.jacobian(y, x, i=1, j=1) # Calculate dEz_i/dy
    
    # A little reminder : kap = 1/(mu_r*omk) 
    
    Hx =  -kap*dEz_i_y # Hrx function of Ezi Equation (36.1)
    Hy =   kap*dEz_i_x # Hry function of Ezi Equation (36.2)
     
    # Calculate Hr x n : nHz is the component along the z-axis, nHx=nHy=0.0 (Annexe A.1)
   
    nHz = -nx*Hy + ny*Hx

    # Calculate n x (Hr x n) : nHxn is the component along the x-axis, nHyn is the component along the y-axis, nHzn=0.0 (Annexe A.1)
    
    nHxn =  ny*nHz 
    nHyn = -nx*nHz 
    
    # Calculate the left side of the first equation of Equation (33)
    
    rEHx = nEx - Z_0*nHxn  # along the x-axis
    rEHy = nEy - Z_0*nHyn  # along the y-axis
    
    # Components of the incident plane wave, wave vector has an only component along the x-axis
    
    Ezinc   =   ampE*be.cos(kx1*x[:,0:1])
    Hyinc   =  -ampH*kx1*be.cos(kx1*x[:,0:1]) 
    
    # Calculate n x Erinc : nExinc is the component along the x-axis, nEyinc is the component along the y-axis, nEzinc=0.0 (Annexe A.1)

    nExinc  =  ny*Ezinc
    nEyinc  = -nx*Ezinc 
    
    # Calculate Hrinc x n : nHzinc is the component along the z-axis, nHxinc=nHyinc=0.0 (Annexe A.1)
    
    nHzinc  = -nx*Hyinc 

    # Calculate n x (Hrinc x n) : nHxninc is the component along the x-axis, nHyninc is the component along the y-axis, nHzninc=0.0 (Annexe A.1)
    
    nHxninc =  ny*nHzinc
    nHyninc = -nx*nHzinc
    
    # Calculate the right side of the first equation of Equation (33)

    rEHxinc = nExinc - Z_0*nHxninc  # along the x-axis
    rEHyinc = nEyinc - Z_0*nHyninc  # along the y-axis
    
    return rEHx - rEHxinc

def EHx_abc_r_grad(x, y,N_BC, dEz_i_x,dEz_i_y,c):

    # Calculate normal outgoing vector n=(nx,ny,0) depending on whatever the point (x,y) belongs to the boundary or not
    # x[:,0:1] refers to x coordinate and x[:,1;2] refers to y coordinate like in the defition of the PDE residual
    result = torch.zeros_like(c)
    nx  = 0.0  
    ny  = 0.0
   
    nx  = be.where(x[:,0:1] == x_lower, -1.0,  nx)
    nx  = be.where(x[:,0:1] == x_upper,  1.0,  nx)

    ny  = be.where(x[:,1:2] == y_lower, -1.0,  ny)
    ny  = be.where(x[:,1:2] == y_upper,  1.0,  ny)

    # Calculate n x Er : nEx is the component along the x-axis, nEy is the component along the y-axis, nEz=0.0 (Annexe A.1)
    # y[:,0:1] refers to Erz and y[:,1:2] refers to Eiz

    nEx =  ny * (torch.matmul(y, c[[0], :].transpose(0, 1)))
    nEy = -nx * (torch.matmul(y, c[[0], :].transpose(0, 1)))

    # Compute H_r from E_i

    # dEz_i_x = dde.grad.jacobian(y, x, i=1, j=0) # Calculate dEz_i/dx
    # dEz_i_y = dde.grad.jacobian(y, x, i=1, j=1) # Calculate dEz_i/dy
    
    # A little reminder : kap = 1/(mu_r*omk) 
    
    Hx =  -kap*(torch.matmul(dEz_i_y, c[[1], :].transpose(0, 1))) # Hrx function of Ezi Equation (36.1)
    Hy =   kap*(torch.matmul(dEz_i_x, c[[1], :].transpose(0, 1))) # Hry function of Ezi Equation (36.2)
     
    # Calculate Hr x n : nHz is the component along the z-axis, nHx=nHy=0.0 (Annexe A.1)
   
    nHz = -nx*Hy + ny*Hx

    # Calculate n x (Hr x n) : nHxn is the component along the x-axis, nHyn is the component along the y-axis, nHzn=0.0 (Annexe A.1)
    
    nHxn =  ny*nHz 
    nHyn = -nx*nHz 
    
    # Calculate the left side of the first equation of Equation (33)
    
    rEHx = nEx - Z_0*nHxn  # along the x-axis
    rEHy = nEy - Z_0*nHyn  # along the y-axis
    
    # Components of the incident plane wave, wave vector has an only component along the x-axis
    
    Ezinc   =   ampE*be.cos(kx1*x[:,0:1])
    Hyinc   =  -ampH*kx1*be.cos(kx1*x[:,0:1]) 
    
    # Calculate n x Erinc : nExinc is the component along the x-axis, nEyinc is the component along the y-axis, nEzinc=0.0 (Annexe A.1)

    nExinc  =  ny*Ezinc
    nEyinc  = -nx*Ezinc 
    
    # Calculate Hrinc x n : nHzinc is the component along the z-axis, nHxinc=nHyinc=0.0 (Annexe A.1)
    
    nHzinc  = -nx*Hyinc 

    # Calculate n x (Hrinc x n) : nHxninc is the component along the x-axis, nHyninc is the component along the y-axis, nHzninc=0.0 (Annexe A.1)
    
    nHxninc =  ny*nHzinc
    nHyninc = -nx*nHzinc
    
    # Calculate the right side of the first equation of Equation (33)

    rEHxinc = nExinc - Z_0*nHxninc  # along the x-axis
    rEHyinc = nEyinc - Z_0*nHyninc  # along the y-axis
    first_product = rEHx - rEHxinc
    second_product1 = ny * y
    second_product2 = nx*ny*kap*dEz_i_x + ny*ny*kap*dEz_i_y
    result[0, :] = torch.mul(2/N_BC, torch.sum(torch.mul(first_product, second_product1), dim=0))
    result[1, :] = torch.mul(2/N_BC, torch.sum(torch.mul(first_product, second_product2), dim=0))

    return result

def EHy_abc_r_(x, y,dEz_i_x,dEz_i_y):
    
    # Calculate normal outgoing vector n=(nx,ny,0) depending on whatever the point (x,y) belongs to the boundary or not
    # x[:,0:1] refers to x coordinate and x[:,1;2] refers to y coordinate like in the defition of the PDE residual
    
    nx  = 0.0  
    ny  = 0.0

    nx  = be.where(x[:,0:1] == x_lower, -1.0,  nx)
    nx  = be.where(x[:,0:1] == x_upper,  1.0,  nx)

    ny  = be.where(x[:,1:2] == y_lower, -1.0,  ny)
    ny  = be.where(x[:,1:2] == y_upper,  1.0,  ny)

    # Calculate n x Er : nEx is the component along the x-axis, nEy is the component along the y-axis, nEz=0.0 (Annexe A.1)
    # y[:,0:1] refers to Erz and y[:,1:2] refers to Eiz

    nEx =  ny * y[:,0:1] 
    nEy = -nx * y[:,0:1]

    # Compute H_r from E_i

    # dEz_i_x = dde.grad.jacobian(y, x, i=1, j=0) # Calculate dEz_i/dx
    # dEz_i_y = dde.grad.jacobian(y, x, i=1, j=1) # Calculate dEz_i/dy
    
    # A little reminder : kap = 1/(mu_r*omk) 
    
    Hx =  -kap*dEz_i_y # Hrx function of Ezi Equation (36.1)
    Hy =   kap*dEz_i_x # Hry function of Ezi Equation (36.2)
     
    # Calculate Hr x n : nHz is the component along the z-axis, nHx=nHy=0.0 (Annexe A.1)
   
    nHz = -nx*Hy + ny*Hx

    # Calculate n x (Hr x n) : nHxn is the component along the x-axis, nHyn is the component along the y-axis, nHzn=0.0 (Annexe A.1)
    
    nHxn =  ny*nHz 
    nHyn = -nx*nHz 
    
    # Calculate the left side of the first equation of Equation (33)
    
    rEHx = nEx - Z_0*nHxn  # along the x-axis
    rEHy = nEy - Z_0*nHyn  # along the y-axis
    
    # Components (real part) of the incident plane wave, wave vector has an only component along the x-axis
    
    Ezinc   =   ampE*be.cos(kx1*x[:,0:1])
    Hyinc   =  -ampH*kx1*be.cos(kx1*x[:,0:1]) 
    
    # Calculate n x Erinc : nExinc is the component along the x-axis, nEyinc is the component along the y-axis, nEzinc=0.0 (Annexe A.1)

    nExinc  =  ny*Ezinc
    nEyinc  = -nx*Ezinc 
    
    # Calculate Hrinc x n : nHzinc is the component along the z-axis, nHxinc=nHyinc=0.0 (Annexe A.1)
    
    nHzinc  = -nx*Hyinc 

    # Calculate n x (Hrinc x n) : nHxninc is the component along the x-axis, nHyninc is the component along the y-axis, nHzninc=0.0 (Annexe A.1)
    
    nHxninc =  ny*nHzinc
    nHyninc = -nx*nHzinc
    
    # Calculate the right side of the first equation of Equation (33)

    rEHxinc = nExinc - Z_0*nHxninc  # along the x-axis
    rEHyinc = nEyinc - Z_0*nHyninc  # along the y-axis
    
   
    return rEHy - rEHyinc

def EHy_abc_r_grad(x, y,N_BC, dEz_i_x,dEz_i_y,c):
    
    # Calculate normal outgoing vector n=(nx,ny,0) depending on whatever the point (x,y) belongs to the boundary or not
    # x[:,0:1] refers to x coordinate and x[:,1;2] refers to y coordinate like in the defition of the PDE residual
    result = torch.zeros_like(c)
    nx  = 0.0  
    ny  = 0.0

    nx  = be.where(x[:,0:1] == x_lower, -1.0,  nx)
    nx  = be.where(x[:,0:1] == x_upper,  1.0,  nx)

    ny  = be.where(x[:,1:2] == y_lower, -1.0,  ny)
    ny  = be.where(x[:,1:2] == y_upper,  1.0,  ny)

    # Calculate n x Er : nEx is the component along the x-axis, nEy is the component along the y-axis, nEz=0.0 (Annexe A.1)
    # y[:,0:1] refers to Erz and y[:,1:2] refers to Eiz

    nEx =  ny * (torch.matmul(y, c[[0], :].transpose(0, 1)))
    nEy = -nx * (torch.matmul(y, c[[0], :].transpose(0, 1)))

    # Compute H_r from E_i

    # dEz_i_x = dde.grad.jacobian(y, x, i=1, j=0) # Calculate dEz_i/dx
    # dEz_i_y = dde.grad.jacobian(y, x, i=1, j=1) # Calculate dEz_i/dy
    
    # A little reminder : kap = 1/(mu_r*omk) 
    
    Hx =  -kap*(torch.matmul(dEz_i_y, c[[1], :].transpose(0, 1))) # Hrx function of Ezi Equation (36.1)
    Hy =   kap*(torch.matmul(dEz_i_x, c[[1], :].transpose(0, 1))) # Hry function of Ezi Equation (36.2)
     
    # Calculate Hr x n : nHz is the component along the z-axis, nHx=nHy=0.0 (Annexe A.1)
   
    nHz = -nx*Hy + ny*Hx

    # Calculate n x (Hr x n) : nHxn is the component along the x-axis, nHyn is the component along the y-axis, nHzn=0.0 (Annexe A.1)
    
    nHxn =  ny*nHz 
    nHyn = -nx*nHz 
    
    # Calculate the left side of the first equation of Equation (33)
    
    rEHx = nEx - Z_0*nHxn  # along the x-axis
    rEHy = nEy - Z_0*nHyn  # along the y-axis
    
    # Components (real part) of the incident plane wave, wave vector has an only component along the x-axis
    
    Ezinc   =   ampE*be.cos(kx1*x[:,0:1])
    Hyinc   =  -ampH*kx1*be.cos(kx1*x[:,0:1]) 
    
    # Calculate n x Erinc : nExinc is the component along the x-axis, nEyinc is the component along the y-axis, nEzinc=0.0 (Annexe A.1)

    nExinc  =  ny*Ezinc
    nEyinc  = -nx*Ezinc 
    
    # Calculate Hrinc x n : nHzinc is the component along the z-axis, nHxinc=nHyinc=0.0 (Annexe A.1)
    
    nHzinc  = -nx*Hyinc 

    # Calculate n x (Hrinc x n) : nHxninc is the component along the x-axis, nHyninc is the component along the y-axis, nHzninc=0.0 (Annexe A.1)
    
    nHxninc =  ny*nHzinc
    nHyninc = -nx*nHzinc
    
    # Calculate the right side of the first equation of Equation (33)

    rEHxinc = nExinc - Z_0*nHxninc  # along the x-axis
    rEHyinc = nEyinc - Z_0*nHyninc  # along the y-axis

    first_product = rEHy - rEHyinc
    second_product1 = -nx * y
    second_product2 = -nx * nx * kap * dEz_i_x - nx * ny * kap * dEz_i_y
    result[0, :] = torch.mul(2 / N_BC, torch.sum(torch.mul(first_product, second_product1), dim=0))
    result[1, :] = torch.mul(2 / N_BC, torch.sum(torch.mul(first_product, second_product2), dim=0))
   
    return result
#-----------------------#
# Imaginary part of E and H
#-----------------------#

def EHx_abc_i_(x, y,dEz_r_x,dEz_r_y):
    
    # Calculate normal outgoing vector n=(nx,ny,0) depending on whatever the point (x,y) belongs to the boundary or not
    # x[:,0:1] refers to x coordinate and x[:,1;2] refers to y coordinate like in the defition of the PDE residual

    nx = 0.0
    ny = 0.0
   
    nx = be.where(x[:,0:1] == x_lower, -1.0,  nx)
    nx = be.where(x[:,0:1] == x_upper,  1.0,  nx)

    ny = be.where(x[:,1:2] == y_lower, -1.0,  ny)
    ny = be.where(x[:,1:2] == y_upper,  1.0,  ny)

    # Calculate n x Er : nEx is the component along the x-axis, nEy is the component along the y-axis, nEz=0.0 (Annexe A.1)
    # y[:,0:1] refers to Erz and y[:,1:2] refers to Eiz

    nEx =  ny * y[:,1:2]
    nEy = -nx * y[:,1:2]


    # Compute H_i from E_r
    
    # dEz_r_x = dde.grad.jacobian(y, x, i=0, j=0)  # Calculate dEz_r/dx
    # dEz_r_y = dde.grad.jacobian(y, x, i=0, j=1)  # Calculate dEz_r/dy
    
    # A little reminder : kap = 1/(mu_r*omk)
   
    Hx =   kap*dEz_r_y   # Hix function of Ezr Equation (36.3)
    Hy =  -kap*dEz_r_x  # Hix function of Ezr Equation (36.4)
    
    # Calculate Hi x n : nHz is the component along the z-axis, nHx=nHy=0.0 (Annexe A.1)
   
    nHz = -nx*Hy + ny*Hx

    # Calculate n x (Hi x n) : nHxn is the component along the x-axis, nHyn is the component along the y-axis, nHzn=0.0 (Annexe A.1)
    
    nHxn =  ny*nHz 
    nHyn = -nx*nHz 
    
    # Calculate the left side of the second equation of Equation (33)
   
    rEHx = nEx - Z_0*nHxn   # along the x-axis
    rEHy = nEy - Z_0*nHyn   # along the y-axis
    
 
    # Components (imaginary part) of the incident plane wave, wave vector has an only component along the x-axis
   
    Ezinc =   ampE*be.sin(-kx1*x[:,0:1])
    Hyinc =  -ampH*kx1*be.sin(-kx1*x[:,0:1]) 
    
  
    # Calculate n x Erinc : nExinc is the component along the x-axis, nEyinc is the component along the y-axis, nEzinc=0.0 (Annexe A.1)
  
    nExinc =  ny*Ezinc
    nEyinc = -nx*Ezinc 
    
    # Calculate Hrinc x n : nHzinc is the component along the z-axis, nHxinc=nHyinc=0.0 (Annexe A.1)
    
    nHzinc = -nx*Hyinc 

    # Calculate n x (Hrinc x n) : nHxninc is the component along the x-axis, nHyninc is the component along the y-axis, nHzninc=0.0 (Annexe A.1)
    
    nHxninc =  ny*nHzinc
    nHyninc = -nx*nHzinc
    
    # Calculate the right side of the second equation of Equation (33)
    
    rEHxinc = nExinc - Z_0*nHxninc  # along the x-axis
    rEHyinc = nEyinc - Z_0*nHyninc  # along the y-axis
    
    return rEHx - rEHxinc

def EHx_abc_i_grad(x, y, N_BC, dEz_r_x,dEz_r_y, c):

    result = torch.zeros_like(c)
    # Calculate normal outgoing vector n=(nx,ny,0) depending on whatever the point (x,y) belongs to the boundary or not
    # x[:,0:1] refers to x coordinate and x[:,1;2] refers to y coordinate like in the defition of the PDE residual

    nx = 0.0
    ny = 0.0
   
    nx = be.where(x[:,0:1] == x_lower, -1.0,  nx)
    nx = be.where(x[:,0:1] == x_upper,  1.0,  nx)

    ny = be.where(x[:,1:2] == y_lower, -1.0,  ny)
    ny = be.where(x[:,1:2] == y_upper,  1.0,  ny)

    # Calculate n x Er : nEx is the component along the x-axis, nEy is the component along the y-axis, nEz=0.0 (Annexe A.1)
    # y[:,0:1] refers to Erz and y[:,1:2] refers to Eiz

    nEx =  ny *(torch.matmul(y, c[[1], :].transpose(0, 1)))
    nEy = -nx * (torch.matmul(y, c[[1], :].transpose(0, 1)))


    # Compute H_i from E_r
    
    # dEz_r_x = dde.grad.jacobian(y, x, i=0, j=0)  # Calculate dEz_r/dx
    # dEz_r_y = dde.grad.jacobian(y, x, i=0, j=1)  # Calculate dEz_r/dy
    
    # A little reminder : kap = 1/(mu_r*omk)
   
    Hx =   kap*(torch.matmul(dEz_r_y, c[[0], :].transpose(0, 1)))   # Hix function of Ezr Equation (36.3)
    Hy =  -kap*(torch.matmul(dEz_r_x, c[[0], :].transpose(0, 1)))  # Hix function of Ezr Equation (36.4)
    
    # Calculate Hi x n : nHz is the component along the z-axis, nHx=nHy=0.0 (Annexe A.1)
   
    nHz = -nx*Hy + ny*Hx

    # Calculate n x (Hi x n) : nHxn is the component along the x-axis, nHyn is the component along the y-axis, nHzn=0.0 (Annexe A.1)
    
    nHxn =  ny*nHz 
    nHyn = -nx*nHz 
    
    # Calculate the left side of the second equation of Equation (33)
   
    rEHx = nEx - Z_0*nHxn   # along the x-axis
    rEHy = nEy - Z_0*nHyn   # along the y-axis
    
 
    # Components (imaginary part) of the incident plane wave, wave vector has an only component along the x-axis
   
    Ezinc =   ampE*be.sin(-kx1*x[:,0:1])
    Hyinc =  -ampH*kx1*be.sin(-kx1*x[:,0:1]) 
    
  
    # Calculate n x Erinc : nExinc is the component along the x-axis, nEyinc is the component along the y-axis, nEzinc=0.0 (Annexe A.1)
  
    nExinc =  ny*Ezinc
    nEyinc = -nx*Ezinc 
    
    # Calculate Hrinc x n : nHzinc is the component along the z-axis, nHxinc=nHyinc=0.0 (Annexe A.1)
    
    nHzinc = -nx*Hyinc 

    # Calculate n x (Hrinc x n) : nHxninc is the component along the x-axis, nHyninc is the component along the y-axis, nHzninc=0.0 (Annexe A.1)
    
    nHxninc =  ny*nHzinc
    nHyninc = -nx*nHzinc
    
    # Calculate the right side of the second equation of Equation (33)
    
    rEHxinc = nExinc - Z_0*nHxninc  # along the x-axis
    rEHyinc = nEyinc - Z_0*nHyninc  # along the y-axis

    first_product = rEHx - rEHxinc
    second_product2 = ny * y
    second_product1 = -nx * ny * kap * dEz_r_x - ny * ny * kap * dEz_r_y
    result[0, :] = torch.mul(2 / N_BC, torch.sum(torch.mul(first_product, second_product1), dim=0))
    result[1, :] = torch.mul(2 / N_BC, torch.sum(torch.mul(first_product, second_product2), dim=0))


    return result

def EHy_abc_i_(x, y,dEz_r_x,dEz_r_y):
   

    # Calculate normal outgoing vector n=(nx,ny,0) depending on whatever the point (x,y) belongs to the boundary or not
    # x[:,0:1] refers to x coordinate and x[:,1;2] refers to y coordinate like in the defition of the PDE residual

    nx = 0.0
    ny = 0.0
   
    nx = be.where(x[:,0:1] == x_lower, -1.0,  nx)
    nx = be.where(x[:,0:1] == x_upper,  1.0,  nx)

    ny = be.where(x[:,1:2] == y_lower, -1.0,  ny)
    ny = be.where(x[:,1:2] == y_upper,  1.0,  ny)

    # Calculate n x Er : nEx is the component along the x-axis, nEy is the component along the y-axis, nEz=0.0 (Annexe A.1)
    # y[:,0:1] refers to Erz and y[:,1:2] refers to Eiz

    nEx =  ny * y[:,1:2]
    nEy = -nx * y[:,1:2]


    # Compute H_i from E_r
    
    # dEz_r_x = dde.grad.jacobian(y, x, i=0, j=0)  # Calculate dEz_r/dx
    # dEz_r_y = dde.grad.jacobian(y, x, i=0, j=1)  # Calculate dEz_r/dy
    
    # A little reminder : kap = 1/(mu_r*omk)
   
    Hx =   kap*dEz_r_y   # Hix function of Ezr Equation (36.3)
    Hy =  -kap*dEz_r_x  # Hix function of Ezr Equation (36.4)
    
    # Calculate Hi x n : nHz is the component along the z-axis, nHx=nHy=0.0 (Annexe A.1)
   
    nHz = -nx*Hy + ny*Hx

    # Calculate n x (Hi x n) : nHxn is the component along the x-axis, nHyn is the component along the y-axis, nHzn=0.0 (Annexe A.1)
    
    nHxn =  ny*nHz 
    nHyn = -nx*nHz 
    
    # Calculate the left side of the second equation of Equation (33)
   
    rEHx = nEx - Z_0*nHxn   # along the x-axis
    rEHy = nEy - Z_0*nHyn   # along the y-axis
    
 
    # Components (imaginary part) of the incident plane wave, wave vector has an only component along the x-axis
   
    Ezinc =   ampE*be.sin(-kx1*x[:,0:1])
    Hyinc =  -ampH*kx1*be.sin(-kx1*x[:,0:1]) 
    
  
    # Calculate n x Erinc : nExinc is the component along the x-axis, nEyinc is the component along the y-axis, nEzinc=0.0 (Annexe A.1)
  
    nExinc =  ny*Ezinc
    nEyinc = -nx*Ezinc 
    
    # Calculate Hrinc x n : nHzinc is the component along the z-axis, nHxinc=nHyinc=0.0 (Annexe A.1)
    
    nHzinc = -nx*Hyinc 

    # Calculate n x (Hrinc x n) : nHxninc is the component along the x-axis, nHyninc is the component along the y-axis, nHzninc=0.0 (Annexe A.1)
    
    nHxninc =  ny*nHzinc
    nHyninc = -nx*nHzinc
    
    # Calculate the right side of the second equation of Equation (33)
    
    rEHxinc = nExinc - Z_0*nHxninc  # along the x-axis
    rEHyinc = nEyinc - Z_0*nHyninc  # along the y-axis
    
    return rEHy - rEHyinc

def EHy_abc_i_grad(x, y,N_BC, dEz_r_x,dEz_r_y,c):
   
    result = torch.zeros_like(c)
    # Calculate normal outgoing vector n=(nx,ny,0) depending on whatever the point (x,y) belongs to the boundary or not
    # x[:,0:1] refers to x coordinate and x[:,1;2] refers to y coordinate like in the defition of the PDE residual

    nx = 0.0
    ny = 0.0
   
    nx = be.where(x[:,0:1] == x_lower, -1.0,  nx)
    nx = be.where(x[:,0:1] == x_upper,  1.0,  nx)

    ny = be.where(x[:,1:2] == y_lower, -1.0,  ny)
    ny = be.where(x[:,1:2] == y_upper,  1.0,  ny)

    # Calculate n x Er : nEx is the component along the x-axis, nEy is the component along the y-axis, nEz=0.0 (Annexe A.1)
    # y[:,0:1] refers to Erz and y[:,1:2] refers to Eiz

    nEx =  ny * (torch.matmul(y, c[[1], :].transpose(0, 1)))
    nEy = -nx * (torch.matmul(y, c[[1], :].transpose(0, 1)))


    # Compute H_i from E_r
    
    # dEz_r_x = dde.grad.jacobian(y, x, i=0, j=0)  # Calculate dEz_r/dx
    # dEz_r_y = dde.grad.jacobian(y, x, i=0, j=1)  # Calculate dEz_r/dy
    
    # A little reminder : kap = 1/(mu_r*omk)
   
    Hx =   kap*torch.matmul(dEz_r_y, c[[0], :].transpose(0, 1))   # Hix function of Ezr Equation (36.3)
    Hy =  -kap*torch.matmul(dEz_r_x, c[[0], :].transpose(0, 1))  # Hix function of Ezr Equation (36.4)
    
    # Calculate Hi x n : nHz is the component along the z-axis, nHx=nHy=0.0 (Annexe A.1)
   
    nHz = -nx*Hy + ny*Hx

    # Calculate n x (Hi x n) : nHxn is the component along the x-axis, nHyn is the component along the y-axis, nHzn=0.0 (Annexe A.1)
    
    nHxn =  ny*nHz 
    nHyn = -nx*nHz 
    
    # Calculate the left side of the second equation of Equation (33)
   
    rEHx = nEx - Z_0*nHxn   # along the x-axis
    rEHy = nEy - Z_0*nHyn   # along the y-axis
    
 
    # Components (imaginary part) of the incident plane wave, wave vector has an only component along the x-axis
   
    Ezinc =   ampE*be.sin(-kx1*x[:,0:1])
    Hyinc =  -ampH*kx1*be.sin(-kx1*x[:,0:1]) 
    
  
    # Calculate n x Erinc : nExinc is the component along the x-axis, nEyinc is the component along the y-axis, nEzinc=0.0 (Annexe A.1)
  
    nExinc =  ny*Ezinc
    nEyinc = -nx*Ezinc 
    
    # Calculate Hrinc x n : nHzinc is the component along the z-axis, nHxinc=nHyinc=0.0 (Annexe A.1)
    
    nHzinc = -nx*Hyinc 

    # Calculate n x (Hrinc x n) : nHxninc is the component along the x-axis, nHyninc is the component along the y-axis, nHzninc=0.0 (Annexe A.1)
    
    nHxninc =  ny*nHzinc
    nHyninc = -nx*nHzinc
    
    # Calculate the right side of the second equation of Equation (33)
    
    rEHxinc = nExinc - Z_0*nHxninc  # along the x-axis
    rEHyinc = nEyinc - Z_0*nHyninc  # along the y-axis

    first_product = rEHy - rEHyinc
    second_product2 = -nx * y
    second_product1 = nx * nx * kap * dEz_r_x + nx * ny * kap * dEz_r_y
    result[0, :] = torch.mul(2 / N_BC, torch.sum(torch.mul(first_product, second_product1), dim=0))
    result[1, :] = torch.mul(2 / N_BC, torch.sum(torch.mul(first_product, second_product2), dim=0))

    return result

# Boundary conditions