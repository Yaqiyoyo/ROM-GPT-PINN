import deepxde as dde
import torch



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float)

def autograd_calculations(xyz, P):
    xyz = xyz.to(device).float().requires_grad_()
    Pi = P(xyz).to(device).float()
    print(f"Shape of Pi: {Pi.shape}")
    print(f"Shape of xyz: {xyz.shape}")

    d2Az_x2 = dde.grad.hessian(Pi, xyz, i=0, j=0) # d2Az/dx2
    d2Az_y2 = dde.grad.hessian(Pi, xyz, i=1, j=1) # d2Az/dy2
    d2Az_z2 = dde.grad.hessian(Pi, xyz, i=2, j=2) # d2Az/dz2


    return Pi, d2Az_x2, d2Az_y2, d2Az_z2

def autograd_calculations_BC(xyz, P):
    xyz = xyz.to(device).float().requires_grad_()
    Pi = P(xyz).to(device).float()
    print(f"Shape of Pi: {Pi.shape}")
    print(f"Shape of xyz: {xyz.shape}")
    return Pi

"""
Model = torch.load('model_square_disk_heterogeneous-20000.pt')
w1 = Model['model_state_dict']['linears.0.weight'].detach().to(device).float()
w2 = Model['model_state_dict']['linears.1.weight'].detach().to(device).float()
w3 = Model['model_state_dict']['linears.2.weight'].detach().to(device).float()
w4 = Model['model_state_dict']['linears.3.weight'].detach().to(device).float()
b1 = Model['model_state_dict']['linears.0.bias'].detach().to(device).float()
b2 = Model['model_state_dict']['linears.1.bias'].detach().to(device).float()
b3 = Model['model_state_dict']['linears.2.bias'].detach().to(device).float()
b4 = Model['model_state_dict']['linears.3.bias'].detach().to(device).float()

P_list = P(w1,w2,w3,w4,b1,b2,b3,b4).to(device)

Xi, Xf         = -1.0, 1.0
Yi, Yf         = -1.0, 1.0
Nc, N_test     =  500, 300
BC_pts         =  200

residual_data = create_residual_data(Xi, Xf, Yi, Yf, Nc, N_test)
xy_resid      = residual_data[0].to(device)
f_hat         = residual_data[1].to(device)
xy_test       = residual_data[2].to(device) 
print(type(xy_resid))

d2Ez_r_x2, d2Ez_r_y2, d2Ez_i_x2, d2Ez_i_y2 = autograd_calculations(xy_resid, P_list)
print("completed autograd calculations")
# u = P_list(xy_resid.float()).to(device)
# Erz = u[:,0]
"""
