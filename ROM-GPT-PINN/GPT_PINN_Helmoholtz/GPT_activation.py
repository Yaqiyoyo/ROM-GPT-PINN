import torch
import torch.nn as nn


torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class P(nn.Module):
    def __init__(self, w1, w2, w3, w4, b1, b2, b3, b4):
        super(P, self).__init__()
        self.layers = [2, 50, 50, 50, 2]
        self.linears = nn.ModuleList(nn.Linear(self.layers[i], self.layers[i + 1]) for i in range(len(self.layers) - 1))

        self.linears[0].weight.data = torch.Tensor(w1).float()
        self.linears[1].weight.data = torch.Tensor(w2).float()
        self.linears[2].weight.data = torch.Tensor(w3).float()    
        self.linears[3].weight.data = torch.Tensor(w4).float()

        self.linears[0].bias.data = torch.Tensor(b1).float()
        self.linears[1].bias.data = torch.Tensor(b2).float()
        self.linears[2].bias.data = torch.Tensor(b3).float()
        self.linears[3].bias.data = torch.Tensor(b4).float().view(-1)

        self.activation = Sin()

    def forward(self, x):
        a = x
        for i in range(0, len(self.layers) - 2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a

# Model = torch.load('model_square_disk_heterogeneous-20000.pt')
# w1 = Model['model_state_dict']['linears.0.weight'].detach().cpu()
# w2 = Model['model_state_dict']['linears.1.weight'].detach().cpu()
# w3 = Model['model_state_dict']['linears.2.weight'].detach().cpu()
# w4 = Model['model_state_dict']['linears.3.weight'].detach().cpu()
# b1 = Model['model_state_dict']['linears.0.bias'].detach().cpu()
# b2 = Model['model_state_dict']['linears.1.bias'].detach().cpu()
# b3 = Model['model_state_dict']['linears.2.bias'].detach().cpu()
# b4 = Model['model_state_dict']['linears.3.bias'].detach().cpu()

# P_list = P(w1,w2,w3,w4,b1,b2,b3,b4).to(device)

# Xi, Xf         = -1.0, 1.0
# Yi, Yf         = -1.0, 1.0
# Nc, N_test     =  500, 300
# BC_pts         =  200

# residual_data = create_residual_data(Xi, Xf, Yi, Yf, Nc, N_test)
# xy_resid      = residual_data[0].to(device)
# f_hat         = residual_data[1].to(device)
# xy_test       = residual_data[2].to(device) 
# print(type(xy_resid))

# u = P_list(xy_resid.float()).to(device)
# Erz = u[:,0]



