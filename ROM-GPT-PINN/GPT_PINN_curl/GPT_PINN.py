import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import math

class GPT(nn.Module):
    def __init__(self,  layers, mu, P, initial_c, xyz_resid,xyz_BC, BC_u,
                 activation_resid, activation_BC, curl2):
        super().__init__()

        self.layers = layers

        self.mu = mu
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1], bias=False) for i in range(len(layers)-1)])
        self.activation = P
        self.curl2 = curl2
        self.activation_resid = activation_resid
        # self.activation_IC    = activation_IC
        self.activation_BC    = activation_BC
        


        # self.IC_u     = IC_u
        self.xyz_BC     = xyz_BC
        self.xyz_resid = xyz_resid
        self.BC_u     = BC_u

        
        self.linears[0].weight.data = torch.ones(self.layers[1], self.layers[0])
        self.linears[1].weight.data = initial_c
        
    def forward(self, datatype=None, test_data=None):
        if test_data is not None: # Test Data Forward Pass
            a = torch.Tensor().to(device)
            for i in range(0, self.layers[1]):
                a = torch.cat((a, self.activation[i](test_data)), 1)

            final_output = self.linears[-1](a)
            return final_output
        
        if datatype == 'residual': # Residual Data Output

            final_output = self.linears[-1](self.activation_resid).to(device)
            return final_output
        

        if datatype == 'boundary': # Boundary Data Output

            final_output = self.linears[-1](self.activation_BC).to(device)
            return final_output
    
    def lossR(self):
        """Residual loss function"""
        u  = self.forward(datatype='residual')
        # Ez_r, Ez_i = u[:,0:1], u[:,1:2]
        curl2Az = torch.matmul(self.curl2, self.linears[1].weight.data[0][:,None])
        source_term = self.mu*torch.sin(math.pi * torch.tensor(self.xyz_resid[:, 0:1])) * torch.sin(math.pi * torch.tensor(self.xyz_resid[:, 1:2])) * torch.sin(math.pi * torch.tensor(self.xyz_resid[:, 2:3]))


        return 0.015*self.loss_function(curl2Az, -source_term)
    
    def lossBC(self, datatype):
        """First initial and both boundary condition loss function"""
        u = self.forward(datatype)
        return 0.125*self.loss_function(u, self.BC_u)
 
    def loss(self):
        """Total Loss Function"""
        loss_R   = self.lossR()
        
        loss_BC  = self.lossBC(datatype='boundary')
        return loss_R + loss_BC 