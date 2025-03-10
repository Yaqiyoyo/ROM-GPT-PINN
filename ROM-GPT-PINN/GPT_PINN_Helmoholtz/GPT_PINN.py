import torch
import torch.nn as nn
from boundary_fun import EHx_abc_r_, EHx_abc_i_, EHy_abc_i_, EHy_abc_r_
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GPT(nn.Module):
    def __init__(self, cond, layers, eps1_r,eps, P, initial_c, xy_BC, BC_u, f_hat,
                 activation_resid, activation_BC, kap_curl2, omk_Ez, dEz_x, dEz_y):
        super().__init__()
        self.cond = cond
        self.layers = layers
        self.eps1_r = eps1_r
        self.eps = eps
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1], bias=False) for i in range(len(layers)-1)])
        self.activation = P
        
        self.activation_resid = activation_resid
        # self.activation_IC    = activation_IC
        self.activation_BC    = activation_BC
        
        self.kap_curl2 = kap_curl2
        # self.kap_curl2Ez_i = kap_curl2Ez_i
        self.omk_eps_r1_Ez = omk_Ez * eps1_r
        self.omk_eps_r2_Ez = omk_Ez * eps
        # self.omk_eps_r1_Ez_i = omk_Ez_i * eps1_r
        # self.omk_eps_r2_Ez_i = omk_Ez_i * eps
        self.dEz_x = dEz_x
        self.dEz_y = dEz_y
        # self.dEz_r_x = dEz_r_x
        # self.dEz_r_y = dEz_r_y
        # self.Pt_nu_P_xx_term = Pt_nu_P_xx_term
        # self.P_x_term        = P_x_term

        # self.IC_u     = IC_u
        self.xy_BC     = xy_BC
        self.BC_u     = BC_u
        self.f_hat    = f_hat

        
        self.linears[0].weight.data = torch.ones(self.layers[1], self.layers[0])
        self.linears[1].weight.data = initial_c
        
    def forward(self, datatype=None, test_data=None):
        if test_data is not None: # Test Data Forward Pass
            a = torch.Tensor().to(device)
            for i in range(0, int(self.layers[1]/2)):
                a = torch.cat((a, self.activation[i](test_data)), 1)

            # Er = self.linears[-1](a[:, [0]])[:, None]
            # Ei = self.linears[-1](a[:, [1]])[:, None]
            # final_output = torch.cat((Er, Ei), dim=1).to(device)
            final_output = self.linears[-1](a)
            return final_output
        
        if datatype == 'residual': # Residual Data Output
            # Er= self.linears[-1](self.activation_resid[:,[0]])
            # Ei= self.linears[-1](self.activation_resid[:,[1]])
            # final_output = torch.cat((Er, Ei), dim=1).to(device)
            final_output = self.linears[-1](self.activation_resid).to(device)
            return final_output
        
        # if datatype == 'initial': # Initial Data Output
        #     final_output = self.linears[-1](self.activation_IC).to(device)
        #     return final_output
        
        if datatype == 'boundary': # Boundary Data Output
            # Er = self.linears[-1](self.activation_BC[:, [0]])[:, None]
            # Ei = self.linears[-1](self.activation_BC[:, [1]])[:, None]
            # final_output = torch.cat((Er, Ei), dim=1).to(device)
            final_output = self.linears[-1](self.activation_BC).to(device)
            return final_output
    
    def lossR(self):
        """Residual loss function"""
        u  = self.forward(datatype='residual')
        # Ez_r, Ez_i = u[:,0:1], u[:,1:2]
        kap_curl2Ez_r = torch.matmul(self.kap_curl2, self.linears[1].weight.data[[0], :].transpose(0, 1))
        kap_curl2Ez_i = torch.matmul(self.kap_curl2, self.linears[1].weight.data[[1], :].transpose(0, 1))
        omk_eps_r1_Ez_r = torch.matmul(self.omk_eps_r1_Ez, self.linears[1].weight.data[[0], :].transpose(0, 1))
        omk_eps_r2_Ez_r = torch.matmul(self.omk_eps_r2_Ez, self.linears[1].weight.data[[0], :].transpose(0, 1))
        omk_eps_r1_Ez_i = torch.matmul(self.omk_eps_r1_Ez, self.linears[1].weight.data[[1], :].transpose(0, 1))
        omk_eps_r2_Ez_i = torch.matmul(self.omk_eps_r2_Ez, self.linears[1].weight.data[[1], :].transpose(0, 1))
        first_product_Ez_r1 = torch.add(kap_curl2Ez_r, omk_eps_r1_Ez_r)  # outside the disk
        first_product_Ez_r2 = torch.add(kap_curl2Ez_r, omk_eps_r2_Ez_r)  # inside the disk
        first_product_Ez_i1 = torch.add(kap_curl2Ez_i, omk_eps_r1_Ez_i)
        first_product_Ez_i2 = torch.add(kap_curl2Ez_i, omk_eps_r2_Ez_i)

        first_product_Ez_r = torch.where(self.cond, first_product_Ez_r2, first_product_Ez_r1)
        first_product_Ez_i = torch.where(self.cond, first_product_Ez_i2, first_product_Ez_i1)
        # ux = torch.matmul(self.P_x_term, self.linears[1].weight.data[0][:,None])
        # u_ux = torch.mul(u, ux)
        # ut_vuxx = torch.matmul(self.Pt_nu_P_xx_term, self.linears[1].weight.data[0][:,None])
        # f = torch.add(ut_vuxx, u_ux)
        return 0.015*self.loss_function(first_product_Ez_r, self.f_hat) + 0.015*self.loss_function(first_product_Ez_i, self.f_hat)
    
    def lossBC(self, datatype):
        """First initial and both boundary condition loss function"""
        u = self.forward(datatype)
        dEz_i_x = torch.matmul(self.dEz_x, self.linears[1].weight.data[[1], :].transpose(0, 1))
        dEz_i_y = torch.matmul(self.dEz_y, self.linears[1].weight.data[[1], :].transpose(0, 1))
        dEz_r_x = torch.matmul(self.dEz_x, self.linears[1].weight.data[[0], :].transpose(0, 1))
        dEz_r_y = torch.matmul(self.dEz_y, self.linears[1].weight.data[[0], :].transpose(0, 1))
        EHx_abc_r_loss = EHx_abc_r_(torch.from_numpy(self.xy_BC).to(device), u, dEz_i_x, dEz_i_y)
        EHy_abc_r_loss = EHy_abc_r_(torch.from_numpy(self.xy_BC).to(device), u, dEz_i_x, dEz_i_y)
        EHx_abc_i_loss = EHx_abc_i_(torch.from_numpy(self.xy_BC).to(device), u, dEz_r_x, dEz_r_y)
        EHy_abc_i_loss = EHy_abc_i_(torch.from_numpy(self.xy_BC).to(device), u, dEz_r_x, dEz_r_y)

        return 0.125*self.loss_function(EHx_abc_r_loss, self.BC_u) + 0.125*self.loss_function(EHy_abc_r_loss, self.BC_u) + 0.125*self.loss_function(EHx_abc_i_loss, self.BC_u) + 0.125*self.loss_function(EHy_abc_i_loss, self.BC_u)
 
    def loss(self):
        """Total Loss Function"""
        loss_R   = self.lossR()
        
        loss_BC  = self.lossBC(datatype='boundary')
        return loss_R + loss_BC 