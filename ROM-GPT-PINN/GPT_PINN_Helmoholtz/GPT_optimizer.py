import torch
from boundary_fun import EHx_abc_r_grad, EHy_abc_r_grad, EHx_abc_i_grad, EHy_abc_i_grad
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class grad_descent(object): 
    def __init__(self, cond,eps1_r,eps, xy_resid, xy_BC, kap_curl2,
                 omk_Ez, dEz_x, dEz_y, Ez, lr_gpt):
        # PDE parameters
        self.eps1_r = eps1_r
        self.eps_r2 = eps
        # Center of the circle
        self.cond = cond
        # Data sizes
        self.N_R  = xy_resid.shape[0]
        # self.N_IC = IC_xt.shape[0]
        self.N_BC = xy_BC.shape[0]
        
        # Precomputed / Data terms  
        self.kap_curl2    = kap_curl2
        # self.kap_curl2Ez_i       = kap_curl2Ez_i
        # self.P_IC_values       = P_IC_values
        # self.IC_u              = IC_u
        self.omk_eps_r1_Ez          = omk_Ez * eps1_r
        self.omk_eps_r2_Ez          = omk_Ez * eps
        # self.omk_eps_r1_Ez_i = omk_Ez_i * eps1_r
        # self.omk_eps_r2_Ez_i = omk_Ez_i * eps

        self.dEz_x = dEz_x
        self.dEz_y = dEz_y
        # self.dEz_r_x = dEz_r_x
        # self.dEz_r_y = dEz_r_y
    
        # Optimizer data/parameter
        self.lr_gpt = lr_gpt
        self.xy_BC = xy_BC
        self.Ez = Ez
    
    def grad_loss(self, c):
        c = c.to(device)
        grad_list = torch.zeros_like(c)
        loss_weights = [0.015, 0.015, 0.125, 0.125, 0.125, 0.125]
        #######################################################################
        #######################################################################        
        #########################  Residual Gradient  #########################



        kap_curl2Ez_r = torch.matmul(self.kap_curl2, c[[0], :].transpose(0, 1))
        kap_curl2Ez_i = torch.matmul(self.kap_curl2, c[[1], :].transpose(0, 1))

        # fEz_1 = omk*eps1_r*Ez_r - kap_curl2Ez_r  # outside the disk
        # fEz_2 = omk*eps2_r*Ez_r - kap_curl2Ez_r  # inside the disk

        # fEz = torch.where(self.cond, fEz_2, fEz_1)

        omk_eps_r1_Ez_r = torch.matmul(self.omk_eps_r1_Ez, c[[0],:].transpose(0,1))
        omk_eps_r2_Ez_r = torch.matmul(self.omk_eps_r2_Ez, c[[0],:].transpose(0,1))
        omk_eps_r1_Ez_i = torch.matmul(self.omk_eps_r1_Ez, c[[1],:].transpose(0,1))
        omk_eps_r2_Ez_i = torch.matmul(self.omk_eps_r2_Ez, c[[1],:].transpose(0,1))
        first_product_Ez_r1 = torch.add(kap_curl2Ez_r, omk_eps_r1_Ez_r)  # outside the disk
        first_product_Ez_r2 = torch.add(kap_curl2Ez_r, omk_eps_r2_Ez_r)  # inside the disk
        first_product_Ez_i1 = torch.add(kap_curl2Ez_i, omk_eps_r1_Ez_i)
        first_product_Ez_i2 = torch.add(kap_curl2Ez_i, omk_eps_r2_Ez_i)

        first_product_Ez_r = torch.where(self.cond, first_product_Ez_r2, first_product_Ez_r1)
        first_product_Ez_i = torch.where(self.cond, first_product_Ez_i2, first_product_Ez_i1)

        second_product_Ez_1 = torch.add(self.kap_curl2, self.omk_eps_r1_Ez)
        second_product_Ez_2 = torch.add(self.kap_curl2, self.omk_eps_r2_Ez)
        # second_product_Ez_i1 = torch.add(self.kap_curl2, self.omk_eps_r1_Ez)
        # second_product_Ez_i2 = torch.add(self.kap_curl2, self.omk_eps_r2_Ez)

        second_product_Ez = torch.where(self.cond, second_product_Ez_2, second_product_Ez_1)
        # second_product_Ez_i = torch.where(self.cond, second_product_Ez_i2, second_product_Ez_i1)
        grad_list[0, :] = loss_weights[0] * torch.mul(2/self.N_R, torch.sum(torch.mul(first_product_Ez_r, second_product_Ez), dim=0))
        grad_list[1, :] = loss_weights[1] * torch.mul(2/self.N_R, torch.sum(torch.mul(first_product_Ez_i, second_product_Ez), dim=0))
        #
        #######################################################################
        #######################################################################        
        ###################  Boundary Gradient  ###################   


        grad_list += loss_weights[2] * EHx_abc_r_grad(torch.from_numpy(self.xy_BC).to(device), self.Ez, self.N_BC, self.dEz_x, self.dEz_y, c)
        grad_list += loss_weights[3] * EHy_abc_r_grad(torch.from_numpy(self.xy_BC).to(device), self.Ez, self.N_BC, self.dEz_x, self.dEz_y, c)
        grad_list += loss_weights[4] * EHx_abc_i_grad(torch.from_numpy(self.xy_BC).to(device), self.Ez, self.N_BC, self.dEz_x, self.dEz_y, c)
        grad_list += loss_weights[5] * EHy_abc_i_grad(torch.from_numpy(self.xy_BC).to(device), self.Ez, self.N_BC, self.dEz_x, self.dEz_y, c)
        return grad_list

    def update(self, c):
        c = torch.sub(c, torch.mul(self.lr_gpt, self.grad_loss(c)))
        return c
        # return c.expand(1, c.shape[0])