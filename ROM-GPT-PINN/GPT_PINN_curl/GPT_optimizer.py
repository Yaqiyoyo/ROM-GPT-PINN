import torch
import math

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class grad_descent(object): 
    def __init__(self, mu, xyz_resid, xyz_BC, P_curl2_term,P_BC_values,lr_gpt):
        # PDE parameters
        self.mu = mu


        # Data sizes
        self.N_R  = xyz_resid.shape[0]

        self.N_BC = xyz_BC.shape[0]
        self.P_curl2_term = P_curl2_term
        self.P_BC_values = P_BC_values
        
        # Precomputed / Data terms  




    
        # Optimizer data/parameter
        self.lr_gpt = lr_gpt
        self.xyz_resid = xyz_resid
        self.xyz_BC = xyz_BC

    
    def grad_loss(self, c):
        c = c.to(device)

        loss_weights = [0.015, 0.125]
        #######################################################################
        #######################################################################        
        #########################  Residual Gradient  #########################
        curl2_term = torch.matmul(self.P_curl2_term, c[:, None])
        source_term = self.mu * torch.sin(math.pi * torch.tensor(self.xyz_resid[:, 0:1])) * torch.sin(math.pi * torch.tensor(self.xyz_resid[:, 1:2])) * torch.sin(math.pi * torch.tensor(self.xyz_resid[:, 2:3]))
        first_product = torch.add(curl2_term, source_term)
        second_product = self.P_curl2_term
        grad_list = torch.mul(2 / self.N_R, torch.sum(torch.mul(first_product, second_product), axis=0))
        #######################################################################
        ####################### boundary gradient #############################
        BC_term = torch.matmul(self.P_BC_values, c[:, None])

        grad_list[:c.shape[0]] += torch.mul(2/self.N_BC, torch.sum(torch.mul(BC_term,  self.P_BC_values), axis=0))
        return grad_list

    def update(self, c):
        c = torch.sub(c, torch.mul(self.lr_gpt, self.grad_loss(c)))

        return c.expand(1, c.shape[0])