import deepxde as dde
import time
import torch

dde.config.set_default_float("float64")

def activation_f(x):
    return torch.tanh(x) * torch.exp(-0.5*x**2)

def PINN_train(data):


    start = time.time()

    loss_weights = [0.015, 0.125] # For the PDE/BC

    # net = the structure of the NN + the activation function + the initializer

    # net = dde.nn.FNN([2] + [50]*3 + [2], "silu", "Glorot uniform")
    net = dde.nn.FNN([3] + [50] * 3 + [1], "sin", "Glorot uniform")
        
    # net = dde.nn.FNN([2] + [50]*3 + [2], activation_f, "Glorot uniform")

    # Build a model 
        
    model = dde.Model(data, net)

    # First training with Adam optimizer 

    model.compile("adam", lr=0.01, loss_weights=loss_weights) 
    losshistory, train_state = model.train(iterations=5000, display_every=1000)

    # Second training with L-BFGS optimizer 
    dde.optimizers.set_LBFGS_options(maxiter=2000)
    model.compile("L-BFGS", loss_weights=loss_weights)
    losshistory, train_state = model.train(iterations=2000, display_every=1000)

    end = time.time()

    print("Elapsed time: ", end - start)
    
    # model.save("model_square_disk_heterogeneous")
    
    return model, losshistory, train_state

