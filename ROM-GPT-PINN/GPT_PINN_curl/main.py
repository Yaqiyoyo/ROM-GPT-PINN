# Import and GPU Support
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import math
import time
import deepxde as dde
from scipy.io import savemat


from get_dde_PINN_data import get_dde_PINN_data
# Full PINN
from dde_PINN_train import PINN_train  # the full pinn training, set the corresponding parameter in this program
from A_z_analytic import A_z_analytic
from Plotting import plot_solution
from GetPredictData import GetPredictData

# GPT PINN
from GPT_activation import P
from GPT_precomp import autograd_calculations, autograd_calculations_BC
from GPT_PINN import GPT  # define the loss function of the gpt-pinn
from GPT_train import gpt_train

from Log.log import logger

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current Device: {device}")
if torch.cuda.is_available():
    print(f"Current Device Name: {torch.cuda.get_device_name()}")

######################################################################

# define physics information: domain, boundary
L = 2.0   # Length of rectangle (R) 
l = 2.0   # Width of rectangle (R)
# We choose l=L to have a square 

# Bounds of x, y
x_lower = 0
x_upper =  1
y_lower = 0
y_upper = 1
z_lower = 0
z_upper = 1

# Center of the circle
mu_max = 5
mu_num = 50   # 30,50
outer = dde.geometry.Cuboid(xmin=[x_lower,y_lower,z_lower], xmax=[x_upper,y_upper,z_upper])
geom = outer

def is_on_boundary(x):
    return geom.on_boundary(x)


mu = np.linspace(1, mu_max, mu_num)  # parameter set

number_of_neurons = 2 # the number of the hidden layer

loss_list         = np.ones(number_of_neurons)
print(f"Expected Final GPT-PINN Depth: {[3,number_of_neurons,1]}\n")
logger.info(f"Expected Final GPT-PINN Depth: {[3,number_of_neurons,1]}\n")

###############################################################################
#################################### Setup ####################################
###############################################################################
mu_neurons    = [1 for i in range(number_of_neurons)] # Neuron parameters
mu_neurons[0] = 1.5  # initial eps value
full_pinn_label = f'mu-{mu_max}-{number_of_neurons}-{mu_neurons[0]}'

P_list = np.ones(number_of_neurons, dtype=object)


lr_gpt          = 0.001   # 0.01 learning rate of gpt pinn
epochs_gpt      = 3000
epochs_gpt_test = 1000
test_cases      = 5  # predict electric field of 5 epsilons


num_domain = 5000
num_boundary = 3500
num_test = 2500

P_curl2_term = torch.ones((num_domain, number_of_neurons)).to(device)

BC_u = torch.zeros((num_boundary*2, 1)).to(device)
P_resid_values = torch.ones((num_domain, number_of_neurons)).to(device)
P_BC_values    = torch.ones((num_boundary*2, number_of_neurons)).to(device)



# Save Data/Plot Options
save_data         = True
plot_pinn_loss    = True
plot_pinn_sol     = True
plot_largest_loss = True
plot_gptpinn_sol = True
train_final_gpt   = True
pinn_train_times = np.ones(number_of_neurons)
gpt_train_times  = np.ones(number_of_neurons)

total_train_time_1 = time.perf_counter()
###############################################################################
################################ Training Loop ################################
###############################################################################
for i in range(0, number_of_neurons):
    print("******************************************************************")
    ########################### Full PINN Training ############################
    mu_pinn_train = mu_neurons[i]
    data = get_dde_PINN_data(geom, mu_pinn_train, num_domain, num_boundary, num_test)
    xyz_train = data.train_x  # training data
    boundary_mask = np.array([is_on_boundary(x) for x in xyz_train])  # boundary points mask
    interior_mask = ~boundary_mask  # interior points mask
    xyz_resid = xyz_train[interior_mask]  # interior points
    xyz_BC = xyz_train[boundary_mask]  # boundary points
    pinn_train_time_1 = time.perf_counter()
    model, losshistory, train_state = PINN_train(data)

    path = fr"./train/Full-PINN-Data ({full_pinn_label})/({mu_pinn_train})"
    if not os.path.exists(path):
        os.makedirs(path)

    model.save(fr"{path}/model_curl_{mu_pinn_train}")

    if (i+1 == number_of_neurons):
        print(f"Begin Final Full PINN Training: nu={mu_pinn_train} (Obtaining Neuron {i+1})")
        logger.info(f"Begin Final Full PINN Training: nu={mu_pinn_train} (Obtaining Neuron {i+1})")
    else:
        print(f"Begin Full PINN Training: nu={mu_pinn_train} (Obtaining Neuron {i+1})")
        logger.info(f"Begin Full PINN Training: nu={mu_pinn_train} (Obtaining Neuron {i+1})")
        

    pinn_train_time_2 = time.perf_counter()
    print(f"PINN Training Time: {(pinn_train_time_2-pinn_train_time_1)/3600} Hours")
    logger.info(f"PINN Training Time: {(pinn_train_time_2-pinn_train_time_1)/3600} Hours")
    # load model
    Model = torch.load(fr'{path}/model_curl_{mu_pinn_train}-{train_state.epoch}.pt')
    w1 = Model['model_state_dict']['linears.0.weight'].detach()
    w2 = Model['model_state_dict']['linears.1.weight'].detach()
    w3 = Model['model_state_dict']['linears.2.weight'].detach()
    w4 = Model['model_state_dict']['linears.3.weight'].detach()
    b1 = Model['model_state_dict']['linears.0.bias'].detach()
    b2 = Model['model_state_dict']['linears.1.bias'].detach()
    b3 = Model['model_state_dict']['linears.2.bias'].detach()
    b4 = Model['model_state_dict']['linears.3.bias'].detach()
    
        
    P_list[i] = P(w1, w2, w3, w4, b1, b2, b3, b4).to(device)

    print(f"\nCurrent GPT-PINN Depth: [3,{i+1},1]")
    logger.info(f"\nCurrent GPT-PINN Depth: [3,{i+1},1]")

    if (plot_pinn_sol):
        nbx = 100
        nby = 100
        x_grid,y_grid,grid_points = GetPredictData(nbx, nby)
        predictions = model.predict(grid_points)
        print(type(predictions))
        A_z_true = A_z_analytic(grid_points, mu_pinn_train)
        plot_solution(predictions, A_z_true, x_grid, y_grid, mu_pinn_train, path)

    if (save_data):        

        np.savetxt(fr"{path}/saved_w1.txt", w1.cpu())
        np.savetxt(fr"{path}/saved_w2.txt", w2.cpu())
        np.savetxt(fr"{path}/saved_w3.txt", w3.cpu())
        np.savetxt(fr"{path}/saved_w4.txt", w4.cpu())

        np.savetxt(fr"{path}/saved_b1.txt", b1.cpu())
        np.savetxt(fr"{path}/saved_b2.txt", b2.cpu())
        np.savetxt(fr"{path}/saved_b3.txt", b3.cpu())
        np.savetxt(fr"{path}/saved_b4.txt", b4.cpu())

    if (plot_pinn_loss):
        dde.utils.external.plot_loss_history(losshistory, fname=fr"{path}/loss_history_mu_{mu_pinn_train}.pdf")
        # loss_vals = pinn_losses[0]
        # epochs    = pinn_losses[1]
        # loss_plot(epochs, loss_vals, title=fr"PINN Losses $\nu={eps_pinn_train}$")

    if (i == number_of_neurons-1) and (train_final_gpt == False):
        break

    ############################ GPT-PINN Training ############################

    num_largest_mag  = int(xyz_resid.shape[0]*0.2)
    idx_list         = torch.ones((number_of_neurons, num_largest_mag),dtype=torch.long)

    layers_gpt = np.array([3, i+1, 1])

    Az, Az_xx, Az_yy, Az_zz = autograd_calculations(torch.tensor(xyz_resid), P_list[i])
    Az_BC = autograd_calculations_BC(torch.tensor(xyz_BC), P_list[i])

    P_curl2_term[:,i][:, None] = Az_xx + Az_yy + Az_zz
    P_resid_values[:,i][:, None] = Az

    P_BC_values[:,i][:, None] = Az_BC



    # Finding The Next Neuron   
    largest_case = 0
    largest_loss = 0

    if (i+1 == number_of_neurons):
        print("\nBegin Final GPT-PINN Training (Largest Loss Training)")
        logger.info("\nBegin Final GPT-PINN Training (Largest Loss Training)")
    else:
        print(f"\nBegin GPT-PINN Training (Finding Neuron {i+2} / Largest Loss Training)")
        logger.info(f"\nBegin GPT-PINN Training (Finding Neuron {i+2} / Largest Loss Training)")

    gpt_train_time_1 = time.perf_counter()
    for pa in mu:
        if layers_gpt[1] == 1:
            c_initial = torch.ones(layers_gpt[1])

        elif pa in mu_neurons[:i+1]:
            index     = np.where(pa == mu_neurons[:i+1])
            c_initial = torch.zeros(layers_gpt[1])
            c_initial[index] = 1

        else:
            dist = np.zeros(layers_gpt[1])
            for k, mu_neuron in enumerate(mu_neurons[:i + 1]):
                dist[k] = np.abs(mu_neuron - pa)
    
            d      = np.argsort(dist)
            first  = d[0] 
            second = d[1] 
    
            a = dist[first]
            b = dist[second]
            bottom = a+b
            
            c_initial = torch.zeros(layers_gpt[1])
            c_initial[first] = b / bottom
            c_initial[second] = a / bottom
        c_initial = c_initial[None,:]

        GPT_NN = GPT(layers_gpt, pa, P_list[0:i+1], c_initial, xyz_resid,xyz_BC, BC_u,
                     P_resid_values[:, 0:(i+1)], P_BC_values[:, 0:(i+1)], P_curl2_term[:, 0:(i+1)]
                     ).to(device)
    
        gpt_losses = gpt_train(GPT_NN, pa, xyz_resid, xyz_BC, P_curl2_term[:, 0:(i+1)],
                               P_BC_values[:, 0:(i+1)], epochs_gpt, lr_gpt, path, largest_loss, largest_case)
        
        largest_loss = gpt_losses[0]
        largest_case = gpt_losses[1]

    gpt_train_time_2 = time.perf_counter()
    print("GPT-PINN Training Completed")
    logger.info("GPT-PINN Training Completed")
    print(f"\nGPT Training Time ({i+1} Neurons): {(gpt_train_time_2-gpt_train_time_1)/3600} Hours")
    logger.info(f"\nGPT Training Time ({i+1} Neurons): {(gpt_train_time_2-gpt_train_time_1)/3600} Hours")
    
    loss_list[i] = largest_loss
    
    if (i+1 < number_of_neurons):
        mu_neurons[i+1] = largest_case
        
    print(f"\nLargest Loss (Using {i+1} Neurons): {largest_loss}")
    logger.info(f"\nLargest Loss (Using {i+1} Neurons): {largest_loss}")
    print(f"Parameter Case: {largest_case}")
    logger.info(f"Parameter Case: {largest_case}")
total_train_time_2 = time.perf_counter()                       

###############################################################################
# Results of largest loss, parameters chosen, and times may vary based on
# the initialization of full PINN and the final loss of the full PINN
print("******************************************************************")
print("*** Full PINN and GPT-PINN Training Complete ***")
logger.info("*** Full PINN and GPT-PINN Training Complete ***")
print(f"Total Training Time: {(total_train_time_2-total_train_time_1)/3600} Hours\n")
logger.info(f"Total Training Time: {(total_train_time_2-total_train_time_1)/3600} Hours\n")
print(f"Final GPT-PINN Depth: {[3,len(P_list),1]}")
logger.info(f"Final GPT-PINN Depth: {[3,len(P_list),1]}")
print(f"\nActivation Function Parameters: \n{mu_neurons}\n")
logger.info(f"\nActivation Function Parameters: \n{mu_neurons}\n")

for j in range(number_of_neurons-1):
    print(f"Largest Loss of GPT-PINN Depth {[3,(j+1),1]}: {loss_list[j]}")
    logger.info(f"Largest Loss of GPT-PINN Depth {[3,(j+1),1]}: {loss_list[j]}")
if (train_final_gpt):
    print(f"Largest Loss of GPT-PINN Depth {[3,(j+2),1]}: {loss_list[-1]}")
    logger.info(f"Largest Loss of GPT-PINN Depth {[3,(j+2),1]}: {loss_list[-1]}")
        
if (plot_largest_loss):
    plt.figure(dpi=150, figsize=(10,8))
    
    if (train_final_gpt):
        range_end = number_of_neurons + 1
        list_end  = number_of_neurons
    else:
        range_end = number_of_neurons 
        list_end  = number_of_neurons - 1
        
    plt.plot(range(1,range_end), loss_list[:list_end], marker='o', markersize=7, 
             c="k", linewidth=3)
    
    plt.grid(True)
    plt.xlim(1,max(range(1,range_end)))
    plt.xticks(range(1,range_end))
    
    plt.yscale("log") 
    plt.xlabel("Number of Neurons",      fontsize=17.5)
    plt.ylabel("Largest Loss",           fontsize=17.5)
    plt.title("GPT-PINN Largest Losses", fontsize=17.5)
    plt.savefig(fr"./train/Full-PINN-Data ({full_pinn_label})/largest_loss.pdf", format='pdf',bbox_inches='tight')
    plt.show()

savemat(fr"./train/Full-PINN-Data ({full_pinn_label})/loss_list.mat",{'loss_list':loss_list})
############################### GPT-PINN Testing ############################## 
# mu_test = mu.tolist()
# for i in mu_neurons:
#     if (i in mu_test):
#         mu_test.remove(i)

# idx = np.random.choice(len(mu_test), test_cases, replace=False)
# mu_test = np.array(mu_test)[idx]

mu_test = np.array([1.205, 2.215, 3.225, 4.235])

print(mu_test)
logger.info(f"EPSILON Test: {mu_test}")
# print(f"\nBegin GPT-PINN Testing ({len(set(idx.flatten()))} Cases)")
# logger.info(f"\nBegin GPT-PINN Testing ({len(set(idx.flatten()))} Cases)")

layers_gpt = np.array([3, len(P_list), 1])

total_test_time_1 = time.perf_counter()
incremental_test_times = np.ones(len(mu_test))
cnt = 0

for mu_test_param in mu_test:
    time0 = time.perf_counter()
    dist = np.zeros(layers_gpt[1])
    for k, mu_neuron in enumerate(mu_neurons):
        dist[k] = np.abs(mu_neuron - mu_test_param)

    d      = np.argsort(dist)
    first  = d[0] 
    second = d[1] 

    a = dist[first]
    b = dist[second]
    bottom = a+b
        
    c_initial = torch.zeros(layers_gpt[1])
    c_initial[first]  = b / bottom
    c_initial[second] = a / bottom
    c_initial = c_initial[None,:]

    # Pt_nu_P_xx_term = Pt_nu_P_xx(nu_test_param, P_t_term, P_xx_term)
    test_path = fr"./test/Full-PINN-Data ({full_pinn_label})/({mu_test_param})"
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    GPT_NN = GPT(layers_gpt, mu_test_param, P_list, c_initial, xyz_resid,xyz_BC, BC_u,
                     P_resid_values, P_BC_values, P_curl2_term).to(device)
    
    gpt_losses = gpt_train(GPT_NN, mu_test_param, xyz_resid, xyz_BC, P_curl2_term,
                               P_BC_values, epochs_gpt_test, lr_gpt, test_path, largest_loss, largest_case, testing=True)
    
    incremental_test_times[cnt] = time.perf_counter()-total_test_time_1
    print(fr"Testing time of mu={mu_test_param}:{time.perf_counter()-time0}")
    cnt += 1
    if (plot_gptpinn_sol):
        nbtestx = 200
        nbtesty = 200
        xtest_grid, ytest_grid, xytest_grid = GetPredictData(nbtestx, nbtesty)
        gpt_predictions = GPT_NN.forward(test_data= torch.tensor(xytest_grid).float())
        testAz_true = A_z_analytic(xytest_grid, mu_test_param)
        plot_solution(gpt_predictions.cpu().detach().numpy(), testAz_true, xtest_grid, ytest_grid, mu_test_param, path=test_path)
        # print(type(gpt_predictions))


savemat(fr"./test/Full-PINN-Data ({full_pinn_label})/incremental_test_times.mat",{'incremental_test_times':incremental_test_times})

total_test_time_2 = time.perf_counter()
print("\nGPT-PINN Testing Completed")
logger.info("\nGPT-PINN Testing Completed")
print(f"\nTotal Testing Time: {(total_test_time_2-total_test_time_1)/3600} Hours")
logger.info(f"\nTotal Testing Time: {(total_test_time_2-total_test_time_1)/3600} Hours")

init_time = (total_train_time_2-total_train_time_1)/3600
test_time = incremental_test_times
line = test_time + init_time
x = range(1,test_time.shape[0]+1)
plt.figure(dpi=150, figsize=(10,8))
plt.plot(x, line, c="k", lw=3.5)
plt.xlabel("Test Case Number", fontsize=22.5)
plt.ylabel("Time (Hours)", fontsize=22.5)
plt.xlim(min(x),max(x))
plt.ylim(min(line),max(line))
xtick = list(range(0,test_cases+1, 5))
xtick[0] = 1
plt.xticks(xtick)
plt.grid(True)
plt.savefig(fr"./test/Full-PINN-Data ({full_pinn_label})/runtime.pdf", format='pdf')
plt.show()
