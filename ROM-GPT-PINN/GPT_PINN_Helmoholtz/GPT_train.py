import torch
from GPT_optimizer import grad_descent
torch.set_default_dtype(torch.float)
from Log.log import logger
import matplotlib.pyplot as plt

def gpt_train(GPT_PINN, cond,eps1_r, eps, xy_resid, xy_BC, kap_curl2, omk_Ez,
            dEz_x, dEz_y, Ez, epochs_gpt, lr_gpt, path, largest_loss=None,
              largest_case=None, testing=False):
    
    GD = grad_descent(cond, eps1_r, eps, xy_resid, xy_BC, kap_curl2,
                      omk_Ez, dEz_x, dEz_y, Ez, lr_gpt)



    if (testing == False):
        print(f"gpt training step: epsilon equal to {eps}")
        logger.info(f"gpt training step: epsilon equal to {eps}")
        loss_values_list = []
        loss_values = GPT_PINN.loss()
        loss_values_list.append(loss_values.cpu().detach().numpy())
        for i in range(1, epochs_gpt+1):
            if (loss_values < largest_loss):
                print(f"loss values {loss_values} less than largest loss {largest_loss}, break!")
                logger.info(f"loss values {loss_values} less than largest loss {largest_loss}, break!")
                break


            else:
                c = GPT_PINN.linears[1].weight.data
                GPT_PINN.linears[1].weight.data = GD.update(c)
                loss_values = GPT_PINN.loss()
                loss_values_list.append(loss_values.cpu().detach().numpy())

                if (i == epochs_gpt):
                    largest_case = eps
                    largest_loss = GPT_PINN.loss()
                    print(f"largest loss: {largest_loss}")
                    logger.info(f"largest loss: {largest_loss}")
                    


            if i % 100 == 0:
                print(f"Epoch {i}: Loss {loss_values}")
                logger.info(f"Epoch {i}: Loss {loss_values}")


        # plt.plot(range(epochs_gpt + 1), loss_values_list, color='blue', linestyle='-', label='train loss')
        # plt.legend(loc = 'best')
        # plt.xlabel('epochs')
        # plt.ylabel('loss')
        # plt.savefig(fr"{path}/loss.png", format='png')
        return largest_loss, largest_case
    
    elif (testing):
        testloss_values_list = []
        for i in range(1, epochs_gpt+1):
            c = GPT_PINN.linears[1].weight.data
            GPT_PINN.linears[1].weight.data = GD.update(c)
            testloss_values = GPT_PINN.loss()
            testloss_values_list.append(testloss_values.cpu().detach().numpy())

            if i % 100 == 0:
                print(f"Epoch {i}: Loss {testloss_values}")
                logger.info(f"Epoch {i}: Loss {testloss_values}")

        plt.plot(range(epochs_gpt), testloss_values_list, label='loss')
        plt.legend(loc='best')
        plt.xlabel('epochs')
        plt.ylabel('GPT-PINN loss')
        plt.title(fr"$\epsilon = {eps:.2f}$", fontsize=20)
        plt.savefig(fr"{path}/loss.pdf", format='pdf', bbox_inches='tight')
        plt.show()