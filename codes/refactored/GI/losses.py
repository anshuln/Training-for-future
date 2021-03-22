'''Implementation of losses

'''
import torch

def classification_loss(Y_pred, Y):
    # print(Y_pred)
    # print(Y.shape,Y_pred.shape)
    Y_new = torch.zeros_like(Y_pred)
    Y_new = Y_new.scatter(1,Y.view(-1,1),1.0)
    # print(Y_new * torch.log(Y_pred+ 1e-15))
    return  -1.*torch.sum((Y_new * torch.log(Y_pred+ 1e-15)),dim=1)

def bxe(Y_pred, Y):
    return -1.*((Y*torch.log(Y_pred+ 1e-15)) + ((1-Y)*torch.log(1-Y_pred + 1e-15)))

def reconstruction_loss(x,y):
    x_1 = x.view(x.size(0),-1)
    y_1 = y.view(y.size(0),-1)
    return torch.sum((x_1-y_1)**2,dim=1) 


