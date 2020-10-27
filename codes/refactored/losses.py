'''Implementation of losses

'''
import torch

def classification_loss(Y_pred, Y, n_classes=10):
    # print(Y_pred)
    # print(Y.shape,Y_pred.shape)
    Y_new = torch.zeros_like(Y_pred)
    Y_new = Y_pred.scatter(1,Y.view(-1,1),1.0)
    # print(Y_new * torch.log(Y_pred+ 1e-15))
    return  -1.*torch.sum((Y_new * torch.log(Y_pred+ 1e-15)),dim=1)

def bxe(real, fake):
    return -1.*((real*torch.log(fake+ 1e-5)) + ((1-real)*torch.log(1-fake + 1e-5)))

def discriminator_loss(real_output, trans_output):
    real_loss = bxe(torch.ones_like(real_output), real_output)
    trans_loss = bxe(torch.zeros_like(trans_output), trans_output)
    total_loss = real_loss + trans_loss
    
    return total_loss.mean()
def discriminator_loss_wasserstein(real_output, trans_output):
    real_loss = torch.mean(real_output)
    trans_loss = -torch.mean(trans_output)
    total_loss = real_loss + trans_loss
    
    return total_loss

def reconstruction_loss(x,y):
    x_1 = x.view(x.size(0),-1)
    y_1 = y.view(y.size(0),-1)
    return torch.sum((x_1-y_1)**2,dim=1) 


def transformer_loss(trans_output,is_wasserstein=False):

    if is_wasserstein:
        return trans_output
    return bxe(torch.ones_like(trans_output), trans_output)

def discounted_transformer_loss(rec_target_data, trans_data,trans_output, pred_class, actual_class,is_wasserstein=False):

    # time_diff = torch.exp(-(real_data[:,-1] - real_data[:,-2]))
    #TODO put time_diff


    re_loss = reconstruction_loss(rec_target_data, trans_data).view(-1,1)
    tr_loss = transformer_loss(trans_output,is_wasserstein).view(-1,1)
    # transformed_class = trans_data[:,-1].view(-1,1)

    # print(actual_class,pred_class)
    class_loss = classification_loss(pred_class,actual_class).view(-1,1)
    loss = torch.mean( 0.0* tr_loss.squeeze() +  0.0*re_loss + 0.5*class_loss)
    # loss = tr_loss.mean()
    return loss, tr_loss.mean(),re_loss.mean(), class_loss.mean()


def ot_transformer_loss(trans_data, source_u,dest_u,disc_output, ot_data, is_wasserstein=True):

    time_diff = torch.exp(-torch.abs(source_u - dest_u)/2).view(-1,1)
    
    re_loss = reconstruction_loss(trans_data, ot_data).view(-1,1)
    disc_loss = transformer_loss(disc_output, is_wasserstein).view(-1,1)
    loss = re_loss * (1 - time_diff) + disc_loss * time_diff
    # print(loss.size())
    return torch.mean(loss)
