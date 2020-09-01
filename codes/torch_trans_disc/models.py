# # Contains the model definition for the classifier
# # We are considering two approaches - 
# # 1. Transformer does heavy lifting and just gives a transformed distribution to the classifier
# # 2. Classifier does heavy lifting and tries to learn P(y|X,t) with special emphasis on time = T+1

from torch import nn
import torch

class ClassifyNet(nn.Module):
    def __init__(self,data_shape, hidden_shape, out_shape, time_conditioning=False,leaky=False):
        super(ClassifyNet,self).__init__()

        self.time_conditioning = time_conditioning

        self.layer_0 = nn.Linear(data_shape,hidden_shape)
        self.relu_0    = TimeReLU(data_shape,1,leaky)
        # self.relu_0 = nn.ReLU()
        if time_conditioning:
            self.layer_1 = nn.Linear(hidden_shape,hidden_shape)
            self.layer_2 = nn.Linear(hidden_shape,hidden_shape)
            self.relu_t = TimeReLU(hidden_shape,1,leaky)
            # self.relu_t = nn.ReLU()
        self.out_layer = nn.Linear(hidden_shape,out_shape)
        self.out_relu = TimeReLU(out_shape,1,leaky)

    def forward(self,X):
        # times = X[:,-1:]
        X  = self.layer_0(X)
        X  = self.relu_0(X)

        if self.time_conditioning:
            X = self.layer_1(X)
            X = self.relu_t(X)
            X = self.layer_2(X)
            X = self.relu_t(X)

        X = self.out_layer(X)
        X = self.out_relu(X)
        X = torch.softmax(X,dim=1)

        return X


class Transformer(nn.Module):
    def __init__(self,data_shape, latent_shape, label_dim=0):
        '''
        Notes to self - 
            1. This should transform both X and Y
            2. We can leverage joint modelling of X and Y somehow?
            3. Are we taking on a tougher task? Will it work to just time travel one unit?
            4. Can we decouple the training process of generating X and Y?
        Seems to collapse domain somehow, maybe network power issue? 
        Classifier seems to aggravate it, but classifier loss is "most potent" on this dataset
        Further, labels seem right when classifier loss is added
        Also, is extrapolating good?
        OT helped mode collapse and far away, do some ablations.
        Did not investigate how classifier loss is helping?
        Training is "hard", i.e. unstable!
        Where are we using time?
        '''
        super(Transformer,self).__init__()
        self.layer_0 = nn.Linear(data_shape,latent_shape)
        self.leaky_relu_0 = TimeReLU(latent_shape,2,True)
        self.layer_0_0 = nn.Linear(latent_shape,latent_shape)
        self.leaky_relu_0_0 = TimeReLU(latent_shape,2,True)
        self.layer_0_1 = nn.Linear(latent_shape,latent_shape)
        self.leaky_relu_0_1 = TimeReLU(latent_shape,2,True)
        # self.layer_0_2 = nn.Linear(latent_shape,latent_shape)

        self.layer_1 = nn.Linear(latent_shape,data_shape-2)
        self.label_dim = label_dim

    def forward(self,X):
        X = (self.layer_1((self.leaky_relu_0_1(self.layer_0_1(self.leaky_relu_0_0(self.layer_0_0(self.leaky_relu_0(self.layer_0(X)))))))))
        if self.label_dim:
            lab = torch.sigmoid(X[:,-1])
            # X_new = self.leaky_relu(X[:,:-1])
            X = torch.cat([X,lab.unsqueeze(1)],dim=1)
        # else:
        #     X = self.leaky_relu(X)
        return X


class Discriminator(nn.Module):
    def __init__(self,data_shape, hidden_shape, is_wasserstein=False):

        super(Discriminator,self).__init__()
        self.layer_0   = nn.Linear(data_shape,hidden_shape)
        self.relu      = nn.LeakyReLU()
        self.out_layer = nn.Linear(hidden_shape,1)
        self.is_wasserstein = is_wasserstein

    def forward(self,X):
        X = self.out_layer(self.relu(self.layer_0(X)))

        if not self.is_wasserstein:
            X = torch.sigmoid(X)
            
        return X


class TimeReLU(nn.Module):
    def __init__(self,data_shape,time_shape,leaky=False):
        super(TimeReLU,self).__init__()
        self.model = nn.Linear(time_shape,data_shape)
        if leaky:
            self.model_alpha = nn.Linear(time_shape,data_shape)
        self.leaky = leaky
        self.time_dim = time_shape
    
    def forward(self,X):
        times = X[:,-self.time_dim:]
        thresholds = self.model(times)
        if self.leaky:
            alphas = self.model_alpha(times)
        else:
            alphas = 0.0
        X = torch.where(X>thresholds,X,alphas*X+thresholds)
        return X

class TimeEncodings(nn.Module):
    def __init__(self,model_dim,time_dim):
        super(TimeEncodings,self).__init__()
        self.model_dim = model_dim
        self.time_dim = time_dim
    
    def forward(self,X):
        times = X[:,-time_dim:]
        n,_ = X.size()
        freq_0 = torch.tensor([1/(100**(2*x/self.model_dim)) for x in range(self.model_dim)]).view(1,self.model_dim).repeat(n,1)

        offsets = torch.ones_like(X) * (3.1415/2)
        offsets[:,::2] = 0.0
        positional_0 = torch.sin(freq_0 * times[:,:1] + offsets)
        if self.time_dim == 2:
            freq_1 = torch.tensor([1/(50**(2*x/self.model_dim)) for x in range(self.model_dim)]).view(1,self.model_dim).repeat(n,1)
            positional_1 = torch.sin(freq_1 * times[:,:1] + offsets)
        else:
            positional_1 = torch.zeros_like(positional_0)
        X = X + positional_0 + positional_1
        return X
# class TimeEmbeddings(nn.Module):
#     def __init__(self,model_dim,encoding):



def classification_loss(Y_pred, Y):
    # print(Y_pred)
    return  -1.*torch.sum((Y * torch.log(Y_pred+ 1e-5)))

def bxe(real, fake):
    return -1.*((real*torch.log(fake+ 1e-5)) + ((1-real)*torch.log(1-fake + 1e-5)))

def discriminator_loss(real_output, trans_output):

    # bxe = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = bxe(torch.ones_like(real_output), real_output)
    trans_loss = bxe(torch.zeros_like(trans_output), trans_output)
    total_loss = real_loss + trans_loss
    
    return total_loss.mean()
def discriminator_loss_wasserstein(real_output, trans_output):

    # bxe = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = torch.mean(real_output)
    trans_loss = -torch.mean(trans_output)
    total_loss = real_loss + trans_loss
    
    return total_loss

def reconstruction_loss(x,y):
    # print(torch.cat([x,y],dim=1))
    return torch.sum((x-y)**2,dim=1)


def transformer_loss(trans_output,is_wasserstein=False):

    if is_wasserstein:
        return trans_output
    return bxe(torch.ones_like(trans_output), trans_output)

def discounted_transformer_loss(real_data, trans_data, ot_data,trans_output, pred_class, actual_class,is_wasserstein=False):

    time_diff = torch.exp(-(real_data[:,-1] - real_data[:,-2]))


    re_loss = reconstruction_loss(ot_data, trans_data)
    tr_loss = transformer_loss(trans_output,is_wasserstein)
    # transformed_class = trans_data[:,-1].view(-1,1)

    # print(actual_class,pred_class)
    class_loss = bxe(actual_class,pred_class)
    
    loss = torch.mean(1.*time_diff * tr_loss + (1-time_diff) * re_loss + 0.5*class_loss)
    # loss = tr_loss.mean()
    return loss, tr_loss.mean(),re_loss.mean(), class_loss.mean()

