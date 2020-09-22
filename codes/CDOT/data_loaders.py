import pandas as pd
import numpy as np
import math
from sklearn.datasets import make_classification, make_moons
from torchvision.transforms.functional import rotate
import torch
import os
from PIL import Image
device = "cpu"
def load_twitter(files_list):

    domains = 5
    X_data, Y_data, U_data = [], [], [] 

    for i, file in enumerate(files_list):

        df = pd.read_csv(file)
        Y_temp = df['y'].values
        X_temp = df.drop(['y', 'time', 'mean'], axis=1).values
        U_temp = np.array([i] * X_temp.shape[0])

        X_data.append(X_temp)
        Y_data.append(Y_temp)
        U_data.append(U_temp)

    return np.array(X_data), np.array(Y_data), np.array(U_data)

def load_news():

    domains = 5

    X_data, Y_data, U_data = [], [], []
    files_list = ['news_0.csv', 'news_1.csv', 'news_2.csv', 'news_3.csv', 'news_4.csv']
    
    # Assuming the last to be target, and first all to be source
    
    for i, file in enumerate(files_list):

        df = pd.read_csv(file)
        Y_temp = df[' shares'].values
        X_temp = df.drop([' timedelta', ' shares'], axis=1).values
        Y_temp = np.array([0 if d<=1400 else 1 for d in Y_temp])
        Y_temp = np.eye(2)[Y_temp]
        U_temp = np.array([i] * X_temp.shape[0])

        X_data.append(X_temp)
        Y_data.append(Y_temp)
        U_data.append(U_temp)


    return np.array(X_data), np.array(Y_data), np.array(U_data)

def load_moons(domains):

    X_data, Y_data, U_data = [], [], []
    for i in range(domains):

        angle = i*math.pi/(domains-1);
        X, Y = make_moons(n_samples=200, noise=0.1)
        rot = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
        X = np.matmul(X, rot)
        
        #plt.scatter(X[:,0], X[:,1], c=Y)
        #plt.savefig('moon_%d' % i)
        #plt.clf()

        Y = np.eye(2)[Y]
        U = np.array([i] * 200)

        X_data.append(X)
        Y_data.append(Y)
        U_data.append(U)

    return np.array(X_data), np.array(Y_data), np.array(U_data)


class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self,X,Y,U,transported_samples,source_indices,target_indices,this_U,index_fun):
        
        super(TransformerDataset,self).__init__()
        self.X = X 
        self.Y = Y 
        self.U = U
        self.this_U = this_U
        self.transported_samples = transported_samples
        self.source_indices = source_indices
        self.target_indices = target_indices
        self.index_fun = index_fun

        print(self.target_indices,self.source_indices)
        

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        X_item = self.X[idx]
        Y_item = self.Y[idx]
        U_item = self.U[idx]
        
        # print(self.source_indices.index[U_item]) #,self.target_indices.index(this_U))
        # print(idx)
        transported_X = self.transported_samples[self.source_indices.index(U_item)][self.source_indices.index(self.this_U)][self.index_fun(idx)]#.transform(X_item[np.newaxis,:])

        transported_X = np.squeeze(transported_X)
        # print(transported_X.shape,X_item.shape)
        return torch.from_numpy(X_item).float(),torch.tensor(U_item).float(),torch.tensor(Y_item).float(),torch.from_numpy(transported_X).float()
        



class RotMNIST(torch.utils.data.Dataset):
    '''
    Class which returns (image,label,angle,bin)
    TODO - 1. Shuffle indices, bin and append angles
           2. Return 
           3. OT - make sure OT takes into account the labels, i.e. OT loss should be inf for interchanging labels.  
    '''
    def __init__(self,indices,bin_width,bin_index,n_bins,transported_samples=None,target_bin=None,n_samples=6000):
        '''
        You give it a set of indices, along with which bins they belong
        It returns images from that MNIST bin
        usage - indices = np.random.shuffle(np.arange(n_samples)) 
        '''
        self.indices = indices # np.random.shuffle(np.arange(n_samples))
        self.bins    = (np.arange(n_samples)/(n_samples/n_bins)).astype('int') + bin_index
        self.n_bins  = n_bins
        self.bin_width = bin_width
        self.transported_samples = transported_samples
        self.target_bin = target_bin
        # self.angles  = self.bins*bin_width + np.random.randint(bin_width-5,bin_width-1,indices.shape)
        # self.normalized_angles  = self.angles/(bin_width * n_bins)
        root = '../../data/'
        processed_folder = os.path.join(root, 'MNIST', 'processed')
        data_file = 'training.pt'
        # print(self.bins,self.bin_width)
        # print("---------- READING MNIST ----------")
        self.data, self.targets = torch.load(os.path.join(processed_folder, data_file))

    def __getitem__(self,idx):
        index = self.indices[idx]
        bin   = torch.tensor(self.bins[idx]).to(device).float()
        # angle = self.angles[idx]
        angle = bin.item() * self.bin_width + np.random.randint(5)
        norm_angle = 1.0*angle/(self.bin_width * self.n_bins)
        image = self.data[index]
        image = Image.fromarray(image.numpy(), mode='L')

        target = self.targets[index]
        image = np.array(rotate(image,angle))#).float().to(device)
        image = torch.tensor(image).to(torch.float).to(device)/(255.0)
        target = target.to(device)

        if self.transported_samples is not None:
            transported_X = self.transported_samples[self.bins[idx]][self.target_bin][idx % 1000] #This should be similar to index fun    #.transform(X_item[
            return image,torch.cat([(bin/self.n_bins).float().view(1),torch.tensor(norm_angle).float().to(device).view(1)],dim=0).to(device),target, torch.from_numpy(transported_X).float().to(device)
        
        # print(bin,norm_angle)

        return image,torch.cat([(bin/self.n_bins).float().view(1),torch.tensor(norm_angle).float().to(device).view(1)],dim=0).to(device),target

    def __len__(self):
        return len(self.indices)