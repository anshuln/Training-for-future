'''Preprocessing and saving all datasets

This file saves all final datasets, which the dataloader shall read and give out.
'''
import pandas as pd
import numpy as np
import math

from datetime import datetime

# from sklearn.datasets import make_classification, make_moons
from torchvision.transforms.functional import rotate
from torchvision.datasets.folder import  has_file_allowed_extension, is_image_file, IMG_EXTENSIONS, pil_loader, accimage_loader,default_loader
from tqdm import tqdm 

import torch
import os
import json
from PIL import Image
from utils import *

MOON_SAMPLES = 200
ROTNIST_SAMPLES = 1000

def load_sleep(filename):

    domains = 5
    
    df =  pd.read_csv(filename)
    df = df.drop(['rcrdtime'], axis=1)
    nan_values = dict()
    for col in df.columns:
        nan_values[col] = df[col].isna().sum()

    df = df.dropna(subset=['Staging1','Staging2','Staging3','Staging4','Staging5'])
    final_cols = []
    for col in nan_values.keys():
        if nan_values[col] <= 500:
            final_cols.append(col)

    print(len(final_cols))
    df = df[final_cols]
    imputer = SimpleImputer(strategy='mean')
    #imputer = KNNImputer(n_neighbors=3, weights="uniform")
    df = pd.DataFrame(imputer.fit_transform(df), columns = df.columns)
    print(df.shape)


    X_data, Y_data, A_data, U_data = [], [], [], []
    indices = []
    index_len = 0
    ckpts = [50, 60, 70, 80, 90]

    for i, ckpt in enumerate(ckpts):

        data_temp = df[df['age_s1'] <= ckpt]
        df = df[df['age_s1'] > ckpt]
        Y_temp = data_temp['Staging1'].values
        Y_temp = np.eye(2)[Y_temp.astype(np.int32)]   # Can we change this to 0/1 labels 
        #A_temp = (data_temp['age_s1'].values-39)/90
        A_temp = (data_temp['age_s1'].values-38)/90
        data_temp = data_temp.drop(['Staging1'], axis=1)
        X_temp = data_temp.drop(['Staging2', 'Staging3', 'Staging4', 'Staging5', 'age_s1', 'age_category_s1'], axis=1).values
        #U_temp = np.array([i]*X_temp.shape[0])*1.0/5
        U_temp = np.array([i+1]*X_temp.shape[0])*1.0/5
        print(X_temp.shape)
        print(Y_temp.shape)
        print(A_temp.shape)
        print(U_temp.shape)
        indices.append(np.arange(index_len,index_len+X_temp.shape[0]))
        index_len += X_temp.shape[0]

        X_temp =X_temp.astype(np.float32)
        A_temp = A_temp.astype(np.float32)
        U_temp =U_temp.astype(np.float32)
        
        X_data.append(X_temp)
        Y_data.append(Y_temp)
        A_data.append(A_temp)
        U_data.append(U_temp)

    np.save(np.array(X_data))
    np.save(np.array(Y_data))
    np.save(np.array(A_data))
    np.save(np.array(U_data))
    # Save indices

def load_moons(domains, root='../../data'):

    """

    Save the Rotated 2-moons dataset to an npy file

    Parameters
    ----------

    domains : No of domains for which samples are to be generated. Each domain is an 18 degree rotation of the previous domain

    root : Root directory where the .npy files sill be stored. The files are stored under `{root}/Moons/processed`

    """

    X_data, Y_data, A_data, U_data = [], [], [], []
    processed_folder = os.path.join(root, 'Moons', 'processed')

    for i in range(domains):

        angle = i*math.pi/(domains-1)

        X, Y = make_moons(n_samples=200, noise=0.1)
        rot = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
        X = np.matmul(X, rot)

        Y = np.eye(2)[Y]  
        U = np.array([i//domains] * 200)
        A = np.array([i//domains] * 200)

        X_data.append(X)
        Y_data.append(Y)
        A_data.append(A)
        U_data.append(U)

    all_indices = [[x for x in range(i*MOON_SAMPLES,(i+1)*MOON_SAMPLES)] for i in range(domains)]

    np.save("{}/X.npy".format(processed_folder), X_data, allow_pickle=True)
    np.save("{}/Y.npy".format(processed_folder), Y_data, allow_pickle=True)
    np.save("{}/A.npy".format(processed_folder), A_data, allow_pickle=True)
    np.save("{}/U.npy".format(processed_folder), U_data, allow_pickle=True)
    np.save("{}/indices.npy".format(processed_folder), all_indices, allow_pickle=True)
    #json.dump(all_indices, open("{}/indices.json".format(processed_folder),"w"))

def load_Rot_MNIST(use_vgg,root="../../data"):

    mnist_ind = (np.arange(60000))
    np.random.shuffle(mnist_ind)
    mnist_ind = mnist_ind[:6000]
    # Save indices
    processed_folder = os.path.join(root, 'MNIST', 'processed')
    data_file = 'training.pt'
    vgg_means = np.array([0.485, 0.456, 0.406]).reshape((3,1,1))
    vgg_stds  = np.array([0.229, 0.224, 0.225]).reshape((3,1,1))
    data, targets = torch.load(os.path.join(processed_folder, data_file))
    all_images = []
    all_labels = []
    all_U = []
    all_A = []
    all_indices = [[x for x in range(i*ROTNIST_SAMPLES,(i+1)*ROTNIST_SAMPLES)] for i in range(6)]
    for idx in range(len(mnist_ind)):
        index = mnist_ind[idx]
        bin = int(idx / 1000)
        angle = bin * 15
        image = data[index]
        image = Image.fromarray(image.numpy(), mode='L')
        image = np.array(rotate(image,angle))#).float().to(device)
        image = image / 255.0
        if use_vgg:
            image = image.reshape((1,28,28)).repeat(3,axis=0)
            image = (image - vgg_means)/vgg_stds
        else:
            image = image.reshape((1,28,28))
            # image = (image - vgg_means)/vgg_stds

        all_images.append(image)
        all_labels.append(targets[index])
        all_U.append(bin/6)
        all_A.append(angle/90)

    np.save("{}/X.npy".format(processed_folder),np.stack(all_images),allow_pickle=True)
    np.save("{}/Y.npy".format(processed_folder),np.array(all_labels),allow_pickle=True)
    np.save("{}/A.npy".format(processed_folder),np.array(all_A),allow_pickle=True)
    np.save("{}/U.npy".format(processed_folder),np.array(all_U),allow_pickle=True)
    json.dump( all_indices,open("{}/indices.json".format(processed_folder),"w"))
    # json.dump(all_indices, open("{}/indices.json".format(processed_folder),"w"))


def load_comp_cars(root_dir="../../data/CompCars", text_file="../../data/CompCars/adagraph_filtered.txt",out_size=(3,224,224)):
    file = open(text_file,"r")
    all_images = []
    all_labels = []
    all_U = []
    all_A = []
    indices = {} 
    counter = 0
    failed = 0
    for line in tqdm(file.readlines()):
        #./data/compcars/78/1/2010/09085b4e3c1674.jpg 2010-5 4
        # if len(line) == 0:
        #   continue
        try:
            path,domain,label = line.strip().split(' ')
        except:
            continue
        try:
            img = default_loader("{}/{}".format(root_dir,path))
        except:
            failed += 1
            continue
        img = img.resize((out_size[1],out_size[2]))

        if domain in indices:
            indices[domain].append(counter)
        else:
            indices[domain] = [counter]
        all_labels.append(int(label))
        all_images.append(np.array(img).reshape(out_size))
        all_U.append([int(domain.split('-')[0]),int(domain.split('-')[1])])
        all_A.append([(float(domain.split('-')[0])-2009)/6,float(domain.split('-')[1])/5])
        counter += 1
        # if counter % 1000 == 1:
        #   np.save("{}/X.npy".format(root_dir),np.stack(all_images),allow_pickle=True)
        #   np.save("{}/Y.npy".format(root_dir),np.stack(all_labels),allow_pickle=True)
        #   np.save("{}/A.npy".format(root_dir),np.array(all_A),allow_pickle=True)
        #   np.save("{}/U.npy".format(root_dir),np.array(all_U),allow_pickle=True)
        #   json.dump(indices,open("{}/indices.json".format(root_dir),"w"))
    np.save("{}/X.npy".format(root_dir),np.stack(all_images),allow_pickle=True)
    np.save("{}/Y.npy".format(root_dir),np.stack(all_labels),allow_pickle=True)
    np.save("{}/A.npy".format(root_dir),np.array(all_A),allow_pickle=True)
    np.save("{}/U.npy".format(root_dir),np.array(all_U),allow_pickle=True)
    json.dump(indices,open("{}/indices.json".format(root_dir),"w"))
    print(f"FAILED ON {failed}")
# load_comp_cars()

def load_house_price(root_dir="../../data/HousePrice", text_file="../../data/HousePrice/raw_sales.csv"):
    all_X = []
    all_labels = []
    # all_U = []
    # all_A = []
    indices = {} 
    failed = 0

    df = pd.read_csv(text_file)
    # print(len(pd.unique(df['postcode'])))
    onehot = pd.get_dummies(df.postcode)
    df = df.drop("postcode",axis=1)
    df = df.join(onehot)

    onehot = pd.get_dummies(df.propertyType)
    df = df.drop("propertyType",axis=1)
    df = df.join(onehot)

    # pd.to_datetime(df["datesold"])
    df = df.join(df.datesold.apply(lambda x:pd.to_datetime(x).timestamp()),rsuffix='stamp').drop("datesold",axis=1) #, axis=1)

    df = df.sort_values(by='datesoldstamp')

    # testtime = 1546300800 - 1.170806e+09

    # train  = df.loc[df["datesoldstamp"]<testtime,:]
    # test = df.loc[df["datesoldstamp"]>testtime,:]

    # label = train["price"]
    # data = train.loc[:, train.columns != 'price']
    data = df.to_numpy()
    for idx,row in enumerate(data):
        all_labels.append(row[0]/10000)

        u = int(datetime.fromtimestamp(int(row[-1])).year)
        all_X.append(np.array(row[1:].tolist()+[u]))
        # all_U.append(u)
        # all_A.append(row[-1])     
        if u in indices:
            indices[u].append(idx)
        else:
            indices[u] = [idx]
    # print(idx)
    all_X = np.stack(all_X)
    all_X = all_X - all_X.min(axis=0).reshape((1,-1))
    all_X = all_X / all_X.max(axis=0).reshape((1,-1))
    all_U = all_X[:,-1]
    all_A = all_X[:,-2]
    print(all_X[:3])
    new_ind = []
    for i in [2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]:
        print(len(indices[i]))
        new_ind.append(indices[i])
    # print(new_ind[-1][-1])
    # print(all_X,all_U,all_A)
    np.save("{}/X.npy".format(root_dir),all_X,allow_pickle=True)
    np.save("{}/Y.npy".format(root_dir),all_labels,allow_pickle=True)
    np.save("{}/A.npy".format(root_dir),all_A,allow_pickle=True)
    np.save("{}/U.npy".format(root_dir),all_U,allow_pickle=True)
    # print(new_ind)
    json.dump(new_ind,open("{}/indices.json".format(root_dir),"w"))
    # print(len(new_ind))

def load_house_price_classification(root_dir="../../data/HousePriceClassification", text_file="../../data/HousePriceClassification/raw_sales.csv"):
    all_X = []
    all_labels = []
    # all_U = []
    # all_A = []
    indices = {} 
    failed = 0

    df = pd.read_csv(text_file)
    # print(len(pd.unique(df['postcode'])))
    onehot = pd.get_dummies(df.postcode)
    df = df.drop("postcode",axis=1)
    df = df.join(onehot)

    onehot = pd.get_dummies(df.propertyType)
    df = df.drop("propertyType",axis=1)
    df = df.join(onehot)

    # pd.to_datetime(df["datesold"])
    df = df.join(df.datesold.apply(lambda x:pd.to_datetime(x).timestamp()),rsuffix='stamp').drop("datesold",axis=1) #, axis=1)

    df = df.sort_values(by='datesoldstamp')


    data = df.to_numpy()
    for idx,row in enumerate(data):
        # print(row)
        if row[1] == 0:
            continue
        all_labels.append(row[1]-1)  # 0-4, easier for training
        u = int(datetime.fromtimestamp(int(row[-1])).year)
        all_X.append(np.array([row[0]] + row[2:].tolist()+[u]))
        # all_U.append(u)
        # all_A.append(row[-1])     
        if u in indices:
            indices[u].append(idx)
        else:
            indices[u] = [idx]
    print(idx)
    all_X = np.stack(all_X)
    print(all_X)
    all_X = all_X - all_X.min(axis=0).reshape((1,-1))
    all_X = all_X / all_X.max(axis=0).reshape((1,-1))
    all_U = all_X[:,-1]
    all_A = all_X[:,-2]
    new_ind = []
    for i in [2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]:
        print(len(indices[i]))
        new_ind.append(indices[i])
    # print(new_ind[-1][-1])
    # print(all_X,all_U,all_A)
    np.save("{}/X.npy".format(root_dir),all_X,allow_pickle=True)
    np.save("{}/Y.npy".format(root_dir),all_labels,allow_pickle=True)
    np.save("{}/A.npy".format(root_dir),all_A,allow_pickle=True)
    np.save("{}/U.npy".format(root_dir),all_U,allow_pickle=True)
    # print(new_ind)
    json.dump(new_ind,open("{}/indices.json".format(root_dir),"w"))

if __name__ == '__main__':
    
    load_house_price()