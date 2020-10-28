'''Preprocessing and saving all datasets

This file saves all final datasets, which the dataloader shall read and give out.
'''
import pandas as pd
import numpy as np
import math
from sklearn.datasets import make_classification, make_moons
from torchvision.transforms.functional import rotate
import torch
import os
import json
from PIL import Image


def load_sleep2(filename):

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

		Y = np.eye(2)[Y]  # Can we change this to 0/1 labels 
		U = np.array([i] * 200)

		X_data.append(X)
		Y_data.append(Y)
		U_data.append(U)

	np.save(np.array(X_data))
	np.save(np.array(Y_data))
	np.save(np.array(A_data))
	np.save(np.array(U_data))

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
	all_indices = [[x for x in range(i*1000,(i+1)*1000)] for i in range(6)]
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

		all_images.append(image)
		all_labels.append(targets[index])
		all_U.append(bin/6)
		all_A.append(angle/90)

	np.save("{}/X.npy".format(processed_folder),np.stack(all_images),allow_pickle=True)
	np.save("{}/Y.npy".format(processed_folder),np.array(all_labels),allow_pickle=True)
	np.save("{}/A.npy".format(processed_folder),np.array(all_A),allow_pickle=True)
	np.save("{}/U.npy".format(processed_folder),np.array(all_U),allow_pickle=True)
	json.dump(all_indices,open("{}/indices.json".format(processed_folder),"w"))

load_Rot_MNIST(use_vgg=False)