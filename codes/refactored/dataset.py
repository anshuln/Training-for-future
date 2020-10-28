import torch
import numpy as np
import os
class RotMNIST(torch.utils.data.Dataset):
	'''
	Class which returns (image,label,angle,bin)
	TODO - 1. Shuffle indices, bin and append angles
		   2. Return 
		   3. OT - make sure OT takes into account the labels, i.e. OT loss should be inf for interchanging labels.  
	'''
	def __init__(self,indices,transported_samples=None,target_bin=None,**kwargs):
		'''

		'''
		self.indices = indices # Indices are the indices of the elements from the arrays which are emitted by this data-loader
		self.transported_samples = transported_samples  # a 2-d array of OT maps
		root = kwargs['data_path']
		self.num_bins = kwargs['num_bins']  # using this we can get the bin corresponding to a U value
		self.target_bin = target_bin
		self.X = np.load("{}/X.npy".format(root))
		self.Y = np.load("{}/Y.npy".format(root))
		self.A = np.load("{}/A.npy".format(root))
		self.U = np.load("{}/U.npy".format(root))
		self.device = kwargs['device']
	def __getitem__(self,idx):
		index = self.indices[idx]
		X = torch.tensor(self.X[idx]).float().to(self.device)   # Check if we need to reshape
		Y = torch.tensor(self.Y[idx]).long().to(self.device)
		A = torch.tensor(self.A[idx]).float().to(self.device).view(1)
		U = torch.tensor(self.U[idx]).float().to(self.device).view(1)
		if self.transported_samples is not None:
			source_bin = int(np.round(U.item() * self.num_bins)) 
			transported_X = torch.from_numpy(self.transported_samples[source_bin][self.target_bin][idx % 1000]).float().to(self.device) #This should be similar to index fun, an indexing function which takes the index of the source sample and returns the corresponding index of the target sample.
			return X,transported_X,A,U,Y

		return X,A,U,Y

	def __len__(self):
		return len(self.indices)
