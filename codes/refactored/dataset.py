import torch
import numpy as np
import os

from utils import get_closest
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

class ClassificationDataSet(torch.utils.data.Dataset):
	
	def __init__(self, indices, transported_samples=None,target_bin=None, **kwargs):
		'''
		TODO: Handle OT
		Pass Transported samples as kwargs?
		'''
		self.indices = indices # Indices are the indices of the elements from the arrays which are emitted by this data-loader
		self.transported_samples = transported_samples  # a 2-d array of OT maps
		
		self.root = kwargs['root_dir']
		self.device = kwargs['device'] if kwargs.get('device') else 'cpu'
		self.transport_idx_func = kwargs['transport_idx_func'] if kwargs.get('transport_idx_func') else lambda x:x%1000
		self.num_bins = kwargs['num_bins'] if kwargs.get('num_bins') else 6
		self.base_bin = kwargs['num_bins'] if kwargs.get('num_bins') else 0   # Minimum whole number value of U
		#self.num_bins = kwargs['num_bins']  # using this we can get the bin corresponding to a U value
		
		self.target_bin = target_bin
		self.X = np.load("{}/X.npy".format(self.root))
		self.Y = np.load("{}/Y.npy".format(self.root))
		self.A = np.load("{}/A.npy".format(self.root))
		self.U = np.load("{}/U.npy".format(self.root))
		self.drop_cols = kwargs['drop_cols_classifier'] if kwargs.get('drop_cols_classifier') else None
		
	def __getitem__(self,idx):

		index = self.indices[idx]
		data = torch.tensor(self.X[index]).float().to(self.device)   # Check if we need to reshape
		label = torch.tensor(self.Y[index]).long().to(self.device)
		auxiliary = torch.tensor(self.A[index]).float().to(self.device).view(-1, 1)
		domain = torch.tensor(self.U[index]).float().to(self.device).view(-1, 1)
		if self.transported_samples is not None:
			source_bin = int(np.round(domain.item() * self.num_bins)) # - self.base_bin
			# print(source_bin,self.target_bin)
			transported_X = torch.from_numpy(self.transported_samples[source_bin][self.target_bin][self.transport_idx_func(idx)]).float().to(self.device) #This should be similar to index fun, an indexing function which takes the index of the source sample and returns the corresponding index of the target sample.
			# print(source_bin,self.target_bin,transported_X.size())
			if self.drop_cols is not None:
				return data[:self.drop_cols],transported_X[:self.drop_cols], auxiliary, domain,  label
			return data,transported_X, auxiliary, domain,  label

		if self.drop_cols is not None:
			return data[:self.drop_cols], auxiliary, domain, label
		return data, auxiliary, domain, label

	def __len__(self):
		return len(self.indices)


class GradDataset(torch.utils.data.Dataset):
	'''
	Class which returns (image,label,angle,bin)
	TODO - 1. Shuffle indices, bin and append angles: Isn't shuffling done by DataLoader that wraps this dataset instance?
		   2. Return 
		   3. OT - make sure OT takes into account the labels, i.e. OT loss should be inf for interchanging labels.  
	'''

	def __init__(self, data_path, source_indices, target_indices, target_bin=None, n_samples=6000, **kwargs):
	
		'''
		You give it a set of indices, along with which bins they belong
		It returns images from that MNIST bin
		usage - indices = np.random.shuffle(np.arange(n_samples)) 
		'''

		self.root = data_path
		
		self.src_indices = source_indices # np.random.shuffle(np.arange(n_samples))
		self.target_indices = target_indices # np.random.shuffle(np.arange(n_samples))
		self.target_labs = {}
		
		self.X = np.load("{}/X.npy".format(self.root))
		self.Y = np.load("{}/Y.npy".format(self.root))
		self.A = np.load("{}/A.npy".format(self.root))
		self.U = np.load("{}/U.npy".format(self.root))
		
		self.device = kwargs['device'] if kwargs.get('device') else 'cpu'
		self.drop_cols = kwargs['drop_cols'] if kwargs.get('drop_cols') else None
		self.rand_target = kwargs['rand_target'] if kwargs.get('rand_target') else False
		self.append_label = kwargs['append_label'] if kwargs.get('append_label') else False
		self.label_dict_func = kwargs['label_dict_func'] if kwargs.get('label_dict_func') else lambda x: int(x)
		self.return_binary   = kwargs['return_binary'] if kwargs.get('return_binary') else False
		self.map_index_curric = kwargs['map_index_curric'] if kwargs.get('map_index_curric') else dict([(i,0) for i in range(len(self.src_indices))])
		if isinstance(self.target_indices[0],int):
			self.target_indices = [self.target_indices]

		self.target_bin = target_bin

		# print(self.bins,self.bin_width)
		# print("---------- READING MNIST ----------")
		if self.rand_target == False:
			for idx,l in enumerate(self.target_indices):
				# print(idx)
				for i in l:
					if self.label_dict_func(self.Y[i].item()) not in self.target_labs.keys():
						self.target_labs[self.label_dict_func(self.Y[i].item())] = {idx : [i]}
					elif idx not in self.target_labs[self.label_dict_func(self.Y[i].item())].keys():
						self.target_labs[self.label_dict_func(self.Y[i].item())][idx] = [i]
					else:
						self.target_labs[self.label_dict_func(self.Y[i].item())][idx].append(i)
		# for k in self.target_labs:
		#   print(self.target_labs[k].keys())
	def __getitem__(self, idx):
		
		index = self.src_indices[idx]
		
		data   = torch.tensor(self.X[index]).float()
		label  = torch.tensor(self.Y[index])
		a_info = torch.tensor(self.A[index]).float()
		
		if self.rand_target:
			target_idx = np.random.randint(idx,len(self.target_indices[self.map_index_curric[idx]]))#idx % len(self.target_indices)
			target_idx = self.target_indices[self.map_index_curric[idx]][target_idx]
		else:
			try:
				target_ids = self.target_labs[self.label_dict_func(label.item())][self.map_index_curric[idx]]
			except KeyError:
				target_ids = self.target_labs[get_closest(list([k for k in self.target_labs.keys() if self.map_index_curric[idx] in self.target_labs[k].keys()]),self.label_dict_func(label.item()))][self.map_index_curric[idx]] 
			target_idx = target_ids[np.random.randint(0,len(target_ids))]

		# target_idx = self.target_indices
		target_data   = torch.tensor(self.X[target_idx]).float()
		a_info_target = torch.tensor(self.U[target_idx]).float()

		if self.drop_cols is not None:
			target_data = target_data[:self.drop_cols]
			data        = data[:self.drop_cols]

		# label = self.Y[index]
		if self.append_label:
			data         = torch.cat([data,label.view(1).float()/5],dim=0)
			target_label = torch.tensor(self.Y[target_idx])/5
			target_data  = torch.cat([target_data,target_label.view(1).float()],dim=0)

		# print(bin,norm_angle)
		if self.return_binary:
			time = 1.0 * ((a_info_target-a_info) > 0.)
		else:
			time = 10*(a_info_target-a_info)
		return data.to(self.device),target_data.to(self.device), time.float().to(self.device)#, U[index].float().to(self.device), label.long().to(self.device)
	
	def __len__(self):
		return len(self.src_indices)        



class MetaDataset(torch.utils.data.Dataset):

	def __init__(self, indices, boost_weights=None,testing=False, **kwargs):
		self.indices = indices # Indices are the indices of the elements from the arrays which are emitted by this data-loader
		
		self.root = kwargs['root_dir']
		self.device = kwargs['device'] if kwargs.get('device') else 'cpu'
		self.num_bins = kwargs['num_bins'] if kwargs.get('num_bins') else 6
		self.base_bin = kwargs['num_bins'] if kwargs.get('num_bins') else 0   # Minimum whole number value of U
		self.pretrain = kwargs['pretrain'] if kwargs.get('pretrain') else False
		#self.num_bins = kwargs['num_bins']  # using this we can get the bin corresponding to a U value
		self.append_label = kwargs['append_label'] if kwargs.get('append_label') else False
		
		self.X = np.load("{}/X.npy".format(self.root))
		self.Y = np.load("{}/Y.npy".format(self.root))
		# self.A = np.load("{}/A.npy".format(self.root))
		self.U = np.load("{}/U.npy".format(self.root))

		if testing:
			self.indices = indices[int(len(indices)*kwargs["test_ratio"]):]

		self.W = boost_weights
		# print("METADATA",self.X.shape)
		self.drop_cols = kwargs['drop_cols_classifier'] if kwargs.get('drop_cols_classifier') else None
		
	def __getitem__(self,idx):

		index = self.indices[idx]
		data = torch.tensor(self.X[index]).float().to(self.device)   # Check if we need to reshape
		label = torch.tensor(self.Y[index]).long().to(self.device)
		# auxiliary = torch.tensor(self.A[index]).float().to(self.device).view(-1, 1)
		domain = torch.tensor(self.U[index]).float().to(self.device).view(-1, 1)

		domain_next = domain + (1/self.num_bins)

		if self.drop_cols is not None:
			data = data[:self.drop_cols]

		if self.append_label:
			data         = torch.cat([data,label.view(1).float()/10],dim=0)

		if self.pretrain:
			data = torch.cat([data,domain.view(1)],dim=0)
		else:
			data = torch.cat([data,domain_next.view(1)],dim=0)


		if self.W is not None:
			return data,domain,torch.tensor(self.W[index]).float().to(self.device).view(-1, 1), label
		return data, domain, domain_next, label

	def __len__(self):
		return len(self.indices)
