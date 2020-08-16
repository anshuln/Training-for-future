# # Contains the model definition for the classifier
# # We are considering two approaches - 
# # 1. Transformer does heavy lifting and just gives a transformed distribution to the classifier
# # 2. Classifier does heavy lifting and tries to learn P(y|X,t) with special emphasis on time = T+1

from torch import nn
import torch

class ClassifyNet(nn.Module):
	def __init__(self,data_shape, hidden_shape, out_shape, time_conditioning=False):
		super(ClassifyNet,self).__init__()

		self.time_conditioning = time_conditioning

		self.layer_0 = nn.Linear(data_shape,hidden_shape)
		self.relu    = nn.ReLU()

		if time_conditioning:
			self.layer_1 = nn.Linear(hidden_shape,hidden_shape)
			self.layer_2 = nn.Linear(hidden_shape,hidden_shape)

		self.out_layer = nn.Linear(hidden_shape,out_shape)

	def forward(self,X):
		X  = self.layer_0(X)
		X  = self.relu(X)

		if self.time_conditioning:
			X = self.layer_1(X)
			X = self.relu(X)
			X = self.layer_2(X)
			X = self.relu(X)

		X = self.out_layer(X)
		X = self.relu(X)
		X = torch.softmax(X,dim=1)

		return X


class Transformer(nn.Module):
	def __init__(self,data_shape, latent_shape):

		super(Transformer,self).__init__()
		self.layer_0 = nn.Linear(data_shape,latent_shape)
		self.layer_1 = nn.Linear(latent_shape,data_shape-2)
		self.leaky_relu = nn.LeakyReLU()

	def forward(self,X):
		X = self.leaky_relu(self.layer_1(self.leaky_relu(self.layer_0(X))))
		return X


class Discriminator(nn.Module):
	def __init__(self,data_shape, hidden_shape, is_wasserstein=False):

		super(Discriminator,self).__init__()
		self.layer_0   = nn.Linear(data_shape,hidden_shape)
		self.relu      = nn.ReLU()
		self.out_layer = nn.Linear(hidden_shape,1)
		self.is_wasserstein = is_wasserstein

	def forward(self,X):
		X = self.out_layer(self.relu(self.layer_0(X)))

		if not self.is_wasserstein:
			X = torch.sigmoid(X)
			
		return X


def classification_loss(Y_pred, Y):
	return  -1.*torch.sum((Y * torch.log(Y_pred)))

def bxe(real, fake):
	return -1.*((real*torch.log(fake)) + ((1-real)*torch.log(1-fake)))

def discriminator_loss(real_output, trans_output):

	# bxe = tf.keras.losses.BinaryCrossentropy(from_logits=True)

	real_loss = bxe(torch.ones_like(real_output), real_output)
	trans_loss = bxe(torch.zeros_like(trans_output), trans_output)
	total_loss = real_loss + trans_loss
	
	return total_loss.mean()

def reconstruction_loss(x,y):
	return torch.sum((x-y)**2,dim=1)


def transformer_loss(trans_output):

	return bxe(torch.ones_like(trans_output), trans_output)

def discounted_transformer_loss(real_data, trans_data, trans_output):

	time_diff = torch.exp(-(real_data[:,-1] - real_data[:,-2]))
	re_loss = reconstruction_loss(real_data[:,0:-2], trans_data)
	tr_loss = transformer_loss(trans_output)

	loss = torch.mean(time_diff * tr_loss + (1-time_diff) * re_loss)
	return loss

