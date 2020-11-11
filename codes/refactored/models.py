import torch
from torch import nn
import torchvision as tv
from torchvision import models as tv_models

DEVICE = "cuda:0"

class TimeReLU(nn.Module):

	"""

	TimeReLU docstring

	"""

	def __init__(self, data_shape, time_shape, leaky=False):

		super(TimeReLU, self).__init__()

		self.leaky = leaky
		self.model = nn.Linear(time_shape, data_shape)
		
		self.time_dim = time_shape        

		if self.leaky:
			self.model_alpha = nn.Linear(time_shape, data_shape)

		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, X, times):

		thresholds = self.model(times)
		orig_shape = X.size()
		# print(orig_shape)
		X = X.view(orig_shape[0],-1)
		if len(times.size()) == 3:
			times = times.squeeze(2)
		# print(X.size(),times.size(),thresholds.size())

		if self.leaky:
			alphas = self.model_alpha(times)
		
		else:
			alphas = 0.0

		X = torch.where(X>thresholds,X-thresholds,alphas*(X-thresholds))
		return X.view(orig_shape)

class Time2Vec(nn.Module):

	'''
	Time encoding inspired by the Time2Vec paper
	'''

	def __init__(self, in_shape, out_shape):

		super(Time2Vec, self).__init__()
		linear_shape = out_shape//4
		dirac_shape = 0
		sine_shape = out_shape - linear_shape - dirac_shape
		self.model_0 = nn.Linear(in_shape, linear_shape)
		self.model_1 = nn.Linear(in_shape, sine_shape)
		#self.model_2 = nn.Linear(in_shape, dirac_shape)

	def forward(self, X):
		if len(X.size()) == 3:
			X = X.squeeze(2)
		te_lin = self.model_0(X)
		te_sin = torch.sin(self.model_1(X))
		#te_dir = torch.max(10, torch.exp(-(self.model_2(X))))
		te = torch.cat([te_lin, te_sin], axis=1)
		return te

class Encoder(nn.Module):

	"""

	Encoder module docstring.

	"""
	'''Deprecated class for Encoder module
	
	We now use VGG for encoding. Also try an end-2-end encoder
	'''

	def __init__(self, input_shape, hidden_shapes, encoding_shape, **kwargs):

		super(Encoder, self).__init__()
		assert(len(hidden_shapes) >= 0)

		self.time_conditioning = kwargs['time_conditioning'] if kwargs.get('time_conditioning') else False
		if self.time_conditioning:
			self.leaky = kwargs['leaky'] if kwargs.get('leaky') else False
			self.use_time2vec = kwargs['use_time2vec'] if kwargs.get('use_time2vec') else False

		self.layers = nn.ModuleList()
		self.relus = nn.ModuleList()

		self.input_shape = input_shape
		self.hidden_shapes = hidden_shapes
		self.encoding_shape = encoding_shape
	
		if len(self.hidden_shapes) == 0:

			self.layers.append(nn.Linear(input_shape, encoding_shape))
			self.relus.append(nn.LeakyReLU())

		else:

			self.layers.append(nn.Linear(self.input_shape, self.hidden_shapes[0]))
			self.relus.append(nn.LeakyReLU())

			for i in range(len(self.hidden_shapes) - 1):

				self.layers.append(nn.Linear(self.hidden_shapes[i], self.hidden_shapes[i+1]))
				self.relus.append(nn.LeakyReLU())

			self.layers.append(nn.Linear(self.encoding_shape, self.hidden_shapes[-1]))
			self.relus.append(nn.LeakyReLU())

	def forward(self, X, times = None):
		
		for i in range(len(self.layers)):

			X = self.relus[i](self.layers[i](X))

		return X

class EncoderCNN(nn.Module):

	"""

	Encoder module docstring.

	"""
	'''Deprecated class for Encoder module
	
	We now use VGG for encoding. Also try an end-2-end encoder
	'''

	def __init__(self):

		super(EncoderCNN, self).__init__()
		self.model = tv_models.vgg16(pretrained=True).features[:16]
	def forward(self,X):
		X = self.model(X).view(-1,16,28,28)
		return X

class ClassifyNet(nn.Module):

	"""

	ClassifyNet module docstring.

	"""
	'''Deprecated class for ClassifyNet module
	
	We now use VGG for encoding. Also try an end-2-end ClassifyNet
	'''

	def __init__(self, input_shape, hidden_shapes, output_shape, **kwargs):

		super(ClassifyNet, self).__init__()
		assert(len(hidden_shapes) >= 0)

		self.time_conditioning = kwargs['time_conditioning'] if kwargs.get('time_conditioning') else False
		if self.time_conditioning:
			self.leaky = kwargs['leaky'] if kwargs.get('leaky') else False
			self.use_time2vec = kwargs['use_time2vec'] if kwargs.get('use_time2vec') else False

		self.layers = nn.ModuleList()
		self.relus = nn.ModuleList()

		self.input_shape = input_shape
		self.hidden_shapes = hidden_shapes
		self.output_shape = output_shape
	
		if len(self.hidden_shapes) == 0:

			self.layers.append(nn.Linear(input_shape, output_shape))
			self.relus.append(nn.LeakyReLU())

		else:

			self.layers.append(nn.Linear(self.input_shape, self.hidden_shapes[0]))
			self.relus.append(nn.LeakyReLU())

			for i in range(len(self.hidden_shapes) - 1):

				self.layers.append(nn.Linear(self.hidden_shapes[i], self.hidden_shapes[i+1]))
				self.relus.append(nn.LeakyReLU())

			self.layers.append(nn.Linear(self.output_shape, self.hidden_shapes[-1]))
			self.relus.append(nn.LeakyReLU())

	def forward(self, X, times = None):
		
		for i in range(len(self.layers)):

			X = self.relus[i](self.layers[i](X))

		X = nn.sigmoid(X)

		return X

class ClassifyNetCNN(nn.Module):

	def __init__(self,data_shape, hidden_shape, n_classes, append_time=False,encode_time=False,time_conditioning=True,leaky=True,use_vgg=False,time2vec=True,):
		
		super(ClassifyNetCNN,self).__init__()
		# assert(len(hidden_shapes) >= 0)

		# self.time_conditioning = kwargs['time_conditioning'] if kwargs.get('time_conditioning') else False
		# if self.time_conditioning:
		# 	self.leaky = kwargs['leaky'] if kwargs.get('leaky') else False
		# 	self.use_time2vec = kwargs['use_time2vec'] if kwargs.get('use_time2vec') else False
		self.time_conditioning = time_conditioning
		self.append_time = append_time
		self.encode_time = encode_time

		self.layers = nn.ModuleList()
		self.relus = nn.ModuleList()

		self.input_shape = data_shape
		self.hidden_shapes = hidden_shape
		self.output_shape = n_classes
		if use_vgg:
			in_channels = 16
		else:
			in_channels = 1	
		# if len(self.hidden_shapes) == 0:

		# 	self.layers.append(nn.Linear(input_shape, output_shape))
		# 	self.relus.append(nn.LeakyReLU())

		if time2vec:
			self.time2vec = Time2Vec(1, 8)
			self.time_dim = 8
		else:
			self.time2vec = None
			self.time_dim = 1

		self.flat = nn.Flatten()
		self.layer_0 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3)
		self.relu_0    = TimeReLU(13*13*8,self.time_dim,leaky)
		# self.relu_0 = nn.ReLU()
		if time_conditioning:
			self.layer_1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
			self.layer_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
			self.relu_1 = TimeReLU(200*2,self.time_dim,leaky)
			self.relu_2 = TimeReLU(72*4,self.time_dim,leaky)
			# self.relu_t = nn.ReLU()
		self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)
		self.out_layer = nn.Linear(72*4,n_classes)
		self.out_relu = TimeReLU(n_classes,self.time_dim,leaky)


	def forward(self,X,times):
		if len(X.size()) < 4:
			X = X.unsqueeze(1)
		
		if self.encode_time:
			X = self.time_enc(X,times)

		if self.append_time:
			times_ = times.unsqueeze(1).unsqueeze(2).repeat(1,1,28,28)
			X = torch.cat([X,times],dim=1)

		if self.time2vec is not None:
			times = self.time2vec(times)

		X  = self.layer_0(X)
		X  = self.down_sampling(X)
		X  = self.relu_0(X,times)


		if self.time_conditioning:
			X = self.layer_1(X)
			X = self.down_sampling(X)
			X = self.relu_1(X,times)
			X = self.layer_2(X)
			X = self.relu_2(X,times)

		# print(X.size())
		X = self.out_layer(X.view(X.size(0),-1))
		X = self.out_relu(X,times)
		X = torch.softmax(X,dim=1)

		return X

class GradNet(nn.Module):

	def __init__(self, data_shape, hidden_shapes, **kwargs):
		
		super(GradNet,self).__init__()

		self.layers = nn.ModuleList()
		self.relus = nn.ModuleList()

		self.input_shape = data_shape
		self.hidden_shapes = hidden_shapes

		self.loss_type = kwargs['loss_type'] if kwargs.get('loss_type') else False
		
		if self.loss_type == 'mse':
			self.output_shape = 1

		elif self.loss_type == 'cosine':
			self.output_shape = kwargs['time_dim'] if kwargs['time_dim'] else 8 # raise some error here
	
		# if len(self.hidden_shapes) == 0:

		# 	self.layers.append(nn.Linear(input_shape, output_shape))
		# 	self.relus.append(nn.LeakyReLU())
			
		# else:

		# 	self.layers.append(nn.Linear(self.input_shape, self.hidden_shapes[0]))
		# 	self.relus.append(nn.LeakyReLU())

		# 	for i in range(len(self.hidden_shapes) - 1):

		# 		self.layers.append(nn.Linear(self.hidden_shapes[i], self.hidden_shapes[i+1]))
		# 		self.relus.append(nn.LeakyReLU())

		# 	self.layers.append(nn.Linear(2*self.hidden_shapes[-1], self.hidden_shapes[-1]))
		# 	self.layers.append(nn.Linear(self.hidden_shapes[-1], self.output_shape))
		# 	self.relus.append(nn.LeakyReLU())
		# 	self.relus.append(nn.LeakyReLU())


	def forward(self, X1, X2):

		# Distinct part

		for i in range(len(self.layers-2)):

			X1 = self.relus[i](self.layers[i](X1))
			X2 = self.relus[i](self.layers[i](X2))
		# Common part

		X = self.relus[-1](self.layers[-1](self.relus[-2](self.layers[-2](torch.cat([X1 ,X2])))))
		return X
		