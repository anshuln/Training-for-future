# # Contains the model definition for the classifier
# # We are considering two approaches - 
# # 1. Transformer does heavy lifting and just gives a transformed distribution to the classifier
# # 2. Classifier does heavy lifting and tries to learn P(y|X,t) with special emphasis on time = T+1
## TODO - 1. Conv
##        2. TimeEncoding
##        3. U-NET type structure?
from torch import nn
import torch
# import torchvision as tv
from torchvision import models as tv_models

device = "cuda:0"

class Encoder(nn.Module):
	"""docstring for Encoder"""
	'''Deprecated class for Encoder module
	
	We now use VGG for encoding. Also try an end-2-end encoder
	'''
	def __init__(self):
		super(Encoder, self).__init__()
		self.model = tv_models.vgg16(pretrained=True).features[:16]
	def forward(self,X):
		X = self.model(X).view(-1,16,28,28)
		return X

class EncoderCars(nn.Module):
	"""docstring for Encoder"""
	'''Deprecated class for Encoder module
	
	We now use VGG for encoding. Also try an end-2-end encoder
	'''
	def __init__(self):
		super(EncoderCars, self).__init__()
		self.model = tv_models.vgg16(pretrained=True).features
	def forward(self,X):
		X = self.model(X).view(-1,32,28,28)
		return X

class GradNet(nn.Module):
	def __init__(self,data_shape, hidden_shape, time_conditioning=True,leaky=False,use_vgg=False):
		super(GradNet,self).__init__()

		if use_vgg:
			in_channels = 16
		else:
			in_channels = 1

		# self.time_encodings = TimeEncodings(data_shape,1)
		self.layer_0 = nn.Conv2d(in_channels=in_channels,out_channels=32,kernel_size=3)
		self.relu_0    = nn.LeakyReLU()
		# self.relu_0 = nn.ReLU()
		# if time_conditioning:
		self.layer_1 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3)
		self.layer_2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3)
			# self.layer_2 = nn.Linear(hidden_shape,hidden_shape)
			# self.relu_t = nn.LeakyReLU()
		self.relu_t = nn.ReLU()
		self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)
		self.out_layer = nn.Linear(3200,1)
		self.out_relu = nn.LeakyReLU()

	def forward(self,X1,X2):
		# times = X[:,-1:]
		# X = self.time_encodings(X)
		# print(X1.size())
		if len(X1.size()) < 4:
			X1 = X1.unsqueeze(1)
		if len(X2.size()) < 4:
			X2 = X2.unsqueeze(1)
		X1  = self.layer_0(X1)
		X1  = self.relu_0(X1)

		# if self.time_conditioning:
		X1 = self.layer_1(X1)
		X1 = self.relu_t(X1)
		X1 = self.down_sampling(X1)
		X1 = self.layer_2(X1)
		X1 = self.relu_t(X1)

		X1 = self.down_sampling(X1)
		# X1 = self.layer_2(X1)
		# X1 = self.relu_t(X1)

		X2  = self.layer_0(X2)
		X2  = self.relu_0(X2)

		# if self.time_conditioning:
		X2 = self.layer_1(X2)
		X2 = self.relu_t(X2)
		X2 = self.down_sampling(X2)
		X2 = self.layer_2(X2)
		X2 = self.relu_t(X2)
		X2 = self.down_sampling(X2)
		# X2 = self.layer_2(X2)
		# X2 = self.relu_t(X2)
		# print(X1.size())
		X = self.out_layer(torch.cat([X1.view(X1.size(0),-1),X2.view(X1.size(0),-1)],dim=-1))
		# X = self.out_relu(X)
		# X = torch.softmax(X,dim=1)

		return X
		
		
class ClassifyNet(nn.Module):
	def __init__(self,data_shape, hidden_shape, n_classes, append_time=False,encode_time=False,time_conditioning=True,leaky=True,use_vgg=False):
		super(ClassifyNet,self).__init__()

		self.time_conditioning = time_conditioning
		self.append_time = append_time
		self.encode_time = encode_time
		if use_vgg:
			in_channels = 16
		else:
			in_channels = 1

		if self.append_time:
			in_channels += 2
		else:
			in_channels += 0
		if self.encode_time:
			self.time_enc = TimeEncodings(784,2)

		self.flat = nn.Flatten()
		self.layer_0 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3)
		self.relu_0    = TimeReLU(13*13*8,1,leaky)
		# self.relu_0 = nn.ReLU()
		if time_conditioning:
			self.layer_1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
			self.layer_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
			self.relu_1 = TimeReLU(200*2,1,leaky)
			self.relu_2 = TimeReLU(72*4,1,leaky)
			# self.relu_t = nn.ReLU()
		self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)
		self.out_layer = nn.Linear(72*4,n_classes)
		self.out_relu = TimeReLU(n_classes,1,leaky)

	def forward(self,X,times):
		if len(X.size()) < 4:
			X = X.unsqueeze(1)
		
		if self.encode_time:
			X = self.time_enc(X,times)

		if self.append_time:
			times_ = times.unsqueeze(1).unsqueeze(2).repeat(1,1,28,28)
			X = torch.cat([X,times],dim=1)

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


class ClassifyNetCars(nn.Module):
	def __init__(self,data_shape, hidden_shape, n_classes, append_time=False,encode_time=False,time_conditioning=True,leaky=True,use_vgg=False):
		super(ClassifyNetCars,self).__init__()

		self.time_conditioning = time_conditioning
		self.append_time = append_time
		self.encode_time = encode_time
		if use_vgg:
			in_channels = 32
		else:
			in_channels = 1

		if self.append_time:
			in_channels += 2
		else:
			in_channels += 0
		if self.encode_time:
			self.time_enc = TimeEncodings(784,2)

		self.flat = nn.Flatten()
		self.layer_0 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3)
		self.relu_0    = TimeReLU(13*13*8,2,leaky)
		# self.relu_0 = nn.ReLU()
		if time_conditioning:
			self.layer_1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
			self.layer_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
			self.relu_1 = TimeReLU(200*2,2,leaky)
			self.relu_2 = TimeReLU(72*4,2,leaky)
			# self.relu_t = nn.ReLU()
		self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)
		self.out_layer = nn.Linear(72*4,n_classes)
		self.out_relu = TimeReLU(n_classes,2,leaky)

	def forward(self,X,times):
		if len(X.size()) < 4:
			X = X.unsqueeze(1)
		
		if self.encode_time:
			X = self.time_enc(X,times)

		if self.append_time:
			times_ = times.unsqueeze(1).unsqueeze(2).repeat(1,1,28,28)
			X = torch.cat([X,times],dim=1)

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

class Transformer(nn.Module):
	def __init__(self,data_shape, latent_shape,append_time=False,encode_time=False, label_dim=0,use_vgg=False):
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
		 Potential tweak - Encode target time information separately and pass to decoder.
		'''
		super(Transformer,self).__init__()
		self.append_time = append_time
		self.encode_time = encode_time
		if use_vgg:
			in_channels_base = 16
		else:
			in_channels_base = 1
		if self.append_time:
			in_channels = in_channels_base + 4
		else:
			in_channels = in_channels_base +  0
		if self.encode_time:
			self.time_enc = TimeEncodings(784,2)
		self.flat = nn.Flatten()
		self.layer_0 = ConvBlock(in_channels=in_channels,out_channels=8*8,kernel_size=3,time_relu_size=6272*8,time_shape=2)
		self.layer_1 = ConvBlock(in_channels=8*8,out_channels=8*8,kernel_size=3,time_relu_size=3136*4,time_shape=2)
		self.layer_2 = ConvBlock(in_channels=8*8,out_channels=16*8,kernel_size=3,time_relu_size=784*8,time_shape=2)
		self.layer_3 = ConvBlock(in_channels=16*8,out_channels=8*8,kernel_size=3,time_relu_size=392*8,time_shape=2)
		self.layer_4 = ConvBlock(in_channels=8*8,out_channels=8*8,kernel_size=3,time_relu_size=1568*8,time_shape=2)
		self.layer_5 = ConvBlock(in_channels=8*8,out_channels=in_channels_base,kernel_size=3,time_relu_size=784*in_channels_base,time_shape=2)
		self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
		self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)
		# self.layer_0_2 = nn.Linear(latent_shape,latent_shape)

		self.label_dim = label_dim

	def forward(self,X,times):
		# times = X[:,-4:]
		# X = torch.cat([self.flat(X),times.view(-1,4)],dim=-1)
		if len(X.size()) < 4:
			X = X.unsqueeze(1)

		if self.encode_time:
			X = self.time_enc(X,times)

		if self.append_time:
			times_ = times.unsqueeze(1).unsqueeze(2).repeat(1,1,28,28)
			X = torch.cat([X,times],dim=1)

		X = self.layer_0(X,times)
		size_0 = X.size()
		X,ind_0 = self.down_sampling(X)
		X = self.layer_1(X,times)
		size_1 = X.size()
		X,ind_1 = self.down_sampling(X)
		X = self.layer_2(X,times)
		size_2 = X.size()
		X,ind_2 = self.down_sampling(X)

		X = self.up_sampling(X,ind_2,output_size=size_2)
		X = self.layer_3(X,times)
		X = self.up_sampling(X,ind_1,output_size=size_1)
		X = self.layer_4(X,times)
		X = self.up_sampling(X,ind_0,output_size=size_0)
		X = self.layer_5(X,times)

		if self.label_dim:
			lab = torch.sigmoid(X[:,-1])
			# X_new = self.leaky_relu(X[:,:-1])
			X = torch.cat([X,lab.unsqueeze(1)],dim=1)
		# else:
		#     X = self.leaky_relu(X)
		# print(X.size())
		return X.squeeze(1)


class Discriminator(nn.Module):
	def __init__(self,data_shape, hidden_shape, is_wasserstein=False,time_conditioning=True,append_time=False,encode_time=False,leaky=False,use_vgg=False):

		super(Discriminator,self).__init__()
		self.flat = nn.Flatten()
		# self.layer_0   = nn.Linear(data_shape,hidden_shape)
		# self.relu      = nn.LeakyReLU()
		# self.out_layer = nn.Linear(hidden_shape,1)
		self.is_wasserstein = is_wasserstein
		self.time_conditioning = time_conditioning
		self.append_time = append_time
		self.encode_time = encode_time

		if use_vgg:
			in_channels = 16
		else:
			in_channels = 1
		if self.append_time:
			in_channels += 2
		else:
			in_channels += 0
		if self.encode_time:
			self.time_enc = TimeEncodings(784,2)

		self.flat = nn.Flatten()
		self.layer_0 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3)
		self.relu_0    = TimeReLU(1352*1,1,leaky)
		# self.relu_0 = nn.ReLU()
		if time_conditioning:
			self.layer_1 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3)
			self.layer_2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3)
			self.relu_1 = TimeReLU(200,1,leaky)
			self.relu_2 = TimeReLU(72,1,leaky)
			# self.relu_t = nn.ReLU()
		self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)
		self.out_layer = nn.Linear(72,1)
		# self.out_relu = TimeReLU(1,2,leaky)

	def forward(self,X,times):
		if len(X.size()) < 4:
			X = X.unsqueeze(1)

		if self.encode_time:
			X = self.time_enc(X,times)

		if self.append_time:
			times_ = times.unsqueeze(1).unsqueeze(2).repeat(1,1,28,28)
			X = torch.cat([X,times],dim=1)

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
	
	def forward(self,X,times):
		# times = X[:,-self.time_dim:]
		orig_shape = X.size()
		# print(orig_shape)
		X = X.view(orig_shape[0],-1)
		if len(times.size()) == 3:
			times = times.squeeze(2)
		thresholds = self.model(times)
		if self.leaky:
			alphas = self.model_alpha(times)
		else:
			alphas = 0.0
		# print("Thresh",thresholds.shape,X.size())
		X = torch.where(X>thresholds,X,alphas*X+thresholds)
		# print(X.size())
		X = X.view(*list(orig_shape))
		return X

class TimeEncodings(nn.Module):
	def __init__(self,model_dim,time_dim,proj_time=False):
		super(TimeEncodings,self).__init__()
		self.model_dim = model_dim
		self.time_dim = time_dim
		self.proj_time = proj_time
		if self.proj_time:
			self.ProjectTime = nn.Linear(time_dim+(time_dim//2),1) # This prepares and amalgamates times for the sin/cos stuff
	
	def forward(self,X,times):

		odd_indices = torch.arange(0,times.size(1)//2,1).long().to(device) + times.size(1)//2
		even_indices = torch.arange(1,times.size(1)//2,1).long().to(device) + times.size(1)//2
		# print(torch.gather(times,1,odd_indices.unsque{eze(0).repeat(times.size(0),1)).size()) #,(times.gather(1,odd_indices).unsqueeze(1)-times.gather(1,even_indices).unsqueeze(1)).size())
		times = torch.cat([times,torch.gather(times,1,odd_indices.unsqueeze(0).repeat(times.size(0),1))-\
							torch.gather(times,1,even_indices.unsqueeze(0).repeat(times.size(0),1))],dim=1)
		if self.proj_time:
			times = self.ProjectTime(times)
		orig_size = X.size()
		X_ = X.view(orig_size[0],-1)
		# freq_0 = torch.tensor([1/(100**(2*x/self.model_dim)) for x in range(self.model_dim)]).view(1,self.model_dim).repeat(n,1)
		for t in range(times.size(1)):
			freq = torch.tensor([1/((100/(t+1))**(2*x/self.model_dim)) for x in range(self.model_dim)]).view(1,self.model_dim).repeat(orig_size[0],1).to(device)
			offsets = torch.ones_like(X_).to(device) * (3.1415/2) * (t % 2)
			# print(offsets.size(),freq.size(), times[:,t].size())
			pos = torch.sin(freq * times[:,t].unsqueeze(1) + offsets)
			X = X_ + pos
		X = X_.view(orig_size)
		return X
# class TimeEmbeddings(nn.Module):
#     def __init__(self,model_dim,encoding):

class ConvBlock(nn.Module):
	def __init__(self,in_channels, out_channels,time_relu_size,time_shape=2,kernel_size=3,leaky_relu=True):
		super(ConvBlock,self).__init__()
		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,  padding=1)
		self.relu = TimeReLU(time_relu_size,time_shape=time_shape,leaky=leaky_relu)

	def forward(self,X,times):
		X = self.conv(X)
		X = self.relu(X,times)
		return X


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
	# print(x.size(),y.size())
	# if len(x.shape) == 3:
	#     x = nn.Flatten()(x)

	x_1 = x.view(x.size(0),-1)
	y_1 = y.view(y.size(0),-1)
	# print(torch.sum((x_1-y_1)**2,dim=1).size())
	# print(x_1.size(),y_1.size(),torch.sum((x_1-y_1)**2,dim=1).size())
	return torch.sum((x_1-y_1)**2,dim=1) #.sum(dim=1)


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

