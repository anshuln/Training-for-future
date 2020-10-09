# # Contains the model definition for the classifier
# # We are considering two approaches - 
# # 1. Transformer does heavy lifting and just gives a transformed distribution to the classifier
# # 2. Classifier does heavy lifting and tries to learn P(y|X,t) with special emphasis on time = T+1
## TODO - 1. Conv
##        2. TimeEncoding
##        3. U-NET type structure?
from torch import nn
import torch
device = "cuda:0"

class ClassifyNet(nn.Module):
	# Model adapted from https://github.com/hehaodele/CIDA/blob/master/rotatingMNIST/model.py#L176
	def __init__(self,data_shape, hidden_shape=256, n_classes=10, append_time=False,encode_time=True,time_conditioning=True,leaky=True):
		super(ClassifyNet,self).__init__()

		self.time_conditioning = time_conditioning
		self.append_time = append_time
		self.encode_time = encode_time
		nh = hidden_shape
		nz = 1

		if self.append_time:
			in_channels = 3
		else:
			in_channels = 1
		if self.encode_time:
			self.time_enc = TimeEncodings(784,2)

		self.flat = nn.Flatten()
		self.conv1 = ConvBlock(in_channels=in_channels, out_channels=nh,kernel_size=3,stride=2,  padding=1,time_relu_size=196*nh,time_shape=2,dropout=0.5,leaky_relu=True)
		self.conv2 = ConvBlock(in_channels=nh, out_channels=nh,kernel_size=3,stride=2,  padding=1,time_relu_size=49*nh,time_shape=2,dropout=0.5,leaky_relu=True)
		self.conv3 = ConvBlock(in_channels=nh, out_channels=nh,kernel_size=3,stride=2,  padding=1,time_relu_size=16*nh,time_shape=2,dropout=0.5,leaky_relu=True)
		self.conv4 = ConvBlock(in_channels=nh, out_channels=nz,kernel_size=4,stride=1,  padding=0,time_relu_size=1*nz,time_shape=2,dropout=0.5,leaky_relu=True)
		# self.conv = nn.Sequential(
	 #            nn.Conv2d(1, nh, 3, 2, 1), nn.BatchNorm2d(nh), TimeReLU(True), nn.Dropout(opt.dropout),  # 14 x 14
	 #            nn.Conv2d(nh, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True), nn.Dropout(opt.dropout),  # 7 x 7
	 #            nn.Conv2d(nh, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True), nn.Dropout(opt.dropout),  # 4 x 4
	 #            nn.Conv2d(nh, nz, 4, 1, 0), nn.ReLU(True),  # 1 x 1
	 #        )
	     # self.fc_pred = nn.Sequential(
        #     nn.Conv2d(nz, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.ReLU(True),
        #     nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.ReLU(True),
        #     nnSqueeze(),
        #     nn.Linear(nh, 10),
        # )

		self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)
		self.out_layer = nn.Linear(nz,n_classes)
		self.out_relu = TimeReLU(n_classes,2,leaky)

	def forward(self,X,times):
		X = X.unsqueeze(1)
		
		if self.encode_time:
			X = self.time_enc(X,times)

		if self.append_time:
			times_ = times.unsqueeze(1).unsqueeze(2).repeat(1,1,28,28)
			X = torch.cat([X,times],dim=1)

		X  = self.conv1(X,times)
		X  = self.conv2(X,times)
		X  = self.conv3(X,times)
		X  = self.conv4(X,times)

		X = self.out_layer(X.view(X.size(0),-1))
		X = self.out_relu(X,times)
		X = torch.softmax(X,dim=1)

		return X


class Transformer(nn.Module):
	def __init__(self,data_shape, latent_shape,append_time=False,encode_time=True, label_dim=0):
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

		if self.append_time:
			in_channels = 5
		else:
			in_channels = 1
		if self.encode_time:
			self.time_enc = TimeEncodings(784,2)
		self.flat = nn.Flatten()
		self.layer_0 = ConvBlock(in_channels=in_channels,out_channels=8,kernel_size=3,time_relu_size=6272,time_shape=4)
		self.layer_1 = ConvBlock(in_channels=8,out_channels=8,kernel_size=3,time_relu_size=3136//2,time_shape=4)
		self.layer_2 = ConvBlock(in_channels=8,out_channels=16,kernel_size=3,time_relu_size=784,time_shape=4)
		self.layer_3 = ConvBlock(in_channels=16,out_channels=8,kernel_size=3,time_relu_size=392,time_shape=4)
		self.layer_4 = ConvBlock(in_channels=8,out_channels=8,kernel_size=3,time_relu_size=1568,time_shape=4)
		self.layer_5 = ConvBlock(in_channels=8,out_channels=1,kernel_size=3,time_relu_size=784,time_shape=4)
		self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
		self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)
		# self.layer_0_2 = nn.Linear(latent_shape,latent_shape)

		self.label_dim = label_dim

	def forward(self,X,times):
		# times = X[:,-4:]
		# X = torch.cat([self.flat(X),times.view(-1,4)],dim=-1)
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
	def __init__(self,data_shape, hidden_shape=512, is_wasserstein=False,time_conditioning=True,append_time=False,encode_time=True,leaky=False):

		super(Discriminator,self).__init__()
		self.flat = nn.Flatten()
		# self.layer_0   = nn.Linear(data_shape,hidden_shape)
		# self.relu      = nn.LeakyReLU()
		# self.out_layer = nn.Linear(hidden_shape,1)
		self.is_wasserstein = is_wasserstein
		self.time_conditioning = time_conditioning
		self.append_time = append_time
		self.encode_time = encode_time
		nh = hidden_shape
		if self.append_time:
			in_channels = 3
		else:
			in_channels = 1
		if self.encode_time:
			self.time_enc = TimeEncodings(784,2)

		# self.conv1 = ConvBlock(in_channels=in_channels, out_channels=nh,kernel_size=2,stride=2,  padding=0,time_relu_size=26*26*nh,time_shape=2,dropout=0.0,leaky_relu=True)
		# self.conv2 = ConvBlock(in_channels=nh, out_channels=nh,kernel_size=1,stride=1,  padding=0,time_relu_size=24*24*nh,time_shape=2,dropout=0.0,leaky_relu=True)
		# self.conv3 = ConvBlock(in_channels=in_channels, out_channels=nh,kernel_size=1,stride=1,  padding=0,time_relu_size=22*22*nh,time_shape=2,dropout=0.0,leaky_relu=True)
        # self.net = nn.Sequential(
        #     nn.Conv2d(nin, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
        #     nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
        #     nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
        #     nnSqueeze(),
        #     nn.Linear(nh, nout),
        # )
		self.flat = nn.Flatten()
		self.layer_0 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3)
		self.relu_0    = TimeReLU(8*1352*in_channels,2,leaky)
		# self.relu_0 = nn.ReLU()
		if time_conditioning:
			self.layer_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
			self.layer_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
			self.relu_1 = TimeReLU(200*64,2,leaky)
			self.relu_2 = TimeReLU(72*64,2,leaky)
			# self.relu_t = nn.ReLU()
		self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)
		self.out_layer = nn.Linear(72*64,1)
		# self.out_relu = TimeReLU(1,2,leaky)

	def forward(self,X,times):
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
	def __init__(self,data_shape,time_shape,leaky=False,time2vec=True,time2vec_hidden_shape=8):
		super(TimeReLU,self).__init__()
		if time2vec:
			self.time_emb = Time2Vec(time_shape,time2vec_hidden_shape)
			time_shape = time2vec_hidden_shape
		self.model = nn.Linear(time_shape,data_shape)
		if leaky:
			self.model_alpha = nn.Linear(time_shape,data_shape)
		self.leaky = leaky
		self.time_dim = time_shape
		self.time2vec = time2vec
	
	def forward(self,X,times):
		# times = X[:,-self.time_dim:]
		orig_shape = X.size()
		# print(orig_shape)
		X = X.view(orig_shape[0],-1)
		if self.time2vec:
			times = self.time_emb(times)
		thresholds = self.model(times)
		if self.leaky:
			alphas = self.model_alpha(times)
		else:
			alphas = 0.0
		# print("Thresh",times.shape)
		X = torch.where(X>thresholds,X,alphas*X+thresholds)
		X = X.view(*list(orig_shape))
		return X


class Time2Vec(nn.Module):
	'''
	Time2Vec implementation from https://arxiv.org/pdf/1907.05321.pdf
	
	'''
	def __init__(self,time_shape,emb_shape):
		'''
		Arguments:
			time_shape {int} -- Input shape of time
			emb_shape {int} -- output shape of embedding
		'''
		super(Time2Vec,self).__init__()
		self.mat = nn.Linear(time_shape,emb_shape)
		self.activation = torch.sin 

	def forward(self,t):
		emb = self.mat(t)
		# print(emb.size())
		emb = torch.cat([emb[:,:1],self.activation(emb[:,1:])],dim=1)
		return emb

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
	def __init__(self,in_channels, out_channels,kernel_size,stride=1,  padding=1,time_relu_size=None,time_shape=2,dropout=0.0,leaky_relu=True):
		super(ConvBlock,self).__init__()
		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,stride=stride,  padding=padding)
		self.batch_norm = nn.BatchNorm2d(out_channels)
		self.relu = TimeReLU(time_relu_size,time_shape=time_shape,leaky=leaky_relu)
		self.dropout = nn.Dropout(dropout)
	def forward(self,X,times):
		X = self.conv(X)
		X = self.batch_norm(X)
		X = self.relu(X,times)
		X = self.dropout(X)
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
	return torch.sum((x-y)**2,dim=1).sum(dim=1)


def transformer_loss(trans_output,is_wasserstein=False):

	if is_wasserstein:
		return trans_output
	return bxe(torch.ones_like(trans_output), trans_output)

def discounted_transformer_loss(rec_target_data, trans_data,trans_output, pred_class, actual_class,is_wasserstein=False):

	# time_diff = torch.exp(-(real_data[:,-1] - real_data[:,-2]))
	#TODO put time_diff


	re_loss = reconstruction_loss(rec_target_data, trans_data)
	tr_loss = transformer_loss(trans_output,is_wasserstein)
	# transformed_class = trans_data[:,-1].view(-1,1)

	# print(actual_class,pred_class)
	class_loss = classification_loss(pred_class,actual_class)
	loss = torch.mean( 1.0* tr_loss.squeeze() +  0.0*re_loss + 0.5*class_loss)
	# loss = tr_loss.mean()
	return loss, tr_loss.mean(),re_loss.mean(), class_loss.mean()

