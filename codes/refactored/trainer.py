'''Abstraction

This is an abstract class for all trainers, it wraps up datasets etc. The train script will call this with the appropriate params
''' 

import torch 
import numpy as np
import json
from utils import *
from dataset import *
from regularized_ot import *
from tqdm import tqdm
from losses import *

def train_transformer_batch(X,source_A,source_U,dest_A,dest_U,Y,X_transported,transformer,discriminator,classifier,transformer_optimizer, is_wasserstein=False,encoder=None):
	# print(X.size())
	'''Trains the transformer on one batch
	
	
	Arguments:
		X {tensor} -- Batch of input X, B x (data_shape)
		Y {[type]} -- Batch of input labels, not one-hot encoded, B x 1
		X_transport {[type]} -- Batch of input X transported in time through OT, B x (data_shape)
		transformer {[type]} -- Transformer model
		discriminator {[type]} -- Discriminator model
		classifier {[type]} -- Classifier model
		transformer_optimizer {[type]} --   
	Keyword Arguments:
		is_wasserstein {bool} -- Whether to use wasserstein discriminator for training (default: {False})
		encoder {[type]} -- Encoder model to encode data (default: {None})
	'''

	if encoder is not None:
		with torch.no_grad():
			X = encoder(X)
			# X_now = encoder(X_now).view(-1,16,28,28)
	transformer_optimizer.zero_grad()
	X_pred = transformer(X,torch.cat([source_U,dest_U],dim=1))

	# is_real = discriminator(X_pred,dest_u)
	pred_disc = discriminator(X_pred, dest_U)

	# feature_pred = discriminator._feature(X_pred, dest_U)
	# feature_real = discriminator._feature(real_X, dest_U)
	# feature_loss = reconstruction_loss(feature_pred, feature_real)

	ot_loss = ot_transformer_loss(X_pred, source_U,dest_U,pred_disc, X_transported,is_wasserstein=is_wasserstein) # TODO - see if correct, why is X needed here?

	Y_pred = classifier(X_pred, dest_U)
	# trans_loss,ld,lr, lc = discounted_transformer_loss(transported_X, X_pred,is_real, pred_class,Y,is_wasserstein)

	class_loss = classification_loss(Y_pred, Y).mean()
	trans_loss = ot_loss + class_loss # + feature_loss

	trans_loss.backward()

	transformer_optimizer.step()

	return trans_loss, ot_loss, class_loss


def train_discriminator_batch(X_old, source_A, source_U, dest_A, dest_U, X_now, transformer, discriminator, discriminator_optimizer, encoder,is_wasserstein=False):
	'''Trains the discriminator on a batch
		
	Arguments:
		X_old {[type]} -- Previous data
		X_now {[type]} -- Current time step data
		transformer {[type]} -- Model
		discriminator {[type]} -- Model
		discriminator_optimizer {[type]} -- Optimizer
	
	Keyword Arguments:
		is_wasserstein {bool} -- Whether to use a WGAN style disc (default: {False})
		encoder {[type]} -- Model through which inputs to discriminator pass through (default: {None})
	'''
	if encoder is not None:
		with torch.no_grad():
			X_old = encoder(X_old)
			X_now = encoder(X_now)  
	discriminator_optimizer.zero_grad()
	X_pred = transformer(X_old,torch.cat([source_U,dest_U],dim=-1))

	# X_pred = transformer(X_old,cat_U)

	is_real_old = discriminator(X_pred,dest_U)
	is_real_now = discriminator(X_now, dest_U)
	
	if is_wasserstein:
		disc_loss = discriminator_loss_wasserstein(is_real_now, is_real_old)
	else:
		disc_loss = discriminator_loss(is_real_now, is_real_old)

	disc_loss.backward()

	discriminator_optimizer.step()
	
	if is_wasserstein:   # We use the simplest strategy for enforcing Lipschitzness
		for p in discriminator.parameters():
			p.data.clamp_(-0.2, 0.2)
	return disc_loss



def train_classifier_batch(X,dest_u,dest_a,Y,classifier,classifier_optimizer,batch_size,verbose=False,encoder=None, transformer=None,source_u=None, kernel=None):
	'''Trains classifier on a batch
	
	[description]
	
	Arguments:
		X {[type]} -- 
		Y {[type]} -- 
		classifier {[type]} -- 
		classifier_optimizer {[type]} -- 
	
	Keyword Arguments:
		transformer {[type]} -- Transformer model. If this is none we just train the classifier on the input data. (default: {None})
		encoder {[type]} --   (default: {None})
	
	Returns:
		[type] -- [description]
	'''
	classifier_optimizer.zero_grad()
	if encoder is not None:
		with torch.no_grad():
			X = encoder(X).view(out_shape)

	if transformer is not None:
		X_pred = transformer(X,torch.cat([source_u.to(device),dest_u.to(device)],dim=1))
	else:
		X_pred = X

	Y_pred = classifier(X_pred,dest_u)

	pred_loss = classification_loss(Y_pred, Y)
	if kernel is not None:
		pred_loss = pred_loss * kernel
	pred_loss = pred_loss.sum()/batch_size
	pred_loss.backward()
	classifier_optimizer.step()
	

	return pred_loss





class TransformerTrainer():
	def __init__(self,args):
		self.DataSet = RotMNIST 
		if args.data == "mnist":
			self.dataset_kwargs = {"data_path":"../../data/MNIST/processed/","device":args.device,"num_bins": 6}
			self.source_domain_indices = [1,2,3,4]
			data_index_file = "../../data/MNIST/processed/indices.json"
			if args.encoder:
				self.out_shape = (-1,16,28,28)
			else:
				self.out_shape = (-1,1,28,28)

			from model_MNIST_conv import ClassifyNet, Transformer, Discriminator
			self.classifier = ClassifyNet(28**2 + 2,256,10, use_vgg=args.encoder).to(args.device)
			self.classifier_optimizer = torch.optim.Adagrad(self.classifier.parameters(),5e-3)
			self.transformer = Transformer(28**2 + 2*2, 256, use_vgg=args.encoder).to(args.device)
			self.transformer_optimizer = torch.optim.Adagrad(self.transformer.parameters(),5e-2)
			self.discriminator = Discriminator(28**2 + 2, 256,args.wasserstein_disc, use_vgg=args.encoder).to(args.device)
			self.discriminator_optimizer = torch.optim.Adagrad(self.discriminator.parameters(),1e-3)

			self.U_source = np.array([1,2,3,4])/6
			self.A_source_mean = np.array([0,15,30,45])/6
		if args.data == "sleep":
			self.dataset_kwargs = {"root_dir":"",}
			self.source_domain_indices = [0,1,2,3]
			self.out_shape = (-1,) # TODO
			data_index_file = "Sleep/indices.npy"
			from model_sleep import  Transformer
		if args.data == "moons":
			self.dataset_kwargs = {"root_dir":"",}
			self.source_domain_indices = [0,1,2,3]
			self.out_shape = (-1,2) # TODO
			data_index_file = "Moons/indices.npy"
			from model_moons import  Transformer
			# Load models and optimizers here!

		self.source_data_indices = json.load(open(data_index_file,"r"))  # List of lists of length same as source_domain_indices, each element is a list of indices of that domain's elements in the overall concatenated dataset
		self.cumulative_data_indices = get_cumulative_data_indices(self.source_data_indices)
		assert len(self.source_domain_indices) == len(self.source_data_indices)
		self.device = args.device
		self.CLASSIFIER_EPOCHS = args.epoch_classifier 
		self.TRANSFORMER_EPOCH = args.epoch_transform
		self.SUBEPOCH = 3
		self.BATCH_SIZE = args.bs
		self.encoder = None
		self.ot_maps = [[None for x in range(len(self.source_data_indices))] for y in range(len(self.source_data_indices))]
				# TODO A_mean and U_source
		self.IS_WASSERSTEIN = args.wasserstein_disc


	def get_ot_maps(self): 
		ot_data = [torch.utils.data.DataLoader(self.DataSet(indices=self.source_data_indices[x],**self.dataset_kwargs),len(self.source_data_indices[x]),False) for x in self.source_domain_indices]
		for i in range(len(self.source_domain_indices)):
			for j in range(i,len(self.source_domain_indices)):
				if i!=j:
					ot_sinkhorn = RegularizedSinkhornTransportOTDA(reg_e=0.5, alpha=10, max_iter=50, norm="max", verbose=False)
					# Prepare data
					if self.encoder is not None:
						data_s = next(iter(ot_data[i]))
						data_t = next(iter(ot_data[j]))
						Xs = self.encoder(data_s[0]).view(len(self.source_domain_indices[i]),-1).detach().cpu().numpy()+1e-6
						ys = data_s[3].view(len(self.source_data_indices[i]),-1).detach().cpu().numpy()
						Xt = self.encoder(data_t[0]).view(len(self.source_data_indices[j]),-1).detach().cpu().numpy()+1e-6
						yt = data_t[3].view(len(self.source_data_indices[j]),-1).detach().cpu().numpy()
					else:
						data_s = next(iter(ot_data[i]))
						data_t = next(iter(ot_data[j]))

						Xs = data_s[0].view(len(self.source_data_indices[i]),-1).detach().cpu().numpy()+1e-6
						ys = data_s[3].view(len(self.source_data_indices[i]),-1).detach().cpu().numpy()
						Xt = data_t[0].view(len(self.source_data_indices[j]),-1).detach().cpu().numpy()+1e-6
						yt = data_t[3].view(len(self.source_data_indices[j]),-1).detach().cpu().numpy() 

					# Compute OT map
					ot_sinkhorn.fit(Xs=Xs, ys=ys, Xt=Xt, yt=yt, iteration=0)
					self.ot_maps[i][j] = ot_sinkhorn.transform(Xs).reshape(self.out_shape)
				else:
					if  self.encoder is not None:
						Xs = self.encoder(next(iter(ot_data[i]))[0]).detach().cpu().numpy()
					else:
						Xs = next(iter(ot_data[i]))[0].view(self.out_shape).detach().cpu().numpy()
					self.ot_maps[i][j] = Xs

	def train_classifier(self):
		class_step = 0
		past_data = self.DataSet(indices=self.cumulative_data_indices[-1],**self.dataset_kwargs)
		for epoch in range(self.CLASSIFIER_EPOCHS):
			past_dataset = torch.utils.data.DataLoader((past_data),self.BATCH_SIZE,True)
			class_loss = 0
			for batch_X, batch_A, batch_U, batch_Y in tqdm(past_dataset):

				# batch_X = torch.cat([batch_X,batch_U.view(-1,2)],dim=1)

				l = train_classifier_batch(X=batch_X,dest_u=batch_U,dest_a=batch_A,Y=batch_Y,classifier=self.classifier,classifier_optimizer=self.classifier_optimizer,verbose=False,encoder=self.encoder, batch_size=self.BATCH_SIZE)
				class_step += 1
				class_loss += l
			print("Epoch %d Loss %f"%(epoch,class_loss),flush=False)

	def train_transformer(self):

		for index in range(1, len(self.source_domain_indices)):

			print('Domain %d' %index)
			print('----------------------------------------------------------------------------------------------')
			past_dataset = torch.utils.data.DataLoader(self.DataSet(indices=self.cumulative_data_indices[index-1],**self.dataset_kwargs),self.BATCH_SIZE,True)  # Used to train discriminator, data from previous time steps

			curr_dataset = torch.utils.data.DataLoader(self.DataSet(indices=self.source_data_indices[index],**self.dataset_kwargs),self.BATCH_SIZE,True,drop_last=True)  # Used to train discriminator, data from current time step

			num_past_batches = len(self.cumulative_data_indices[index-1]) // self.BATCH_SIZE
			all_dataset = torch.utils.data.DataLoader(self.DataSet(indices=self.cumulative_data_indices[index],transported_samples=self.ot_maps,target_bin=index,**self.dataset_kwargs),self.BATCH_SIZE,True)   # Used to train transformer, data from previous + current time steps

			num_all_batches  = len(self.cumulative_data_indices[index-1]) // self.BATCH_SIZE
			all_steps_t = 0  # Book-keeping variables
			all_steps_d = 0  # Book-keeping variables

		
			all_dataset_iterator  = iter(all_dataset)
			past_dataset_iterator = iter(past_dataset)
			curr_dataset_iterator = iter(curr_dataset) 
			for epoch in range(self.TRANSFORMER_EPOCH):
				loss_trans, loss_disc = 0,0
				
				loss1, loss2 = 0,0
				step_t,step_d = 0,0

				loop1 = True
				loop2 = True
				
				
				try:
					for j in range(self.SUBEPOCH):
						# Discriminator training loop
						batch_X, batch_A, batch_U, batch_Y = next(past_dataset_iterator)
						batch_U = batch_U.view(-1,1)
						batch_A = batch_A.view(-1,1)
						this_U = torch.tensor([self.U_source[index]]*batch_U.shape[0])
						this_A = torch.tensor([self.A_source_mean[index]]*batch_A.shape[0])
						this_U = this_U.view(-1,1).float().to(self.device)
						this_A = this_A.view(-1,1).float().to(self.device)
						# cat_U = torch.cat([batch_U, this_U], dim=1)
						# cat_A = torch.cat([batch_A, this_A], dim=1)
						# batch_X = torch.cat([batch_X, batch_U, this_U], dim=1)
						# Do this in a better way
						try:
							real_X,real_U,real_A,_ = next(curr_dataset_iterator)
						except StopIteration:
							curr_dataset_iterator = iter(curr_dataset)
							real_X,real_U,real_A,_ = next(curr_dataset_iterator)
						loss_d = train_discriminator_batch(X_old=batch_X, source_A=batch_A, source_U=batch_U,dest_A=this_A, dest_U=this_U, X_now=real_X, transformer=self.transformer, discriminator=self.discriminator, discriminator_optimizer=self.discriminator_optimizer, encoder=self.encoder, is_wasserstein=self.IS_WASSERSTEIN)
						loss_disc += loss_d


					
					for j in range(self.SUBEPOCH):
				
						batch_X, batch_W, batch_A, batch_U, batch_Y = next(all_dataset_iterator)
					
						batch_U = batch_U.view(-1,1)
						batch_A = batch_A.view(-1,1)
						this_U = torch.tensor([self.U_source[index]]*batch_U.shape[0])
						this_A = torch.tensor([self.A_source_mean[index]]*batch_A.shape[0])

						this_U = this_U.view(-1,1).float().to(self.device)
						this_A = this_A.view(-1,1).float().to(self.device)

						loss_t, loss_ot, loss_c = train_transformer_batch(X=batch_X,source_A=batch_A,source_U=batch_U,dest_A=this_A,dest_U=this_U,Y=batch_Y,X_transported=batch_W,transformer=self.transformer,discriminator=self.discriminator,classifier=self.classifier,transformer_optimizer=self.transformer_optimizer, is_wasserstein=self.IS_WASSERSTEIN,encoder=self.encoder)
						loss_trans += loss_t
						# writer.add_scalar('Loss/transformer',loss_t.detach().numpy(),epoch)

					print('Epoch %d - %9.9f %9.9f' % (epoch, loss_disc.detach().cpu().numpy(), loss_trans.detach().cpu().numpy()))

				except StopIteration:
					all_dataset_iterator = iter(all_dataset)
					past_dataset_iterator = iter(past_dataset)


	def train_final_classifier(self):

			source_dataset = torch.utils.data.DataLoader(self.DataSet(indices=self.cumulative_data_indices[-1],**self.dataset_kwargs),self.BATCH_SIZE,True)



			step = 0
			for epoch in range(self.CLASSIFIER_EPOCHS//2):

                loss = 0

                for batch_X, batch_A, batch_U, batch_Y in source_dataset:
                    batch_U = batch_U.view(-1,2)
                    this_U = np.array([U_target[i]*BIN_WIDTH]*batch_U.shape[0]).reshape((batch_U.shape[0],1)) +\
                    np.random.randint(0,5,size=(batch_U.shape[0],1))
                    this_U = np.hstack([np.array([U_target[i]]*batch_U.shape[0]).reshape((batch_U.shape[0],1)),
                        this_U/(BIN_WIDTH * 6)])
                    this_U = torch.tensor(this_U).float().view(-1,2).to(device)
                      # batch_X = torch.cat([batch_X, batch_U, this_U], dim=1)
                    step += 1
                    loss += train_classifier(batch_X,batch_U,this_U, batch_Y, self.classifier,self.transformer, self.classifier_optimizer,encoder=self.encoder)

                print('Epoch: %d - ClassificationLoss: %f' % (epoch, loss))




	def train(self):
		self.get_ot_maps()
		self.train_classifier()
		self.train_transformer()
		self.train_final_classifier()
		# self.evaluate_classifier()

class CrossGradTrainer():
	def __init__(self,args):
		self.DataSet = GradDataset 
		if args.data == "mnist":
			self.dataset_kwargs = {"root_dir":"",}
			self.source_domain_indices = [0,1,2,3]
			data_index_file = "MNIST/indices.txt"
			self.out_shape = (-1,16,28,28)
			from model_MNIST import GradModel
		if args.data == "sleep":
			self.dataset_kwargs = {"root_dir":"",}
			self.source_domain_indices = [0,1,2,3]
			self.out_shape = (-1,) # TODO
			data_index_file = "Sleep/indices.txt"
			from model_sleep import GradModel
		if args.data == "moons":
			self.dataset_kwargs = {"root_dir":"",}
			self.source_domain_indices = [0,1,2,3]
			self.out_shape = (-1,2) # TODO
			data_index_file = "Moons/indices.txt"
			from model_moons import GradModel
			# Load models and optimizers here!

		self.source_data_indices = np.load(data_index_file)
		self.cumulative_data_indices = get_cumulative_data_indices(self.source_data_indices)


		self.data_shape = (-1,16,28,28)
	def train_cross_grad(self):
		for idx in range(1,len(self.source_domain_indices)):
			source_indices = self.cumulative_data_indices[idx-1]
			grad_target_indices =  self.source_data_indices[idx]
			data_set = self.DataSet(source_indices=source_indices,target_indices=target_indices,**self.dataset_kwargs) #RotMNISTCGrad(source_indices,grad_target_indices,BIN_WIDTH,src_indices[0]-1,6,src_indices[idx]-1)
			for epoch in range(self.EPOCH):
				data = torch.utils.data.DataLoader(data_set,self.BATCH_SIZE,self.shuffle)
				for img_1,img_2,time_diff in data:
					self.optimizer_gn.zero_grad()
					if self.encoder is not None:
						i1 = self.encoder(img_1).view(self.data_shape)
						i2 = self.encoder(img_2).view(self.data_shape)

					time_diff_pred = self.model_gn(i1,i2)
					loss = ((time_diff.view(-1,1) - time_diff_pred.view(-1,1))**2).sum()
					loss.backward()
					self.optimizer.step()
					print('Epoch %d - %f' % (epoch, loss.detach().cpu().numpy()),flush=True,end='\r')       

	def construct_final_dataset(self):
		self.final_dataset = []
		past_data = torch.utils.data.DataLoader(self.DataSetClassifier(indices=self.cumulative_data_indices[-1],**self.dataset_classification_kwargs),self.BATCH_SIZE,shuffle=False,drop_last=True) 
		time = self.target_u
		new_images = []
		new_labels = []
		for img,u,label in past_data:
			# new_img = torch.zeros_like(img).normal_(0.,1.)
			new_img = img.clone().detach()
			new_img.requires_grad = True
			optim = torch.optim.SGD([new_img],lr=1e-3)
			if self.encoder is not None:
				i1 = self.encoder(img_1).view(self.data_shape)
				i2 = self.encoder(img_2).view(self.data_shape)
			for s in range(self.cg_steps):
				loss = ((self.model_gn(i1,i2) - (u[:,1]-time).view(-1,1))**2).sum()
				grad = torch.autograd.grad(loss,i2)
				print('Step %d - %f' % (s, loss.detach().cpu().numpy()),flush=True,end='\r')
				with torch.no_grad():
					i2 = i2 - 1e-1*grad[0].data
					i2 = i2.detach().clone()
					i2.requires_grad = True
			new_images.append(new_img.detach().cpu().numpy())
			new_labels.append(label.view(-1,1).detach().cpu().numpy())
		new_ds_x, new_ds_y = np.vstack(new_images), np.vstack(new_labels)
		new_ds_u = np.hstack([np.array([time]*len(new_ds_x)).reshape(-1,1),np.array([5/6]*len(new_ds_x)).reshape(-1,1)])
		self.classification_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(new_ds_x).float().to(device),torch.tensor(new_ds_u).float().to(device),
			 torch.tensor(new_ds_y).long().to(device)),self.CLASSIFICATION_BATCH_SIZE,self.shuffle)
	def train(self):
		self.train_classifier()  # Train classifier initially
		self.train_cross_grad()  # Train model for cross-grad
		self.construct_final_dataset()  # Perturb source dataset for finetuning
		self.train_classifier(self.classification_dataset)  

'''Game plan

We need to have all dataloaders give out tensors, take away numpy array stuff from train loop.
Break down train loop into multiple functions as well.
We give only time, X to all models. This will make stuff compatible. We then change the models to do appropriate things.
'''