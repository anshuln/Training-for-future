'''Abstraction

This is an abstract class for all trainers, it wraps up datasets etc. The train script will call this with the appropriate params
''' 

import torch 
import numpy as np
import json
import pickle
from models import *
from utils import *
from dataset import *
from regularized_ot import *
from tqdm import tqdm
from losses import *
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


np.set_printoptions(precision = 3)
def train_transformer_batch(X,source_A,source_U,dest_A,dest_U,Y,X_transported,transformer,discriminator,classifier,transformer_optimizer, is_wasserstein=False,encoder=None):
	
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



def train_classifier_batch(X,dest_u,dest_a,Y,classifier,classifier_optimizer,batch_size,verbose=False,encoder=None, transformer=None,source_u=None, kernel=None,loss_fn=classification_loss):
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
			X = encoder(X) #.view(out_shape)

	if transformer is not None:
		# print(source_u.size(),dest_u.size())
		X_pred = transformer(X,torch.cat([source_u.squeeze(-1),dest_u],dim=1))
	else:
		X_pred = X

	Y_pred = classifier(X_pred,dest_a)
	pred_loss = loss_fn(Y_pred, Y)

	if verbose:
		with torch.no_grad():
			print(torch.cat([Y_pred+1e-15,Y.view(-1,1).float(),pred_loss.view(-1,1)],dim=1).detach().cpu().numpy())

	if kernel is not None:
		pred_loss = pred_loss * kernel
	pred_loss = pred_loss.sum()/batch_size
	pred_loss.backward()
	classifier_optimizer.step()
	


	return pred_loss





class TransformerTrainer():
	def __init__(self,args):
		self.DataSet = ClassificationDataSet 
		if args.data == "mnist":
			self.dataset_kwargs = {"root_dir":"../../data/MNIST/processed/","device":args.device, "num_bins": 6}
			self.source_domain_indices = [0,1,2,3]
			self.target_a = 5/6
			self.target_u = 5/6
			self.target_domain_indices = [5]
			data_index_file = "../../data/MNIST/processed/indices.json"
			self.out_shape = (-1,16,28,28)

			from model_MNIST_conv import Transformer, Discriminator
			from models import ClassifyNetCNN
			self.classifier = ClassifyNetCNN(28**2 + 2,256,10, use_vgg=args.encoder).to(args.device)
			self.classifier_optimizer = torch.optim.Adagrad(self.classifier.parameters(),5e-3)
			self.transformer = Transformer(28**2 + 2*2, 256, use_vgg=args.encoder).to(args.device)
			self.transformer_optimizer = torch.optim.Adagrad(self.transformer.parameters(),5e-2)
			self.discriminator = Discriminator(28**2 + 2, 256,args.wasserstein_disc, use_vgg=args.encoder).to(args.device)
			self.discriminator_optimizer = torch.optim.Adagrad(self.discriminator.parameters(),1e-3)

			self.U_source = np.array([1,2,3,4])/6
			self.A_source_mean = np.array([0,15,30,45])/6

			self.classifier_loss_fn = classification_loss
			self.task = 'classification'
			if args.encoder:
				self.encoder = EncoderCNN().to(args.device)
			else:
				self.encoder = None

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


		if args.data == "house":
			self.dataset_kwargs = {"data_path":"../../data/HousePrice","device":args.device, "drop_cols":30, "rand_target":False, "append_label":True, "label_dict_func": lambda x:int(x)//10, 'transport_idx_func': lambda x: x%200, "num_bins": 13, "base_bin": 6}
			self.source_domain_indices = [6,7,8,9,10]
			self.target_a = 1.0
			self.target_u = 11/13
			self.U_source = [6/13,7/13,8/13,9/13,10/13]
			self.A_source_mean = [6/13,7/13,8/13,9/13,10/13]
			self.target_domain_indices = [11]
			data_index_file = "../../data/HousePrice/indices.json"
			from models import ClassifyNet, Transformer#, Discriminator
			self.classifier = ClassifyNet(32,[10,2],1, use_vgg=args.encoder,time_conditioning=True,use_time2vec=True).to(args.device)
			self.classifier_optimizer = torch.optim.Adagrad(self.classifier.parameters(),5e-1)

			self.transformer = Transformer(32,[10,2],32, use_vgg=args.encoder,time_conditioning=True,use_time2vec=True).to(args.device)
			self.transformer_optimizer = torch.optim.Adagrad(self.transformer.parameters(),5e-2)
			# self.discriminator = Discriminator(28**2 + 2, 256,args.wasserstein_disc, use_vgg=args.encoder).to(args.device)
			# self.discriminator_optimizer = torch.optim.Adagrad(self.discriminator.parameters(),1e-3)

			self.classifier_loss_fn = reconstruction_loss
			self.task = 'regression'
			if args.encoder:
				self.encoder = EncoderCNN().to(args.device)
			else:
				self.encoder = None         

		if args.data == "house_classifier":
			self.dataset_kwargs = {"data_path":"../../data/HousePriceClassification","device":args.device, "drop_cols":30, "rand_target":False, "append_label":True, "label_dict_func": lambda x:int(x)//10, 'transport_idx_func': lambda x: x%200, "num_bins": 13, "base_bin": 6}
			self.source_domain_indices = [6,7,8,9,10]
			self.target_a = 1.0
			self.target_u = 11/13
			self.U_source = [6/13,7/13,8/13,9/13,10/13]
			self.A_source_mean = [6/13,7/13,8/13,9/13,10/13]
			self.target_domain_indices = [11]
			data_index_file = "../../data/HousePrice/indices.json"
			from models import ClassifyNet, Transformer#, Discriminator
			# from models import GradNet, ClassifyNet, Encoder
			self.classifier = ClassifyNet(32,[16,16,8],5, use_vgg=args.encoder,time_conditioning=True,use_time2vec=True).to(args.device)
			self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(),lr=5e-2)
			self.transformer = Transformer(32,[10,2],32, use_vgg=args.encoder,time_conditioning=True,use_time2vec=True).to(args.device)
			self.transformer_optimizer = torch.optim.Adagrad(self.transformer.parameters(),5e-2)
			self.classifier_loss_fn = classification_loss
			self.task = 'classification'
			if args.encoder:
				self.encoder = EncoderCNN().to(args.device)
			else:
				self.encoder = None

		data_indices = json.load(open(data_index_file,"r")) #, allow_pickle=True)
		# self.source_data_indices = np.load(data_index_file, allow_pickle=True)
		self.source_data_indices = [data_indices[i] for i in self.source_domain_indices]
		self.cumulative_data_indices = get_cumulative_data_indices(self.source_data_indices)
		self.target_indices = [data_indices[i] for i in self.target_domain_indices][0]  # TODO Flatten this list instead of picking 0th ele
		# self.target_indices = self.cumulative_data_indices[-1]
		# print(self.cumulative_data_indices)
		self.shuffle = True
		assert len(self.source_domain_indices) == len(self.source_data_indices)
		self.device = args.device
		self.CLASSIFIER_EPOCHS = args.epoch_classifier 
		self.TRANSFORMER_EPOCH = args.epoch_transform
		self.SUBEPOCH = 3
		self.BATCH_SIZE = args.bs
		self.ot_maps = [[None for x in range(len(self.source_data_indices))] for y in range(len(self.source_data_indices))]
				# TODO A_mean and U_source
		self.IS_WASSERSTEIN = args.wasserstein_disc


	def get_ot_maps(self): 
		ot_data = [torch.utils.data.DataLoader(self.DataSet(indices=self.source_data_indices[x],**self.dataset_kwargs),len(self.source_data_indices[x]),False) for x in range(len(self.source_domain_indices))]
		for i in range(len(self.source_domain_indices)):
			for j in range(i,len(self.source_domain_indices)):
				if i!=j:
					ot_sinkhorn = RegularizedSinkhornTransportOTDA(reg_e=0.5, alpha=10, max_iter=50, norm="max", verbose=False)
					# Prepare data
					if self.encoder is not None:
						data_s = next(iter(ot_data[i]))
						data_t = next(iter(ot_data[j]))
						Xs = self.encoder(data_s[0]).view(len(self.source_data_indices[i]),-1).detach().cpu().numpy()+1e-6
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
						Xs = next(iter(ot_data[i]))[0].detach().cpu().numpy()
					self.ot_maps[i][j] = Xs
		pickle.dump(self.ot_maps,open("OT_Maps.pkl","wb"))

	def train_classifier(self):
		class_step = 0
		past_data = self.DataSet(indices=self.cumulative_data_indices[len(self.source_domain_indices)-1],**self.dataset_kwargs)
		for epoch in range(self.CLASSIFIER_EPOCHS):
			past_dataset = torch.utils.data.DataLoader((past_data),self.BATCH_SIZE,True)
			class_loss = 0
			for batch_X, batch_A, batch_U, batch_Y in tqdm(past_dataset):

				# batch_X = torch.cat([batch_X,batch_U.view(-1,2)],dim=1)

				l = train_classifier_batch(X=batch_X,dest_u=batch_U,dest_a=batch_A,Y=batch_Y,classifier=self.classifier,classifier_optimizer=self.classifier_optimizer,verbose=False,encoder=self.encoder, batch_size=self.BATCH_SIZE,loss_fn=self.classifier_loss_fn)
				class_step += 1
				class_loss += l
			print("Epoch %d Loss %f"%(epoch,class_loss/len(past_data)),flush=False)
		torch.save(self.classifier.state_dict(), "classifier.pth")


	def train_transformer(self):
		for _ in range(3):
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
				for epoch in range(self.TRANSFORMER_EPOCH//3):
					loss_trans, loss_disc = 0,0
					
					loss1, loss2 = 0,0
					step_t,step_d = 0,0

					loop1 = True
					loop2 = True
					
					
					try:
						# for j in range(self.SUBEPOCH):
						#   # Discriminator training loop
						#   batch_X, batch_A, batch_U, batch_Y = next(past_dataset_iterator)
						#   batch_U = batch_U.view(-1,1)
						#   batch_A = batch_A.view(-1,1)
						#   this_U = torch.tensor([self.U_source[index]]*batch_U.shape[0])
						#   this_A = torch.tensor([self.A_source_mean[index]]*batch_A.shape[0])
						#   this_U = this_U.view(-1,1).float().to(self.device)
						#   this_A = this_A.view(-1,1).float().to(self.device)
						#   # cat_U = torch.cat([batch_U, this_U], dim=1)
						#   # cat_A = torch.cat([batch_A, this_A], dim=1)
						#   # batch_X = torch.cat([batch_X, batch_U, this_U], dim=1)
						#   # Do this in a better way
						#   try:
						#       real_X,real_U,real_A,_ = next(curr_dataset_iterator)
						#   except StopIteration:
						#       curr_dataset_iterator = iter(curr_dataset)
						#       real_X,real_U,real_A,_ = next(curr_dataset_iterator)
						#   loss_d = train_discriminator_batch(X_old=batch_X, source_A=batch_A, source_U=batch_U,dest_A=this_A, dest_U=this_U, X_now=real_X, transformer=self.transformer, discriminator=self.discriminator, discriminator_optimizer=self.discriminator_optimizer, encoder=self.encoder, is_wasserstein=self.IS_WASSERSTEIN)
						#   loss_disc += loss_d


						
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

						print('Epoch %d - %9.9f %9.9f' % (epoch, loss_disc, loss_trans.detach().cpu().numpy()))

					except StopIteration:
						all_dataset_iterator = iter(all_dataset)
						past_dataset_iterator = iter(past_dataset)


	def train_final_classifier(self):

			source_dataset = torch.utils.data.DataLoader(self.DataSet(indices=self.cumulative_data_indices[-1],**self.dataset_kwargs),self.BATCH_SIZE,True)



			step = 0
			for epoch in range(self.CLASSIFIER_EPOCHS//4):

				loss = 0

				for batch_X, batch_A, batch_U, batch_Y in source_dataset:
					# batch_U = batch_U.view(-1,2)
					# this_U = np.array([self.target_u*self.BIN_WIDTH]*batch_U.shape[0]).reshape((batch_U.shape[0],1)) #+\
					# np.random.randint(0,5,size=(batch_U.shape[0],1))
					this_U = np.array([self.target_u]*batch_U.shape[0]).reshape((batch_U.shape[0],1))
					this_U = torch.tensor(this_U).float().view(-1,1).to(self.device)
					  # batch_X = torch.cat([batch_X, batch_U, this_U], dim=1)
					step += 1
					loss += train_classifier_batch(X=batch_X,source_u=batch_U,dest_u=this_U,dest_a=this_U, Y=batch_Y, classifier=self.classifier,transformer=self.transformer, classifier_optimizer=self.classifier_optimizer,encoder=self.encoder,loss_fn=self.classifier_loss_fn,batch_size=self.BATCH_SIZE)

				print('Epoch: %d - ClassificationLoss: %f' % (epoch, loss))


	def eval_classifier(self):
		# TODO change for handling regression
		# if self.data == "house":
		#   self.dataset_kwargs["drop_cols_classifier"] = None
		td = ClassificationDataSet(indices=self.target_indices,**self.dataset_kwargs)
		target_dataset = torch.utils.data.DataLoader(td,self.BATCH_SIZE,self.shuffle,drop_last=False)
		Y_pred = []
		Y_label = []
		for batch_X, batch_A,batch_U, batch_Y in tqdm(target_dataset):
			batch_U = batch_U.view(-1,1)
			if self.encoder is not None:
				batch_X = self.encoder(batch_X)
			batch_Y_pred = self.classifier(batch_X, batch_A).detach().cpu().numpy()
			# print(batch_Y_pred.shape)
			if self.task == 'classification':
				Y_pred = Y_pred + [np.argmax(batch_Y_pred,axis=1)]
				Y_label = Y_label + [batch_Y.detach().cpu().numpy()]
			elif self.task == 'regression':
				Y_pred = Y_pred + [batch_Y_pred.reshape(-1,1)]
				Y_label = Y_label + [batch_Y.detach().cpu().numpy().reshape(-1,1)]
		print(len(Y_pred),len(Y_label))
		# print(Y_pred[0].shape,Y_label[0].shape)
		if self.task == 'classification':
			Y_pred = np.hstack(Y_pred)
			Y_label = np.hstack(Y_label)
			print('shape: ',Y_pred.shape)
			print(accuracy_score(Y_label, Y_pred))
			print(confusion_matrix(Y_label, Y_pred))
			print(classification_report(Y_label, Y_pred))    
		else:
			Y_pred = np.vstack(Y_pred)
			Y_label = np.vstack(Y_label)
			print('MAE: ',np.mean(np.abs(Y_label-Y_pred),axis=0))
			print('MSE: ',np.mean((Y_label-Y_pred)**2,axis=0))




	def train(self):
		self.get_ot_maps()
		# self.ot_maps = pickle.load(open("OT_Maps.pkl","rb"))
		self.train_classifier()
		self.eval_classifier()
		# self.classifier.load_state_dict(torch.load("classifier.pth"))
		self.train_transformer()
		# self.CLASSIFIER_EPOCHS = self.CLASSIFIER_EPOCHS//2
		self.train_final_classifier()
		self.eval_classifier()

class CrossGradTrainer():

	def __init__(self,args):

		self.DataSet = GradDataset
		self.DataSetClassifier = ClassificationDataSet
		self.CLASSIFIER_EPOCHS = args.epoch_classifier
		self.SUBEPOCHS = 5
		self.EPOCH = args.epoch_transform // self.SUBEPOCHS
		self.BATCH_SIZE = args.bs
		self.CLASSIFICATION_BATCH_SIZE = 100
		self.data = args.data 

		if args.data == "mnist":
			self.dataset_kwargs = {"root_dir":"../../data/MNIST/processed/","device":args.device, 'return_binary':False}
			self.source_domain_indices = [0,1,2,3]
			self.target_a = 5/6
			self.target_u = 5/6
			self.target_domain_indices = [5]
			data_index_file = "../../data/MNIST/processed/indices.json"
			self.out_shape = (-1,16,28,28)
			from models import GradNetCNN, ClassifyNetCNN, EncoderCNN
			self.classifier = ClassifyNetCNN(28**2 + 2,256,10, use_vgg=args.encoder).to(args.device)
			self.classifier_optimizer = torch.optim.Adagrad(self.classifier.parameters(),5e-3)
			self.model_gn = GradNetCNN(28**2 + 2*2, 256, use_vgg=args.encoder).to(args.device)
			self.optimizer_gn = torch.optim.Adagrad(self.model_gn.parameters(),5e-3)
			self.classifier_loss_fn = classification_loss
			self.task = 'classification'
			self.ord_class_loss_fn  = lambda x,y: torch.abs(x-y)
			if args.encoder:
				self.encoder = EncoderCNN().to(args.device)
			else:
				self.encoder = None


		if args.data == "house":
			self.dataset_kwargs = {"root_dir":"../../data/HousePrice","device":args.device, "drop_cols":30, "rand_target":False, "append_label":True, "label_dict_func": lambda x:int(x)//10, 'return_binary':True}
			self.source_domain_indices = [6,7,8,9,10]
			self.target_a = 1.0
			self.target_u = 11/12
			self.target_domain_indices = [11]
			data_index_file = "../../data/HousePrice/indices.json"
			from models import GradNet, ClassifyNet, Encoder
			self.classifier = ClassifyNet(32,[10,2],1, use_vgg=args.encoder,time_conditioning=False,use_time2vec=False,task='regression').to(args.device)
			self.classifier_optimizer = torch.optim.Adagrad(self.classifier.parameters(),5e-1)
			loss_type = 'bce' if self.dataset_kwargs['return_binary'] else 'reg'
			self.model_gn = GradNet(31,[100,100], use_vgg=args.encoder,loss_type=loss_type).to(args.device)
			self.optimizer_gn = torch.optim.Adam(self.model_gn.parameters(),5e-2)
			self.classifier_loss_fn = reconstruction_loss
			self.ord_class_loss_fn  = bxe if self.dataset_kwargs['return_binary'] else lambda x,y: torch.abs(x-y)
			self.task = 'regression'
			if args.encoder:
				self.encoder = EncoderCNN().to(args.device)
			else:
				self.encoder = None

		if args.data == "house_classifier":
			self.dataset_kwargs = {"root_dir":"../../data/HousePriceClassification","device":args.device, "drop_cols":30, "rand_target":False, "append_label":True, "label_dict_func": lambda x:int(x)//10}
			self.source_domain_indices = [6,7,8,9,10]
			self.target_a = 1.0
			self.target_u = 11/12
			self.target_domain_indices = [11]
			data_index_file = "../../data/HousePriceClassification/indices.json"
			from models import GradNet, ClassifyNet, Encoder
			self.classifier = ClassifyNet(32,[16,16,8],5, use_vgg=args.encoder,time_conditioning=True,use_time2vec=True).to(args.device)
			self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(),lr=5e-2)
			self.model_gn = GradNet(31,[16,16], use_vgg=args.encoder).to(args.device)
			self.optimizer_gn = torch.optim.Adam(self.model_gn.parameters(),5e-2)
			self.classifier_loss_fn = classification_loss
			self.task = 'classification'
			if args.encoder:
				self.encoder = EncoderCNN().to(args.device)
			else:
				self.encoder = None

		if args.data == "sleep":
			self.dataset_kwargs = {"root_dir":"",}
			self.source_domain_indices = [0,1,2,3]
			self.out_shape = (-1,2) # TODO
			data_index_file = "Sleep/indices.txt"
			#from model_sleep import GradModel

		if args.data == "moons":
			self.dataset_kwargs = {"root_dir":"",}
			self.source_domain_indices = [0,1,2,3]
			self.out_shape = (-1,2) # TODO
			data_index_file = "../../data/Moons/processed/indices.npy"
			self.data_path = '../../data/Moons/processed'
			#from model_moons import GradModel
			# Load models and optimizers here!

		if args.data == "cars":
			self.dataset_kwargs = {"root_dir":"../../data/CompCars/","device":args.device}
			self.source_domain_indices = np.arange(29) #[0,1,2,3]
			# self.target_u = 4/6
			data_index_file = "../../data/CompCars/indices_list.json"
			self.out_shape = (-1,16,28,28)
			from model_MNIST_conv import GradNet, ClassifyNetCars, EncoderCars
			self.classifier = ClassifyNetCars(28**2 + 2,256,10, use_vgg=args.encoder).to(args.device)
			self.classifier_optimizer = torch.optim.Adagrad(self.classifier.parameters(),5e-3)
			self.model_gn = GradNet(28**2 + 2*2, 256, use_vgg=args.encoder).to(args.device)
			self.optimizer_gn = torch.optim.Adagrad(self.model_gn.parameters(),5e-2)
			if args.encoder:
				self.encoder = EncoderCars().to(args.device)
			else:
				self.encoder = None

		data_indices = json.load(open(data_index_file,"r")) #, allow_pickle=True)
		# self.source_data_indices = np.load(data_index_file, allow_pickle=True)
		self.source_data_indices = [data_indices[i] for i in self.source_domain_indices]
		self.cumulative_data_indices = get_cumulative_data_indices(self.source_data_indices)
		self.target_indices = [data_indices[i] for i in self.target_domain_indices][0]  # TODO Flatten this list instead of picking 0th ele
		# self.target_indices = self.cumulative_data_indices[-1]
		# print(self.cumulative_data_indices)
		self.shuffle = True
		# self.data_shape = (-1,16,28,28)
		self.cg_steps = args.aug_steps
		self.device = args.device
		

	def _init_mnist(self):

		self.dataset_kwargs = {"root_dir":"",}
		self.source_domain_indices = [0,1,2,3]
		data_index_file = "MNIST/indices.txt"
		self.out_shape = (-1,16,28,28)

		# Initialize models here

	def get_curric_index_pairs(self,idx):
		'''Returns pairs for training ordinal classifier such that first you train on easier examples and increase complexity 
		
		source is a list of indices
		tgt is a list of lists
		mapping is a dict of each `idx` according to dataloader to one of the lists of tgt
		
		Arguments:
			idx {[type]} -- [description]
		
		Returns:
			[type] -- [description]
		'''
		source = []
		tgt = []
		map_curric = []
		counter  = 0
		total_ex = 0

		total_len = len(self.source_domain_indices)
		for i in range(1,idx+1):
			# This outer loop is so that we do not have catastrophic forgetting
			for j in range(i):
				source += self.source_data_indices[j]
				tgt.append(self.source_data_indices[j + (total_len - i)])
				map_curric += ([(x+total_ex,counter) for x in range(len(self.source_data_indices[j]))])
				counter += 1
				total_ex += len(self.source_data_indices[j])

		return source, tgt, dict(map_curric)


	def train_classifier(self,past_dataset=None,encoder=None):
		
		class_step = 0
		# for i in range(len(self.source_domain_indices)):
		if past_dataset is None:
			past_data = ClassificationDataSet(indices=self.source_data_indices[-1],**self.dataset_kwargs)
			print(len(past_data))
			past_dataset = torch.utils.data.DataLoader((past_data),self.BATCH_SIZE,True)
		for epoch in range(self.CLASSIFIER_EPOCHS):
			
			class_loss = 0
			for batch_X, batch_A, batch_U, batch_Y in tqdm(past_dataset):

				# batch_X = torch.cat([batch_X,batch_U.view(-1,2)],dim=1)

				l = train_classifier_batch(X=batch_X,dest_u=batch_U,dest_a=batch_A,Y=batch_Y,classifier=self.classifier,classifier_optimizer=self.classifier_optimizer,verbose=False,encoder=encoder, batch_size=self.BATCH_SIZE,loss_fn=self.classifier_loss_fn)
				class_step += 1
				class_loss += l
			print("Epoch %d Loss %f"%(epoch,class_loss),flush=False)

			# past_dataset = None



	def train_cross_grad(self):
		log = open("cross-grad-log.txt","w")

		for sub_ep in range(self.SUBEPOCHS):
			for idx in range(1, len(self.source_domain_indices)):

				source_indices, grad_target_indices, map_index_curric = self.get_curric_index_pairs(idx)
				# print(source_indices,grad_target_indices)
			# source_indices = self.cumulative_data_indices[-1]
			# grad_target_indices =  self.cumulative_data_indices[-1]
				self.dataset_kwargs['map_index_curric'] = map_index_curric
				data_set = self.DataSet(self.dataset_kwargs['root_dir'], source_indices=source_indices,target_indices=grad_target_indices,**self.dataset_kwargs) #RotMNISTCGrad(source_indices,grad_target_indices,BIN_WIDTH,src_indices[0]-1,6,src_indices[idx]-1)
				# print("Training Cross grad with {}".format(len(data_set)))
				print("Transforming to {} domain with {} ex".format(idx,len(data_set)))
				for epoch in range(int(self.EPOCH*(1+idx / 8))):
					nl = 0
					ntd = 0
					data = torch.utils.data.DataLoader(data_set,self.BATCH_SIZE,self.shuffle)
					for img_1,img_2,time_diff in data:
						self.optimizer_gn.zero_grad()
						if self.encoder is not None:
							i1 = self.encoder(img_1)#.view(self.data_shape)
							i2 = self.encoder(img_2)#.view(self.data_shape)
						else:
							i1 = img_1 
							i2 = img_2
						# print(i1.size(),i2.size())
						time_diff_pred = self.model_gn(i1,i2) 
						loss = self.ord_class_loss_fn(time_diff.view(-1,1),time_diff_pred.view(-1,1))
						# print(loss,(1.0*(time_diff>0.0)),time_diff_pred)
						# assert False
						loss = loss.sum()
						loss.backward()
						self.optimizer_gn.step()
						with torch.no_grad():
							nl += loss.item()
							ntd += time_diff.sum().item()
					print('Epoch %d - %f %f' % (epoch, nl/len(data_set),ntd/len(data_set)))
					# print('Epoch %d - %f %f \n' % (epoch, nl/len(data_set),ntd/len(data_set)),file=log)
					print(torch.cat([time_diff[:10].view(-1,1),time_diff_pred[:10].view(-1,1)],dim=1).detach().cpu().numpy(),file=log)
					# log.write("\n")

		log.close()
		self.dataset_kwargs["drop_cols_classifier"] = None


	def test_ord_classifier(self):
		source_indices = self.cumulative_data_indices[-2]
		grad_target_indices =  self.source_data_indices[-1]

		data_set = self.DataSet(self.dataset_kwargs['root_dir'], source_indices=source_indices,target_indices=grad_target_indices,**self.dataset_kwargs)
		print("TESTING")
		data = torch.utils.data.DataLoader(data_set,200,self.shuffle)

		all_td = []
		all_td_pred = []
		for img_1,img_2,time_diff in data:
			all_td.append(time_diff.view(-1,1).detach().cpu().numpy())
			all_td_pred.append(self.model_gn(img_1,img_2).view(-1,1).detach().cpu().numpy())

		all_td = np.concatenate(all_td,axis=0)
		all_td_pred = np.concatenate(all_td_pred,axis=0)

		acc = ((all_td_pred - all_td) ** 2).mean()
		var_pred = ((all_td_pred - all_td_pred.mean()) ** 2).mean()
		var_act  = ((all_td - all_td.mean()) ** 2).mean()

		print(acc,var_pred,var_act)
		self.lr = np.max(1 - (acc/(var_act + 1e-15)), 0)   # * (var_pred / (var_act + 1e-15))

		print(self.lr)


	def construct_final_dataset(self):

		if self.data == "house":
			self.dataset_kwargs["drop_cols_classifier"] = self.dataset_kwargs["drop_cols"]
		self.final_dataset = []
		past_data = torch.utils.data.DataLoader(ClassificationDataSet(indices=self.source_data_indices[len(self.source_domain_indices)-1],**self.dataset_kwargs),self.BATCH_SIZE,shuffle=False,drop_last=True) 
		time = self.target_a - 1 + self.lr
		new_images = []
		new_labels = []
		for img,a,u,label in past_data:
			# new_img = torch.zeros_like(img).normal_(0.,1.)
			new_img = img.clone().detach()
			new_img.requires_grad = True
			# optim = torch.optim.SGD([new_img],lr=1e-3)
			with torch.no_grad():
				if self.encoder is not None:
					i1 = self.encoder(img)#.view(self.data_shape)
					i2 = self.encoder(new_img)#.view(self.data_shape)
				else:
					i1 = img[:,:-1].clone().detach()        # Uncomment for MNIST
					i2 = new_img[:,:-1].clone().detach()    # Uncomment for MNIST
					# i1 = torch.cat([img,label.view(-1,1).float()],dim=1) 
					# i2 = torch.cat([new_img,label.view(-1,1).clone().detach().float()],dim=1)
			i2.requires_grad = True
			optim = torch.optim.SGD([i2], lr=0.5, momentum=0.9)
			for s in range(max(int(self.cg_steps*self.lr),1)):
				optim.zero_grad()
				tgt = ((a-time)>0)*1.0 if self.dataset_kwargs['return_binary'] else a - time
				loss = self.ord_class_loss_fn(self.model_gn(i1,i2) , tgt.view(-1,1)).sum()
				loss.backward()
				# grad = torch.autograd.grad(loss,i2)
				# if s % 50 == 0:
				print('Step %d - %f, %f , %f' % (s, loss.detach().cpu().numpy(), self.model_gn(i1,i2).detach().sum().cpu().numpy(), (a-time).view(-1,1).sum().cpu().detach()),flush=False)
				# with torch.no_grad():
				#   i2 = i2 - 7.5*grad[0].data
				#   i2 = i2.detach().clone()
				#   i2.requires_grad = True
				optim.step()
			new_images.append(i2.detach().cpu().numpy())
			new_labels.append(label.view(-1,1).detach().cpu().numpy())
			# print(torch.cat([img[:5,:1],img[:5,-3:],i2[:5,:1],i2[:5,-3:]],dim=-1).detach().cpu().numpy())
		new_ds_x, new_ds_y = np.vstack(new_images), np.vstack(new_labels)
		new_ds_u = np.hstack([np.array([time]*len(new_ds_x)).reshape(-1,1),np.array([self.target_u]*len(new_ds_x)).reshape(-1,1)])
		try:
			new_ds_x = np.hstack([new_ds_x,new_ds_u[:,0].reshape((-1,1))])
		except:
			pass
		print("Finetune with len {}".format(len(new_ds_x)))
		self.classification_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(new_ds_x).float().to(self.device),torch.tensor(new_ds_u[:,0]).float().view(-1,1).to(self.device),torch.tensor(new_ds_u[:,1]).view(-1,1).float().to(self.device),
			 torch.tensor(new_ds_y).long().to(self.device)),self.CLASSIFICATION_BATCH_SIZE,self.shuffle)
		self.dataset_kwargs["drop_cols_classifier"] = None


	def eval_classifier(self):
		# TODO change for handling regression
		if self.data == "house":
			self.dataset_kwargs["drop_cols_classifier"] = None
		td = ClassificationDataSet(indices=self.target_indices,**self.dataset_kwargs)
		target_dataset = torch.utils.data.DataLoader(td,self.BATCH_SIZE,self.shuffle,drop_last=False)
		Y_pred = []
		Y_label = []
		for batch_X, batch_A,batch_U, batch_Y in tqdm(target_dataset):
			batch_U = batch_U.view(-1,1)
			if self.encoder is not None:
				batch_X = self.encoder(batch_X)
			batch_Y_pred = self.classifier(batch_X, batch_A).detach().cpu().numpy()
			# print(batch_Y_pred.shape)
			if self.task == 'classification':
				Y_pred = Y_pred + [np.argmax(batch_Y_pred,axis=1)]
				Y_label = Y_label + [batch_Y.detach().cpu().numpy()]
			elif self.task == 'regression':
				Y_pred = Y_pred + [batch_Y_pred.reshape(-1,1)]
				Y_label = Y_label + [batch_Y.detach().cpu().numpy().reshape(-1,1)]
		print(len(Y_pred),len(Y_label))
		# print(Y_pred[0].shape,Y_label[0].shape)
		if self.task == 'classification':
			Y_pred = np.hstack(Y_pred)
			Y_label = np.hstack(Y_label)
			print('shape: ',Y_pred.shape)
			print(accuracy_score(Y_label, Y_pred))
			print(confusion_matrix(Y_label, Y_pred))
			print(classification_report(Y_label, Y_pred))    
		else:
			Y_pred = np.vstack(Y_pred)
			Y_label = np.vstack(Y_label)
			print('MAE: ',np.mean(np.abs(Y_label-Y_pred),axis=0))
			print('MSE: ',np.mean((Y_label-Y_pred)**2,axis=0))







	def train_classifier_sanity_check(self,j=[0,1,2,3,4]):
		self.sane_classifiers = [ClassifyNet(30,[10,2],1,time_conditioning=False,use_time2vec=False).to(self.device) for _ in range(len(self.source_domain_indices))]
		self.sane_classifier_optimizer = [torch.optim.Adam(self.sane_classifiers[i].parameters(),5e-2) for i in range(len(self.source_domain_indices))]
		self.dataset_kwargs["drop_cols_classifier"] = self.dataset_kwargs["drop_cols"]
		for i in range(len(self.source_domain_indices)):
			past_data = ClassificationDataSet(self.dataset_kwargs['root_dir'], indices=self.source_data_indices[i],**self.dataset_kwargs)
			# print(len(past_data))
			past_dataset = torch.utils.data.DataLoader((past_data),self.BATCH_SIZE,True)

			for epoch in range(self.CLASSIFIER_EPOCHS):
				
				class_loss = 0
				for batch_X, batch_A, batch_U, batch_Y in tqdm(past_dataset):

					# batch_X = torch.cat([batch_X,batch_U.view(-1,2)],dim=1)

					l = train_classifier_batch(X=batch_X,dest_u=None,dest_a=None,Y=batch_Y,classifier=self.sane_classifiers[i],classifier_optimizer=self.sane_classifier_optimizer[i],verbose=False,encoder=None, batch_size=self.BATCH_SIZE,loss_fn=self.classifier_loss_fn)
					class_loss += l
				print("Domain %d Loss %f"%(i,class_loss/len(past_data)),flush=False)

			torch.save(self.sane_classifiers[i].state_dict(), "sanity_check_{}".format(i))


	def get_sane_time_diff(self,i1,i2):
		i1_lab = i1[:,-1]
		i2_lab = i2[:,-1]
		preds_i1 = torch.zeros(i1.size(0),len(self.source_domain_indices))
		preds_i2 = torch.zeros(i1.size(0),len(self.source_domain_indices))

		for i in range(len(self.source_domain_indices)):
			# print(self.sane_classifiers[i](i1[:,:-1]).size(),i1_lab.size())
			# print(torch.cat([self.sane_classifiers[i](i1[:10,:-1]).view(-1,1),i1_lab[:10].view(-1,1)],dim=1))
			preds_i1[:,i] = torch.abs(self.sane_classifiers[i](i1[:,:-1]) - i1_lab.view(-1,1)).squeeze(1) 
			preds_i2[:,i] = torch.abs(self.sane_classifiers[i](i2[:,:-1]) - i2_lab.view(-1,1)).squeeze(1)

		preds_i1 = preds_i1.argmin(dim=1).view(-1,1)
		preds_i2 = preds_i2.argmin(dim=1).view(-1,1)

		time_diffs = (torch.tensor(self.source_domain_indices)/12.0).repeat(i1.size(0),1).float()
		return 10*(torch.gather(time_diffs,1,preds_i1) - torch.gather(time_diffs,1,preds_i2)).to(self.device)


	def eval_sane_classifiers(self):
		log = open("cross-grad-log.txt","w")
		self.sane_classifiers = [ClassifyNet(30,[10,2],1,time_conditioning=False,use_time2vec=False).to(self.device) for _ in range(len(self.source_domain_indices))]
		for i in range(len(self.source_domain_indices)):
			self.sane_classifiers[i].load_state_dict(torch.load("sanity_check_{}".format(i)))

		for idx in range(1, len(self.source_domain_indices)):

			source_indices = self.cumulative_data_indices[idx-1]
			grad_target_indices =  self.source_data_indices[idx]

			data_set = self.DataSet(self.dataset_kwargs['root_dir'], source_indices=source_indices,target_indices=grad_target_indices,**self.dataset_kwargs) #RotMNISTCGrad(source_indices,grad_target_indices,BIN_WIDTH,src_indices[0]-1,6,src_indices[idx]-1)
			nl = 0
			ntd = 0
			data = torch.utils.data.DataLoader(data_set,self.BATCH_SIZE,self.shuffle)
			for img_1,img_2,time_diff in data:
				if self.encoder is not None:
					i1 = self.encoder(img_1)#.view(self.data_shape)
					i2 = self.encoder(img_2)#.view(self.data_shape)
				else:
					i1 = img_1 
					i2 = img_2
				# print(i1.size(),i2.size())
				time_diff_pred = self.get_sane_time_diff(i1,i2)
				loss = torch.abs((time_diff.view(-1,1) - time_diff_pred.view(-1,1))).sum()
				with torch.no_grad():
					nl += loss.item()
					ntd += time_diff.sum().item()
			print('Epoch %d - %f %f' % (1, nl/len(data_set),ntd/len(data_set)))
			print('Epoch %d - %f %f \n' % (1, nl/len(data_set),ntd/len(data_set)),file=log)
			print(torch.cat([time_diff[10:40].view(-1,1),time_diff_pred[10:40].view(-1,1)],dim=1).detach().cpu().numpy(),file=log)
	def train(self):

		# print(self.encoder)
		
		# self.train_classifier(encoder=self.encoder)  # Train classifier initially
		# torch.save(self.classifier.state_dict(), "classifier_house_reg.pth")
		# self.classifier.load_state_dict(torch.load("classifier_house_reg.pth"))
		# self.eval_classifier()
		# print(self.dataset_kwargs)

		self.train_cross_grad()  # Train model for cross-grad
		self.test_ord_classifier()
		self.construct_final_dataset()  # Perturb source dataset for finetuning
		# self.CLASSIFIER_EPOCHS = self.CLASSIFIER_EPOCHS//2
		# print(self.dataset_kwargs)
		# if self.lr > 0.3:
		#   self.train_classifier(self.classification_dataset)
		# else:
		#   self.train_classifier(encoder=self.encoder)

		self.eval_classifier()

		# self.train_classifier_sanity_check()
		# self.eval_sane_classifiers()
'''Game plan

We need to have all dataloaders give out tensors, take away numpy array stuff from train loop.
Break down train loop into multiple functions as well.
We give only time, X to all models. This will make stuff compatible. We then change the models to do appropriate things.
'''