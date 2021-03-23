'''Abstraction

This is an abstract class for all trainers, it wraps up datasets etc. The train script will call this with the appropriate params
''' 

import torch 
import numpy as np
import json
import pickle

from matplotlib import pyplot as plt

# from models import *
from utils import *
from dataset_GI import *
from config_GI import Config
from tqdm import tqdm
from losses import *
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.tensorboard import SummaryWriter
import time 

np.set_printoptions(precision = 3)



def train_classifier_batch(X,dest_u,dest_a,Y,classifier,classifier_optimizer,batch_size,verbose=False,encoder=None, transformer=None,source_u=None, kernel=None,loss_fn=classification_loss):
	'''Trains classifier on a batch
	
	
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
	log = open("Classifier_log.txt","a")
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

	# if verbose:
	# 	with torch.no_grad():
	# 		print(torch.cat([Y_pred[:20],Y[:20].view(-1,1).float(),pred_loss[:20].view(-1,1)],dim=1).detach().cpu().numpy(),file=log)

	if kernel is not None:
		pred_loss = pred_loss * kernel
	pred_loss = pred_loss.sum()/batch_size
	pred_loss.backward()
	classifier_optimizer.step()
	


	return pred_loss


def finetune(X, U, Y, delta, classifier, classifier_optimizer,classifier_loss_fn,writer=None,step=None):
	'''Finetunes model using gradient interpolation with given `delta`
	
	[description]
	
	Arguments:
		X {[type]} -- Datapoints
		U {[type]} -- Time indices
		Y {[type]} -- Labes
		delta {[type]} -- 
		classifier {[type]} -- Model
		classifier_optimizer {[type]} -- Optimizer
		classifier_loss_fn {[type]} -- Loss function, can be MSE, MAE, CE etc
	
	Keyword Arguments:
		writer {[type]} -- Tensorboard writer
		step {[type]} -- Step for tensorboard writer
	
	'''
	classifier_optimizer.zero_grad()

	U_grad = U.clone() - delta
	U_grad.requires_grad_(True)
	# Y_pred = classifier(torch.cat([X[:,:-2],X[:,-2].view(-1,1)-delta.view(-1,1),U_grad.view(-1,1)],dim=1), U_grad)
	Y_pred = classifier(X,U_grad,logits=True)
	partial_Y_pred_t = torch.autograd.grad(Y_pred, U_grad, grad_outputs=torch.ones_like(Y_pred), retain_graph=True)[0]
	

	Y_pred = Y_pred + delta * partial_Y_pred_t

	if Y_pred.size(-1) > 1:  # Not regression, in this case we need to apply softmax to the logits
		Y_pred = torch.softmax(Y_pred,dim=-1)
	pred_loss = classifier_loss_fn(Y_pred,Y).mean()
	pred_loss.backward()
	
	classifier_optimizer.step()

	return pred_loss


def adversarial_finetune(X, U, Y, delta, classifier, classifier_optimizer,classifier_loss_fn,delta_lr=0.1,delta_clamp=0.15,delta_steps=10,lambda_GI=0.5,writer=None,step=None,string=None):
	
	classifier_optimizer.zero_grad()
	
	delta.requires_grad_(True)
	
	# This block of code computes delta adversarially
	for ii in range(delta_steps):

		U_grad = U.clone() - delta
		U_grad.requires_grad_(True)
		Y_pred = classifier(X, U_grad, logits=True)
		if len(Y.shape)>1 and Y.shape[1] > 1:
			Y_true = torch.argmax(Y, 1).view(-1,1).float()

		partial_logit_pred_t = []
		if len(Y_pred.shape)<2 or Y_pred.shape[1] < 2:
			partial_Y_pred_t = torch.autograd.grad(Y_pred, U_grad, grad_outputs=torch.ones_like(Y_pred), retain_graph=True)[0]
		else:           
			for idx in range(Y_pred.shape[1]):
				logit = Y_pred[:,idx].view(-1,1)
				partial_logit_pred_t.append(torch.autograd.grad(logit, U_grad, grad_outputs=torch.ones_like(logit), retain_graph=True)[0])

			
				partial_Y_pred_t = torch.cat(partial_logit_pred_t, 1)


		# partial_Y_pred_t = torch.autograd.grad(Y_pred, U_grad, grad_outputs=torch.ones_like(Y_pred), retain_graph=True)[0]
		Y_pred = Y_pred + delta * partial_Y_pred_t
		if len(Y_pred.shape)>1 and Y_pred.shape[1] > 1:
			Y_pred = torch.softmax(Y_pred,dim=-1)
		#print(Y_pred.shape)
		#print(Y_pred[0:10])
		# loss = -torch.mean(Y_true * torch.log(Y_pred + 1e-9))
		loss = classifier_loss_fn(Y_pred,Y)
		partial_loss_delta = torch.autograd.grad(loss, delta, grad_outputs=torch.ones_like(loss), retain_graph=True)[0]
		delta = delta + delta_lr*partial_loss_delta
		# with torch.no_grad():
		#   print('{} {}'.format(ii, delta.cpu().clone().detach().numpy()))
	
	delta = delta.clamp(-1*delta_clamp, delta_clamp).detach().clone()
	# print(step,writer is not None)
	if writer is not None:
		writer.add_histogram(string,delta.view(1,-1),step)

	# This block of code actually optimizes our model
	U_grad = U.clone() - delta
	U_grad.requires_grad_(True)
	Y_pred = classifier(X, U_grad, logits=True)

	partial_logit_pred_t = []

	if len(Y_pred.shape)<2 or Y_pred.shape[1] < 2:
		partial_Y_pred_t = torch.autograd.grad(Y_pred, U_grad, grad_outputs=torch.ones_like(Y_pred), retain_graph=True)[0]
	else:
		for idx in range(Y_pred.shape[1]):
			logit = Y_pred[:,idx].view(-1,1)
			partial_logit_pred_t.append(torch.autograd.grad(logit, U_grad, grad_outputs=torch.ones_like(logit), retain_graph=True)[0])
		#print(partial_logit_pred_t)
			partial_Y_pred_t = torch.cat(partial_logit_pred_t, 1)

	# partial_Y_pred_t = torch.cat(partial_logit_pred_t, 1)
	#print(partial_Y_pred_t.shape)
	# partial_Y_pred_t = torch.autograd.grad(Y_pred, U_grad, grad_outputs=torch.ones_like(Y_pred), retain_graph=True)[0]
	Y_pred = Y_pred + delta * partial_Y_pred_t
	if len(Y_pred.shape)>1 and Y_pred.shape[1] > 1:
		Y_pred = torch.softmax(Y_pred,dim=-1)
	# Y_true = Y
	#Y_true = torch.argmax(Y, 1).view(-1,1).float()

	
	# pred_loss = -torch.mean(Y_true * torch.log(Y_pred + 1e-9))# + (1 - Y_true) * torch.log(1 - Y_pred + 1e-9))
	#pred_loss = torch.mean((Y_pred - Y_true)**2)
	Y_orig_pred = classifier(X,U)
	pred_loss = classifier_loss_fn(Y_pred,Y).mean() + lambda_GI*classifier_loss_fn(Y_orig_pred,Y).mean() #+ 5*(partial_Y_pred_t**2).mean()
	pred_loss.backward()
	
	classifier_optimizer.step()

	return pred_loss



# It is possible that the gains are due to "finetuning" and not due to the gradient regularization
# Multi-step ideas
# Saturate loss first then do grad-reg (finetune initially?)
# Radomize domains, add some early stopping?
# Ensemble final prediction?
# Gradient decreases as delta increases, also performance increasing till a point then goes down
# Early stopping not working!
class GradRegTrainer():
	def __init__(self,args):

		# self.DataSet = MetaDataset

		config = Config(args)
	
		self.DataSetClassifier = ClassificationDataSet
		self.CLASSIFIER_EPOCHS = config.epoch_classifier
		self.FINETUNING_EPOCHS = config.epoch_finetune 
		self.SUBEPOCHS = 1
		self.BATCH_SIZE = config.bs
		self.CLASSIFICATION_BATCH_SIZE = 100
		# self.PRETRAIN_EPOCH = 5
		self.data = args.data 
		self.update_num_steps = 1

		self.writer = SummaryWriter(comment='{}'.format(time.time()),log_dir="new_runs")


		self.dataset_kwargs = config.dataset_kwargs
		self.source_domain_indices = config.source_domain_indices   #[6,7,8,9,10]
		self.target_domain_indices = config.target_domain_indices   #[11]
		data_index_file = config.data_index_file    #"../../data/HousePrice/indices.json"
		self.classifier = config.classifier(**config.model_kwargs).to(args.device) 
		self.lr = config.lr
		self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(),config.lr)
		# loss_type = 'bce' if self.dataset_kwargs['return_binary'] else 'reg'
		
		self.classifier_loss_fn = config.classifier_loss_fn   #reconstruction_loss
		self.task = config.loss_type
		
		if args.encoder:
			self.encoder = config.encoder(**config.encoder_kwargs).to(args.device)
		else:
			self.encoder = None

		self.delta_lr=config.delta_lr
		self.delta_clamp=config.delta_clamp
		self.delta_steps=config.delta_steps
		self.lambda_GI=config.lambda_GI
		self.num_finetune_domains = config.num_finetune_domains

		data_indices = json.load(open(data_index_file,"r")) #, allow_pickle=True)
		self.source_data_indices = [data_indices[i] for i in self.source_domain_indices]
		self.cumulative_data_indices = get_cumulative_data_indices(self.source_data_indices)
		# print(self.cumulative_data_indices)
		self.target_indices = [data_indices[i] for i in self.target_domain_indices][0]  # TODO Flatten this list instead of picking 0th ele
		self.shuffle = True
		self.device = args.device
		self.delta  = args.delta
		self.patience = 2
		self.early_stopping = args.early_stopping
		if args.seed is not None:
			torch.random.manual_seed(int(args.seed))
			np.random.seed(int(args.seed))
		self.seed = args.seed


	def train_classifier(self,past_dataset=None,encoder=None):
			'''Train the classifier initially
			
			In its current form the function just trains a baseline model on the entire train data
			
			Keyword Arguments:
				past_dataset {[type]} -- If this is not None, then the `past_dataset` is used to train the model (default: {None})
				encoder {[type]} -- Encoder model to train (default: {None})
			'''
			class_step = 0
		# for i in range(len(self.source_domain_indices)):
		# if past_dataset is None:
			past_data = ClassificationDataSet(indices=self.cumulative_data_indices[-1],**self.dataset_kwargs)
			past_dataset = torch.utils.data.DataLoader((past_data),self.BATCH_SIZE,True)
			for epoch in range(self.CLASSIFIER_EPOCHS):
				
				class_loss = 0
				for batch_X, batch_A, batch_U, batch_Y in tqdm(past_dataset):

					# batch_X = torch.cat([batch_X,batch_U.view(-1,2)],dim=1)

					l = train_classifier_batch(X=batch_X,dest_u=batch_U,dest_a=batch_U,Y=batch_Y,classifier=self.classifier,classifier_optimizer=self.classifier_optimizer,verbose=(class_step%20)==0,encoder=encoder, batch_size=self.BATCH_SIZE,loss_fn=self.classifier_loss_fn)
					class_step += 1
					class_loss += l
					self.writer.add_scalar("loss/classifier",l.item(),class_step)
				print("Epoch %d Loss %f"%(epoch,class_loss),flush=False)

			# past_dataset = None

	def finetune_grad_int(self, num_domains=2):
		'''Finetunes using gradient interpolation
		
		Keyword Arguments:
			num_domains {number} -- Number of domains on which to fine-tune (default: {2})
		'''

		dom_indices = np.arange(len(self.source_domain_indices)).astype('int')[-1*num_domains:]

		for i in dom_indices:
			self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(),5e-4)
			past_data = ClassificationDataSet(indices=self.source_data_indices[i],**self.dataset_kwargs)
			past_dataset = torch.utils.data.DataLoader((past_data),self.BATCH_SIZE,True)
			
			# For early stopping we look at the next domain, if that is not possible we look at training loss itself 
			try:
				val_dataset =  torch.utils.data.DataLoader(ClassificationDataSet(indices=self.source_data_indices[i+1],**self.dataset_kwargs),self.BATCH_SIZE,True)
				# print("Sucess")
			except:
				val_dataset =  torch.utils.data.DataLoader(ClassificationDataSet(indices=self.source_data_indices[i],**self.dataset_kwargs),self.BATCH_SIZE,True)
			bad_ep = 0
			prev_net_val_loss = 1000000000

			# print('Finetuning step %d Domain %d' %(ii, index))
			# ii+=1
			# print('------------------------------------------------------------------------------------------')
			step = 0
			for epoch in range(self.FINETUNING_EPOCHS):
				
				loss = 0
				for batch_X, _, batch_U, batch_Y in tqdm(past_dataset):

					batch_U = batch_U.view(-1,1)
					delta = (torch.rand(batch_U.size()).float()*(0.1-(-0.1)) - 0.1).to(batch_X.device)
					# TODO pass delta hyperparams here
					l = adversarial_finetune(batch_X, batch_U, batch_Y, delta, self.classifier, self.classifier_optimizer,self.classifier_loss_fn,delta_lr=self.delta_lr,delta_clamp=self.delta_clamp,delta_steps=self.delta_steps,lambda_GI=self.lambda_GI,writer=self.writer,step=step,string="delta_{}".format(i))
					loss = loss + l
					self.writer.add_scalar("loss/test_{}".format(i),l.item(),step)
					step += 1

				print("Epoch %d Loss %f"%(epoch,loss))
			
			# Validation -
				if self.early_stopping:
					with torch.no_grad():
						net_val_loss = 0
						for batch_X, _, batch_U, batch_Y in tqdm(val_dataset):
							batch_Y_pred = self.classifier(batch_X, batch_U)
							net_val_loss += self.classifier_loss_fn(batch_Y_pred,batch_Y).sum().item()
						
						if net_val_loss > prev_net_val_loss:
							bad_ep += 1
						else:
							# torch.save()
							best_model = self.classifier.state_dict()
							bad_ep = 0
						# print(prev_net_val_loss,net_val_loss)
						prev_net_val_loss = min(net_val_loss,prev_net_val_loss)

					if bad_ep > self.patience:
						print("Early stopping for domain {}".format(i))
						self.classifier.load_state_dict(best_model)
						break


	def eval_classifier(self):

		if self.data == "house":
			self.dataset_kwargs["drop_cols_classifier"] = None
		td = ClassificationDataSet(indices=self.target_indices,**self.dataset_kwargs)
		target_dataset = torch.utils.data.DataLoader(td,self.BATCH_SIZE,self.shuffle,drop_last=False)
		Y_pred = []
		Y_label = []
		for batch_X, batch_A,batch_U, batch_Y in target_dataset:
			batch_U = batch_U.view(-1,1)
			if self.encoder is not None:
				batch_X = self.encoder(batch_X)
			batch_Y_pred = self.classifier(batch_X, batch_U).detach().cpu().numpy()
			# print(batch_Y_pred.shape)
			if self.task == 'classification':
				if batch_Y_pred.shape[1] > 1:
					Y_pred = Y_pred + [np.argmax(batch_Y_pred,axis=1)]
				else:
					Y_pred = Y_pred + [(batch_Y_pred>0.5)*1.0]

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
			# print(np.hstack([Y_pred,Y_label]))
			print(Y_pred.shape,Y_label.shape)
			print('MAE: ',np.mean(np.abs(Y_label-Y_pred),axis=0))
			print('MSE: ',np.mean((Y_label-Y_pred)**2,axis=0))


	def visualize_trajectory(self,indices,filename=''):
		td = ClassificationDataSet(indices=indices,**self.dataset_kwargs)
		fig, ax = plt.subplots(3, 3)
		ds = iter(torch.utils.data.DataLoader(td,1,False))
		for i in range(3):
			for j in range(3):
				x,a,u,y = next(ds)
				x_ = []
				y_ = []
				y__ = []
				y___ = []
				actual_time = u.view(1).detach().cpu().numpy()
				for t in np.arange(actual_time-0.2,actual_time+0.2,0.002):
					x_.append(t)
					t = torch.tensor([t]).float().to(x.device)
					t.requires_grad_(True)
					delta = (x[0,-1] - t).detach()
					# The last two features in the housing dataset are time, so it makes sense to pass these while visualizing
					y_pred = self.classifier(torch.cat([x[:,:-2],x[:,-2].view(-1,1)-delta.view(-1,1), t.view(-1,1)],dim=1), t.view(-1,1)) 
					partial_Y_pred_t = torch.autograd.grad(y_pred, t, grad_outputs=torch.ones_like(y_pred))[0]
					y_.append(y_pred.item())
					y__.append(partial_Y_pred_t.item())
					y___.append((partial_Y_pred_t*delta + y_pred).item())
					# TODO gradient addition business
				ax[i,j].plot(x_,y_)
				ax[i,j].plot(x_,y__)
				ax[i,j].plot(x_,y___)
				ax[i,j].set_title("time-{}".format(actual_time))

				# print(x_,y_)
				ax[i,j].scatter(u.view(-1,1).detach().cpu().numpy(),y.view(-1,1).detach().cpu().numpy())
		plt.savefig('traj_{}.png'.format(filename))
		plt.close()

	def train(self):

		
		self.train_classifier(encoder=self.encoder)  # Train classifier initially
		# torch.save(self.classifier.state_dict(), "classifier.pth")
		# self.classifier.load_state_dict(torch.load("classifier_time_huge.pth"))      
		# vis_ind = np.array(self.cumulative_data_indices[-1])
		# np.random.shuffle(vis_ind)
		# vis_ind = [self.source_data_indices[0][3], self.source_data_indices[1][47], self.source_data_indices[2][102], self.source_data_indices[2][210], self.source_data_indices[3][168], self.source_data_indices[3][342], self.source_data_indices[4][42], self.source_data_indices[4][44],self.source_data_indices[4][189]]
		# self.visualize_trajectory(vis_ind[:9],"plots/{}_{}_base".format(self.seed,self.delta))
		print("-----------------------------------------")
		print("Performance of the base classifier")
		self.eval_classifier()
		self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(),self.lr)
		self.finetune_grad_int(num_domains=self.num_finetune_domains)
		# self.visualize_trajectory(vis_ind[:9],"plots/{}_{}".format(self.seed,self.delta))
		
		print("-----------------------------------------")
		print("Performance after fine-tuning")
		self.eval_classifier()



'''Game plan

We need to have all dataloaders give out tensors, take away numpy array stuff from train loop.
Break down train loop into multiple functions as well.
We give only time, X to all models. This will make stuff compatible. We then change the models to do appropriate things.
'''