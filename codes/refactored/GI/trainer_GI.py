'''Abstraction

This is an abstract class for all trainers, it wraps up datasets etc. The train script will call this with the appropriate params
''' 

import torch 
import numpy as np
import json
import pickle
import os 

from matplotlib import pyplot as plt

# from models import *
from utils import *
from dataset_GI import *
from config_GI import Config
from tqdm import tqdm
from losses import *
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
import time 

np.set_printoptions(precision = 3)


def visualize_single(x,u,classifier):
	x_ = []
	y_ = []
	y__ = []
	y___ = []
	actual_time = u.view(1).detach().cpu().numpy()
	for t in np.arange(actual_time-0.2,actual_time+0.2,0.002):
		x_.append(t)
		t = torch.tensor([t]).float().to(x.device)
		t.requires_grad_(True)
		delta = (u - t).detach()
		# The last two features in the housing dataset are time, so it makes sense to pass these while visualizing
		y_pred = classifier(x.view(1,-1),t.view(-1,1)) 
		partial_Y_pred_t = torch.autograd.grad(y_pred, t, grad_outputs=torch.ones_like(y_pred),create_graph=True)[0]
		pred = partial_Y_pred_t #*delta + y_pred
		pred_grad = torch.autograd.grad(y_pred,t,grad_outputs=torch.ones_like(y_pred))[0]
		# print(make_dot(pred_grad,params={**{"delta":t},**(dict(classifier.named_parameters()))},show_saved=True))
		# assert False
		y_.append(y_pred.item())
		y__.append(partial_Y_pred_t.item())
		y___.append((partial_Y_pred_t*delta + y_pred).item())


		# TODO gradient addition business
	plt.plot(x_,y_)
	# plt.plot(x_,y__)
	grad = np.array(y__)
	grad = grad[1:] - grad[:-1]
	grad = grad/0.002
	print(y_,grad)
	# plt.plot(x_,y___)
	# plt.plot(x_[1:],grad)
	# plt.ylim(-10,10)
	# plt.set_title("time-{}".format(actual_time))
	plt.savefig('plot.png')
	assert False


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
	# print(torch.cat([Y_pred[:20],Y[:20].view(-1,1).float(),pred_loss[:20].view(-1,1)],dim=1).detach().cpu().numpy())

	# if verbose:
	#   with torch.no_grad():
	#       print(torch.cat([Y_pred[:20],Y[:20].view(-1,1).float(),pred_loss[:20].view(-1,1)],dim=1).detach().cpu().numpy(),file=log)

	if kernel is not None:
		pred_loss = pred_loss * kernel
	pred_loss = pred_loss.sum()/batch_size
	pred_loss.backward()
	classifier_optimizer.step()
	


	return pred_loss


def finetune(X, U, Y, delta, classifier, classifier_optimizer,classifier_loss_fn,writer=None,step=None, **kwargs):
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
	Y_pred = classifier(X, U_grad, logits=True)

	partial_logit_pred_t = []

	if len(Y_pred.shape)<2 or Y_pred.shape[1] < 2:
		partial_Y_pred_t = torch.autograd.grad(Y_pred, U_grad, grad_outputs=torch.ones_like(Y_pred), create_graph=True)[0]
	else:
		for idx in range(Y_pred.shape[1]):
			logit = Y_pred[:,idx].view(-1,1)
			partial_logit_pred_t.append(torch.autograd.grad(logit, U_grad, grad_outputs=torch.ones_like(logit), create_graph=True)[0])
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
	pred_loss = classifier_loss_fn(Y_pred,Y).mean() + 0.5*classifier_loss_fn(Y_orig_pred,Y).mean() #+ 5*(partial_Y_pred_t**2).mean()
	pred_loss.backward()
	
	classifier_optimizer.step()

	return pred_loss

	# U_grad = U.clone() - delta
	# U_grad.requires_grad_(True)
	# # Y_pred = classifier(torch.cat([X[:,:-2],X[:,-2].view(-1,1)-delta.view(-1,1),U_grad.view(-1,1)],dim=1), U_grad)
	# Y_pred = classifier(X,U_grad,logits=True)
	# partial_Y_pred_t = torch.autograd.grad(Y_pred, U_grad, grad_outputs=torch.ones_like(Y_pred), retain_graph=True)[0]
	

	# Y_pred = Y_pred + delta * partial_Y_pred_t

	# if Y_pred.size(-1) > 1:  # Not regression, in this case we need to apply softmax to the logits
	#   Y_pred = torch.softmax(Y_pred,dim=-1)
	# pred_loss = classifier_loss_fn(Y_pred,Y).mean()
	# pred_loss.backward()
	
	# classifier_optimizer.step()

	# return pred_loss


def additive_finetune(X, U, Y, delta, classifier, classifier_optimizer,classifier_loss_fn,delta_lr=0.1,delta_clamp=0.15,delta_steps=10,lambda_GI=0.5,writer=None,step=None,string=None):

	classifier_optimizer.zero_grad()

	U_grad = U.clone() - delta
	U_grad.requires_grad_(True)
	Y_pred = classifier(X, U_grad,delta=delta)
	Y_pred_ = classifier(X,U)



	pred_loss_1 = classifier_loss_fn(Y_pred,Y).mean() 
	pred_loss_2 = classifier_loss_fn(Y_pred_,Y).mean()

	pred_loss = pred_loss_1 + pred_loss_2
	# print(pred_loss)
	# print(pred_loss_1.item(),pred_loss_2.item(), pred_loss.item())

	pred_loss.backward()
	classifier_optimizer.step()

	return pred_loss



def adversarial_finetune(X, U, Y, delta, classifier, classifier_optimizer,classifier_loss_fn,delta_lr=0.1,delta_clamp=0.15,delta_steps=10,lambda_GI=0.5,writer=None,step=None,string=None):
	
	classifier_optimizer.zero_grad()
	
	delta.requires_grad_(True)

	# This block of code computes delta adversarially
	d1 = delta.detach().cpu().numpy()
	# print(d1)
	# optim = torch.optim.SGD([delta], lr=delta_lr,momentum=0.9)
	for ii in range(delta_steps):
		# optim.zero_grad()
		delta = delta.clone().detach()
		delta.requires_grad_(True)
		U_grad = U.clone() - delta
		U_grad.requires_grad_(True)
		Y_pred = classifier(X, U_grad, logits=True)
		if len(Y.shape)>1 and Y.shape[1] > 1:
			Y_true = torch.argmax(Y, 1).view(-1,1).float()

		partial_logit_pred_t = []
		if len(Y_pred.shape)<2 or Y_pred.shape[1] < 2:
			partial_Y_pred_t = torch.autograd.grad(Y_pred, U_grad, grad_outputs=torch.ones_like(Y_pred), retain_graph=True,create_graph=True)[0]
		else:           
			for idx in range(Y_pred.shape[1]):
				logit = Y_pred[:,idx].view(-1,1)
				partial_logit_pred_t.append(torch.autograd.grad(logit, U_grad, grad_outputs=torch.ones_like(logit), create_graph=True)[0])

			
			partial_Y_pred_t = torch.cat(partial_logit_pred_t, 1)


		# partial_Y_pred_t = torch.autograd.grad(Y_pred, U_grad, grad_outputs=torch.ones_like(Y_pred), retain_graph=True)[0]
		# partial_Y_pred_t.requires_grad_(True)
		Y_pred = Y_pred + delta * partial_Y_pred_t

		if len(Y_pred.shape)>1 and Y_pred.shape[1] > 1:
			Y_pred = torch.softmax(Y_pred,dim=-1)
		# loss = -torch.mean(Y_true * torch.log(Y_pred + 1e-9))
		loss = classifier_loss_fn(Y_pred,Y).mean()
		# loss.backward()
		partial_loss_delta = torch.autograd.grad(loss, delta, grad_outputs=torch.ones_like(loss), retain_graph=True)[0]
		# print(partial_loss_delta,loss)
		delta = delta + delta_lr*partial_loss_delta

		if torch.norm(partial_loss_delta) < 1e-3 or delta > delta_clamp or delta < -1*delta_clamp:
			break
# 
		# print(torch.cat([loss.view(-1,1),y_grad.view(-1,1),partial_loss_delta,partial_Y_pred_t.view(-1,1), p.view(-1,1),(torch.sqrt(loss)*y_grad.squeeze()).view(-1,1)],dim=1))
		# with torch.no_grad():
		#   print('{} {}'.format(ii, delta.cpu().clone().detach().numpy()))
	delta = delta.clamp(-1*delta_clamp, delta_clamp).detach().clone()
	# print(ii,delta)
	d2 = delta.detach().cpu().numpy()
	# print(np.hstack([d1,d2]))
	# print(step,writer is not None)
	if writer is not None:
		writer.add_scalar(string,torch.abs(delta).mean(),step)
		writer.add_scalar(string+"_grad",torch.abs(partial_loss_delta).mean(),step)

	# This block of code actually optimizes our model
	U_grad = U.clone() - delta
	U_grad.requires_grad_(True)
	Y_pred = classifier(X, U_grad, logits=True)

	partial_logit_pred_t = []

	if len(Y_pred.shape)<2 or Y_pred.shape[1] < 2:
		partial_Y_pred_t = torch.autograd.grad(Y_pred, U_grad, grad_outputs=torch.ones_like(Y_pred), create_graph=True)[0]
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
	# print(pred_loss,Y_pred.size(),Y.size(),classifier_loss_fn(Y_pred,Y).size())
	pred_loss.backward()
	
	classifier_optimizer.step()

	return pred_loss, delta


def adversarial_finetune_goodfellow(X, U, Y, delta, classifier, classifier_optimizer,classifier_loss_fn,delta_lr=0.1,delta_clamp=0.15,delta_steps=10,lambda_GI=0.5,writer=None,step=None,string=None):
	
	classifier_optimizer.zero_grad()
	
	delta.requires_grad_(True)
	
	for ii in range(delta_steps):

		U_grad = U.clone() + delta
		U_grad.requires_grad_(True)
		Y_pred = classifier(X, U_grad)

		loss = classifier_loss_fn(Y_pred,Y).mean()
		partial_loss_delta = torch.autograd.grad(loss, delta, grad_outputs=torch.ones_like(loss), retain_graph=True)[0]

		delta = delta + delta_lr*partial_loss_delta
		#print('%d %f' %(ii, delta.clone().detach().numpy()))
	
	delta = delta.clamp(-1*delta_clamp, delta_clamp)
	delta = delta.detach()
	U_grad = U.clone() + delta
	U_grad.requires_grad_(True)
	Y_pred = classifier(X, U_grad)

	
	pred_loss = classifier_loss_fn(Y_pred,Y).mean()# + (1 - Y_true) * torch.log(1 - Y_pred + 1e-9))
	#pred_loss = torch.mean((Y_pred - Y_true)**2)
	pred_loss.backward()
	
	classifier_optimizer.step()

	return pred_loss


def finetune_gradient_regularization(X, U, Y, delta, classifier, classifier_optimizer,classifier_loss_fn,delta_lr=0.1,delta_clamp=0.15,delta_steps=10,lambda_GI=0.5,writer=None,step=None,string=None):
	# This does simple gradient regularization

	classifier_optimizer.zero_grad()
	time = U.clone().requires_grad_(True)
	Y_pred = classifier(X,time, logits=True)
	partial_Y_pred_t = []
	if len(Y_pred.shape)<2 or Y_pred.shape[1] < 2:
		partial_Y_pred_t = torch.autograd.grad(Y_pred, time, grad_outputs=torch.ones_like(Y_pred), create_graph=True)[0]
	else:
		for idx in range(Y_pred.shape[1]):
			logit = Y_pred[:,idx].view(-1,1)
			partial_Y_pred_t.append(torch.autograd.grad(logit, time, grad_outputs=torch.ones_like(logit), create_graph=True)[0])
		#print(partial_logit_pred_t)
		partial_Y_pred_t = torch.cat(partial_Y_pred_t, 1)

	if len(Y_pred.shape)>1 and Y_pred.shape[1] > 1:
		Y_pred = torch.softmax(Y_pred,dim=-1)

	grad = partial_Y_pred_t**2

	pred_loss = 1.0*classifier_loss_fn(Y_pred,Y).mean() + lambda_GI*(grad.mean())

	pred_loss.backward()
	
	classifier_optimizer.step()


	return pred_loss


def finetune_gradient_regularization_curvature(X, U, Y, delta, classifier, classifier_optimizer,classifier_loss_fn,delta_lr=0.1,delta_clamp=0.15,delta_steps=10,lambda_GI=0.5,writer=None,step=None,string=None):
	# This regularizes the norm of the second derivative

	classifier_optimizer.zero_grad()
	time = U.clone().requires_grad_(True)
	Y_pred = classifier(X,time, logits=True)
	partial_Y_pred_t = []
	if len(Y_pred.shape)<2 or Y_pred.shape[1] < 2:
		partial_Y_pred_t = torch.autograd.grad(Y_pred, time, grad_outputs=torch.ones_like(Y_pred), create_graph=True)[0]
	else:
		for idx in range(Y_pred.shape[1]):
			logit = Y_pred[:,idx].view(-1,1)
			partial_Y_pred_t.append(torch.autograd.grad(logit, time, grad_outputs=torch.ones_like(logit), create_graph=True)[0])
		#print(partial_logit_pred_t)
		partial_Y_pred_t = torch.cat(partial_Y_pred_t, 1)

	if len(Y_pred.shape)>1 and Y_pred.shape[1] > 1:
		Y_pred = torch.softmax(Y_pred,dim=-1)

	derivative = partial_Y_pred_t
	# print(derivative.size(),derivative.sum(dim=1).size())
	curvature = torch.autograd.grad(derivative,time,grad_outputs=torch.ones_like(Y_pred),create_graph=True)[0]
	pred_loss = classifier_loss_fn(Y_pred,Y).mean() + lambda_GI*(curvature**2).mean()
	pred_loss.backward()

	classifier_optimizer.step()

	# print((curvature**2).mean())
	


	return pred_loss    


def select_delta(X, U, Y, delta, classifier, classifier_optimizer,classifier_loss_fn,delta_lr=0.1,delta_clamp=0.15,delta_steps=10,lambda_GI=0.5,writer=None,step=None,string=None):

	classifier_optimizer.zero_grad()
	delta = delta.detach().clone()
	delta.requires_grad_(True)
	
	# This block of code computes delta adversarially
	d1 = delta.detach().cpu().numpy()
	for ii in range(delta_steps):

		U_grad = U.clone() - delta
		U_grad.requires_grad_(True)
		Y_pred = classifier(X, U_grad, logits=True)
		if len(Y.shape)>1 and Y.shape[1] > 1:
			Y_true = torch.argmax(Y, 1).view(-1,1).float()

		partial_logit_pred_t = []
		if len(Y_pred.shape)<2 or Y_pred.shape[1] < 2:
			partial_Y_pred_t = torch.autograd.grad(Y_pred, U_grad, grad_outputs=torch.ones_like(Y_pred), retain_graph=True,create_graph=True)[0]
		else:           
			for idx in range(Y_pred.shape[1]):
				logit = Y_pred[:,idx].view(-1,1)
				partial_Y_pred_t.append(torch.autograd.grad(logit, U_grad, grad_outputs=torch.ones_like(logit), retain_graph=True)[0])

			
			partial_Y_pred_t = torch.cat(partial_logit_pred_t, 1)


		# partial_Y_pred_t = torch.autograd.grad(Y_pred, U_grad, grad_outputs=torch.ones_like(Y_pred), retain_graph=True)[0]
		# partial_Y_pred_t.requires_grad_(True)
		# print(Y_pred.size(),partial_Y_pred_t.size(),delta.size())
		partial_Y_pred_t = partial_Y_pred_t.view(Y_pred.size())
		# print((delta * partial_Y_pred_t).size(),partial_Y_pred_t.size())
		Y_pred = Y_pred + delta * partial_Y_pred_t

		if len(Y_pred.shape)>1 and Y_pred.shape[1] > 1:
			Y_pred = torch.softmax(Y_pred,dim=-1)
		# loss = -torch.mean(Y_true * torch.log(Y_pred + 1e-9))
		loss = classifier_loss_fn(Y_pred,Y)
		partial_loss_delta = torch.autograd.grad(loss, delta, grad_outputs=torch.ones_like(loss), retain_graph=True)[0]

		delta = delta + delta_lr*partial_loss_delta
# 
		# print(torch.cat([loss.view(-1,1),y_grad.view(-1,1),partial_loss_delta,partial_Y_pred_t.view(-1,1), p.view(-1,1),(torch.sqrt(loss)*y_grad.squeeze()).view(-1,1)],dim=1))
		# with torch.no_grad():
		#   print('{} {}'.format(ii, delta.cpu().clone().detach().numpy()))
	
	delta = delta.clamp(-1*delta_clamp, delta_clamp).detach().clone()
	return delta


def finetune_num_int(X, U, Y, delta, k, classifier, classifier_optimizer,classifier_loss_fn,writer=None,step=None, **kwargs):

	classifier_optimizer.zero_grad()

	with torch.no_grad():
		Y_ = classifier(X,U)
	grad_sum = torch.zeros_like(Y_)
	pred_loss = None
	for i in range(1,k+1):
		# print(i)
		U_grad = U.clone() - i*delta
		U_grad.requires_grad_(True)
		Y_pred = classifier(X, U_grad, logits=True)

		partial_Y_pred_t = []

		if len(Y_pred.shape)<2 or Y_pred.shape[1] < 2:
			partial_Y_pred_t = torch.autograd.grad(Y_pred, U_grad, grad_outputs=torch.ones_like(Y_pred), retain_graph=True)[0]
		else:
			for idx in range(Y_pred.shape[1]):
				logit = Y_pred[:,idx].view(-1,1)
				partial_Y_pred_t.append(torch.autograd.grad(logit, U_grad, grad_outputs=torch.ones_like(logit), retain_graph=True)[0])
			#print(partial_logit_pred_t)
			partial_Y_pred_t = torch.cat(partial_Y_pred_t, 1)
	
	#print(partial_Y_pred_t)
		grad_sum = grad_sum + partial_Y_pred_t
		# print(Y_pred.size(),grad_sum.size())
		Y_pred = Y_pred + delta * grad_sum
		if len(Y_pred.shape)>1 and Y_pred.shape[1] > 1:
			Y_pred_ = torch.softmax(Y_pred,dim=-1)
		else:
			Y_pred_ = Y_pred

		if pred_loss is not None:
			pred_loss = pred_loss + classifier_loss_fn(Y_pred_,Y)
		else:
			pred_loss = classifier_loss_fn(Y_pred_,Y)


	Y_pred_ = classifier(X,U)
	if len(Y_pred_.shape)>1 and Y_pred_.shape[1] > 1:
		Y_pred_ = torch.softmax(Y_pred_,dim=-1)
	pred_loss_act = classifier_loss_fn(Y_pred_,Y)
	pred_loss = (pred_loss + pred_loss_act).mean()
	#pred_loss = torch.mean((Y_pred - Y_true)**2)
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



# TODOs - Delta adversarial, inference multistep (set self.delta well), time bias for final layer 
class GradRegTrainer():
	def __init__(self,args):

		# self.DataSet = MetaDataset
		if args.model == "baseline":
			from config_baseline import Config
			config = Config(args)
		elif args.model == "tbaseline" or args.model == "goodfellow" or args.model == "inc_finetune":
			from config_tbaseline import Config
			config = Config(args)
		elif args.model == "GI" or args.model == "t_inc_finetune" or args.model == "t_goodfellow" or args.model == "t_GI" or args.model == "grad_reg_curvature" or args.model == "grad_reg" or args.model == "fixed_GI" or args.model == "GI_t_delta" or args.model == "GI_v3" or args.model == "GI_num_int":
			from config_GI import Config
			config = Config(args)

		self.log = config.log
		self.goodfellow = args.goodfellow
		self.DataSetClassifier = ClassificationDataSet
		self.CLASSIFIER_EPOCHS = config.epoch_classifier
		self.FINETUNING_EPOCHS = config.epoch_finetune 

		self.SUBEPOCHS = 1
		self.BATCH_SIZE = config.bs
		self.CLASSIFICATION_BATCH_SIZE = 100
		# self.PRETRAIN_EPOCH = 5
		self.data = args.data 
		self.model = args.model
		self.update_num_steps = 1
		self.delta = config.delta
		self.trelu_limit = args.trelu_limit
		self.single_trelu = args.single_trelu
		self.ensemble = args.ensemble
		self.writer = None # SummaryWriter(comment='{}-{}-{}'.format(time.time(),self.model,self.data),log_dir="new_runs")


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


		self.inc_finetune = self.model in ["inc_finetune","t_inc_finetune", "t_GI"]
		
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
		self.max_k  = args.max_k
		self.patience = 2
		self.early_stopping = args.early_stopping
		self.seed = args.seed
		# print(self.delta)
		if self.model in ["GI_t_delta","GI"] :
			self.delta = [(torch.rand(1).float()*(0.1-(-0.1)) - 0.1).to(self.device) for _ in range(len(self.source_data_indices))]


	def train_classifier(self,past_dataset=None,encoder=None,inc_finetune=False):
		'''Train the classifier initially
		
		In its current form the function just trains a baseline model on the entire train data
		
		Keyword Arguments:
			past_dataset {[type]} -- If this is not None, then the `past_dataset` is used to train the model (default: {None})
			encoder {[type]} -- Encoder model to train (default: {None})
		'''
		if not self.inc_finetune:
			class_step = 0
			# for i in range(len(self.source_domain_indices)):
			# if past_dataset is None:
			past_data = ClassificationDataSet(indices=self.cumulative_data_indices[-1],**self.dataset_kwargs)
			past_dataset = torch.utils.data.DataLoader((past_data),self.BATCH_SIZE,True)
			for epoch in range(self.CLASSIFIER_EPOCHS):
				class_loss = 0
				for batch_X, batch_A, batch_U, batch_Y in (past_dataset):

					# batch_X = torch.cat([batch_X,batch_U.view(-1,2)],dim=1)
					l = train_classifier_batch(X=batch_X,dest_u=batch_U,dest_a=batch_U,Y=batch_Y,classifier=self.classifier,classifier_optimizer=self.classifier_optimizer,verbose=(class_step%20)==0,encoder=encoder, batch_size=self.BATCH_SIZE,loss_fn=self.classifier_loss_fn)
					class_step += 1
					class_loss += l
					if self.writer is not None:
						self.writer.add_scalar("loss/classifier",l.item(),class_step)
				print("Epoch %d Loss %f"%(epoch,class_loss/len(past_data)),flush=False)
			# past_dataset = None
		else:
			class_step = 0
			for i in range(len(self.source_domain_indices)):
			# if past_dataset is None:
				# self.classifier_optimizer
				past_data = ClassificationDataSet(indices=self.source_data_indices[i],**self.dataset_kwargs)
				past_dataset = torch.utils.data.DataLoader((past_data),self.BATCH_SIZE,True)
				for epoch in range(int(self.CLASSIFIER_EPOCHS*(1-(i/10)))):
					class_loss = 0
					for batch_X, batch_A, batch_U, batch_Y in tqdm(past_dataset):

						# batch_X = torch.cat([batch_X,batch_U.view(-1,2)],dim=1)

						l = train_classifier_batch(X=batch_X,dest_u=batch_U,dest_a=batch_U,Y=batch_Y,classifier=self.classifier,classifier_optimizer=self.classifier_optimizer,verbose=(class_step%20)==0,encoder=encoder, batch_size=self.BATCH_SIZE,loss_fn=self.classifier_loss_fn)
						class_step += 1
						class_loss += l
						if self.writer is not None:
							self.writer.add_scalar("loss/classifier",l.item(),class_step)
					print("Epoch %d Loss %f"%(epoch,class_loss/(len(past_data)/(self.BATCH_SIZE))),flush=False)


	def predict_ensemble(self,X,U,delta,k):
		# Ensemble predictions using linearized predictors

		# Get original prediction
		Y_pred = self.classifier(X,U,logits=True)

		# Get prediction at t-delta
		U_grad = U.clone() - delta 
		U_grad.requires_grad_(True)

		Y_pred_ = self.classifier(X, U_grad, logits=True)

		partial_Y_pred_t = []

		if len(Y_pred_.shape)<2 or Y_pred_.shape[1] < 2:
			partial_Y_pred_t = torch.autograd.grad(Y_pred_, U_grad, grad_outputs=torch.ones_like(Y_pred_), retain_graph=True)[0]
		else:
			for idx in range(Y_pred_.shape[1]):
				logit = Y_pred_[:,idx].view(-1,1)
				partial_Y_pred_t.append(torch.autograd.grad(logit, U_grad, grad_outputs=torch.ones_like(logit), retain_graph=True)[0])
			#print(partial_logit_pred_t)
			partial_Y_pred_t = torch.cat(partial_Y_pred_t, 1)

		Y_pred_ = Y_pred_ + delta * partial_Y_pred_t
		# print(Y_pred_ - Y_pred)
		if len(Y_pred.shape)>1 and Y_pred.shape[1] > 1:
			Y_pred = torch.softmax(Y_pred,dim=-1)

		print(torch.abs(Y_pred - Y_pred_).mean())
		Y_pred = (Y_pred + Y_pred_)/2
		# Get grad at   t-delta
		# Get linearized
		# Apply mean

		return Y_pred
	def finetune_grad_int(self, num_domains=2):
		'''Finetunes using gradient interpolation
		
		Keyword Arguments:
			num_domains {number} -- Number of domains on which to fine-tune (default: {2})
		'''
		dom_indices = np.arange(len(self.source_domain_indices)).astype('int')[-1*num_domains:]
		for i in dom_indices:
			delta_ = torch.FloatTensor(1,1).uniform_(-0.1,0.1).to(self.device) #self.delta 
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
				

				if self.model == "GI_t_delta":
					print("Selecting delta for {} domain  {} epoch".format(i,epoch))
					sample_batch_X, _, sample_batch_U, sample_batch_Y = next(iter(past_dataset))
					self.delta[i] = select_delta(sample_batch_X, sample_batch_U, sample_batch_Y, self.delta[i], self.classifier, self.classifier_optimizer,self.classifier_loss_fn,delta_lr=self.delta_lr,delta_clamp=self.delta_clamp,delta_steps=self.delta_steps,lambda_GI=self.lambda_GI,writer=self.writer,step=step,string="delta_{}".format(i))


				loss = 0
				for batch_X, _, batch_U, batch_Y in tqdm(past_dataset):

					batch_U = batch_U.view(-1,1)
					# print(batch_U)
					if self.model == "goodfellow" or self.model == "t_goodfellow":
						delta = (torch.zeros(batch_U.size()).float()).to(batch_X.device)
						# TODO pass delta hyperparams here
						l = adversarial_finetune_goodfellow(batch_X, batch_U, batch_Y, delta, self.classifier, self.classifier_optimizer,self.classifier_loss_fn,delta_lr=self.delta_lr,delta_clamp=self.delta_clamp,delta_steps=self.delta_steps,lambda_GI=self.lambda_GI,writer=self.writer,step=step,string="delta_{}".format(i))
					elif self.model in ["GI","t_GI"] :
						# delta = (torch.rand(batch_U.size()).float()*(0.1-(-0.1)) - 0.1).to(batch_X.device)
						# TODO pass delta hyperparams here
						# print(delta_)
						# delta_ = torch.FloatTensor(1,1).uniform_(-0.1,0.1).to(self.device) #self.delta 

						self.delta = torch.tensor(delta_) #+ torch.rand(1).float().to(self.device)*0.001
						l,delta_ = adversarial_finetune(batch_X, batch_U, batch_Y, self.delta, self.classifier, self.classifier_optimizer,self.classifier_loss_fn,delta_lr=self.delta_lr,delta_clamp=self.delta_clamp,delta_steps=self.delta_steps,lambda_GI=self.lambda_GI,writer=self.writer,step=step,string="delta_{}".format(i))
					elif self.model in ["grad_reg"]:
						delta = (torch.rand(batch_U.size()).float()*(0.1-(-0.1)) - 0.1).to(batch_X.device)
						# TODO pass delta hyperparams here
						l = finetune_gradient_regularization(batch_X, batch_U, batch_Y, delta, self.classifier, self.classifier_optimizer,self.classifier_loss_fn,delta_lr=self.delta_lr,delta_clamp=self.delta_clamp,delta_steps=self.delta_steps,lambda_GI=self.lambda_GI,writer=self.writer,step=step,string="delta_{}".format(i))

					elif self.model in ["grad_reg_curvature"]:
						delta = (torch.rand(batch_U.size()).float()*(0.1-(-0.1)) - 0.1).to(batch_X.device)
						# TODO pass delta hyperparams here
						l = finetune_gradient_regularization_curvature(batch_X, batch_U, batch_Y, delta, self.classifier, self.classifier_optimizer,self.classifier_loss_fn,delta_lr=self.delta_lr,delta_clamp=self.delta_clamp,delta_steps=self.delta_steps,lambda_GI=self.lambda_GI,writer=self.writer,step=step,string="delta_{}".format(i))
					elif self.model in ["fixed_GI"]:

						delta = self.delta #(torch.rand(batch_U.size()).float()*(0.1-(-0.1)) - 0.1).to(batch_X.device)
						# TODO pass delta hyperparams here
						l = finetune(batch_X, batch_U, batch_Y, delta, self.classifier, self.classifier_optimizer,self.classifier_loss_fn,delta_lr=self.delta_lr,delta_clamp=self.delta_clamp,delta_steps=self.delta_steps,lambda_GI=self.lambda_GI,writer=self.writer,step=step,string="delta_{}".format(i))
					elif self.model in ["GI_t_delta"]:

						delta = self.delta[i] #(torch.rand(batch_U.size()).float()*(0.1-(-0.1)) - 0.1).to(batch_X.device)
						# TODO pass delta hyperparams here
						l = finetune(batch_X, batch_U, batch_Y, delta, self.classifier, self.classifier_optimizer,self.classifier_loss_fn,delta_lr=self.delta_lr,delta_clamp=self.delta_clamp,delta_steps=self.delta_steps,lambda_GI=self.lambda_GI,writer=self.writer,step=step,string="delta_{}".format(i))
					elif self.model in ["GI_num_int"]:
						delta = self.delta 
						# k = int((epoch / self.FINETUNING_EPOCHS)*self.max_k) + 1  # This is curriculum learning, We may need a different function
						k = int(self.max_k)
						l = finetune_num_int(batch_X, batch_U, batch_Y, delta,k, self.classifier, self.classifier_optimizer,self.classifier_loss_fn,delta_lr=self.delta_lr,delta_clamp=self.delta_clamp,delta_steps=self.delta_steps,lambda_GI=self.lambda_GI,writer=self.writer,step=step,string="delta_{}".format(i))
					
					elif self.model in ["GI_v3"]:
						delta = self.delta 
						# k = int((epoch / self.FINETUNING_EPOCHS)*self.max_k)
						l = additive_finetune(batch_X, batch_U, batch_Y, delta, self.classifier, self.classifier_optimizer,self.classifier_loss_fn,delta_lr=self.delta_lr,delta_clamp=self.delta_clamp,delta_steps=self.delta_steps,lambda_GI=self.lambda_GI,writer=self.writer,step=step,string="delta_{}".format(i))  
					loss = loss + l
					if self.writer is not None:
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


	def eval_classifier(self,log, ensemble=False):

		if self.data == "house":
			self.dataset_kwargs["drop_cols_classifier"] = None
		self.classifier.eval()
		td = ClassificationDataSet(indices=self.target_indices,**self.dataset_kwargs)
		target_dataset = torch.utils.data.DataLoader(td,self.BATCH_SIZE,False,drop_last=False)
		Y_pred = []
		Y_label = []
		for batch_X, batch_A,batch_U, batch_Y in target_dataset:
			batch_U = batch_U.view(-1,1)
			if self.encoder is not None:
				batch_X = self.encoder(batch_X)
			if ensemble:
				batch_Y_pred = self.predict_ensemble(batch_X, batch_U, delta=self.delta,k=self.max_k).detach().cpu().numpy()
			else:
				batch_Y_pred = self.classifier(batch_X, batch_U).detach().cpu().numpy()
			if self.task == 'classification':
				if batch_Y_pred.shape[1] > 1:
					Y_pred = Y_pred + [np.argmax(batch_Y_pred,axis=1).reshape((-1,1))]
				else:
					Y_pred = Y_pred + [(batch_Y_pred>0.5)*1.0]

				# if batch_Y.shape[1] > 1:
				#   Y_label = Y_label + [np.argmax(batch_Y.detach().cpu().numpy(),axis=1).reshape((batch_Y_pred.shape[0],1))]
				# else:
				Y_label = Y_label + [batch_Y.detach().cpu().numpy()]
			elif self.task == 'regression':
				Y_pred = Y_pred + [batch_Y_pred.reshape(-1,1)]
				Y_label = Y_label + [batch_Y.detach().cpu().numpy().reshape(-1,1)]
		if self.task == 'classification':
			Y_pred = np.vstack(Y_pred)
			Y_label = np.hstack(Y_label)
			print('shape: ',Y_pred.shape, Y_label.shape)
			print('Accuracy: ',accuracy_score(Y_label, Y_pred),file=log)
			print(confusion_matrix(Y_label, Y_pred),file=log)
			print(classification_report(Y_label, Y_pred),file=log)    
		else:
			Y_pred = np.vstack(Y_pred)
			Y_label = np.vstack(Y_label)
			# print(np.hstack([Y_pred,Y_label]))
			print(Y_pred.shape,Y_label.shape)
			print('MAE: ',np.mean(np.abs(Y_label-Y_pred),axis=0),file=log)
			print('MSE: ',np.mean((Y_label-Y_pred)**2,axis=0),file=log)


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
					delta = (actual_time - t)#.detach()
					t = torch.tensor([t]).float().to(x.device).view(1,1)
					t.requires_grad_(True)
					# The last two features in the housing dataset are time, so it makes sense to pass these while visualizing
					y_pred = self.classifier(x,t) 
					partial_Y_pred_t = torch.autograd.grad(y_pred, t, grad_outputs=torch.ones_like(y_pred))[0]
					y_.append(y_pred.item())
					y__.append(partial_Y_pred_t.item())
					# print((partial_Y_pred_t*delta + y_pred))
					# y___.append((partial_Y_pred_t*delta + y_pred).cpu().item())
					# TODO gradient addition business
				ax[i,j].plot(x_,y_)
				ax[i,j].plot(x_,y__)
				# ax[i,j].plot(x_,y___)
				ax[i,j].set_title("time-{}".format(actual_time))

				# print(x_,y_)
				ax[i,j].scatter(u.view(-1,1).detach().cpu().numpy(),y.view(-1,1).detach().cpu().numpy())
		plt.savefig('{}.png'.format(filename))
		plt.close()

	def train(self):

		# if os.path.exists("classifier_{}_{}.pth".format(self.data,self.seed)):
		try:
			self.classifier.load_state_dict(torch.load("classifier_{}_{}.pth".format(self.data,self.seed)))
			print("Loading Model")
		except Exception as e:
			print(e)	
			self.train_classifier(encoder=self.encoder)  # Train classifier initially
			torch.save(self.classifier.state_dict(), "classifier_{}_{}.pth".format(self.data,self.seed))
		# print("Loading")
		#       
		# vis_ind = np.array(self.cumulative_data_indices[-1])
		# np.random.shuffle(vis_ind)
		# vis_ind = [self.source_data_indices[0][3], self.source_data_indices[1][47], self.source_data_indices[2][102], self.source_data_indices[2][210], self.source_data_indices[3][168], self.source_data_indices[3][342], self.source_data_indices[4][42], self.source_data_indices[4][44],self.source_data_indices[4][189]]
		# self.visualize_trajectory(vis_ind[:9],"plots/{}_{}_base".format(self.seed,self.delta))
		log = open("testing_{}_{}.txt".format(self.model,self.data),"a")
		print("#####################################",file=self.log)
		print("Delta - {}".format(self.delta),file=self.log)
		print("trelu - {} single_trelu - {}".format(self.trelu_limit,self.single_trelu),file=self.log)
		print("Performance of the base classifier",file=self.log)
		self.eval_classifier(log=self.log)
		self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(),self.lr)
		if self.model in ["GI","t_GI","goodfellow","t_goodfellow","grad_reg","grad_reg_curvature", "fixed_GI", "GI_t_delta", "GI_v3", "GI_num_int"]:
			self.finetune_grad_int(num_domains=self.num_finetune_domains)
			print("Performance after fine-tuning",file=self.log)
			self.eval_classifier(log=self.log,ensemble=self.ensemble)
			# self.visualize_trajectory(vis_ind[:9],"plots/{}_{}_{}".format(self.seed,self.delta,self.goodfellow))
		
		# print("-----------------------------------------",file=log)
