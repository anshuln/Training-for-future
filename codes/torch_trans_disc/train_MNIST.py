import numpy as np
import pandas as pd
import argparse
import math
import os
import matplotlib.pyplot as plt
#from transport import *
# from sklearn.datasets import make_classification, make_moons
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from torch.utils.tensorboard import SummaryWriter
import time 
from tqdm import tqdm


from model_MNIST import *
from data_loaders import *
# from regularized_ot import *
device = "cuda:2"

def train_transformer_batch(X,Y,source_u,dest_u,transformer,discriminator,classifier,transformer_optimizer,is_wasserstein=False):

	transformer_optimizer.zero_grad()
	X_pred = transformer(X,torch.cat([source_u,dest_u],dim=1))
	# domain_info = X[:,-1].view(-1,1)
	# X_pred_domain_info = torch.cat([X_pred, domain_info], dim=1)

	is_real = discriminator(X_pred,dest_u)

	# X_pred_class = torch.cat([X_pred,domain_info],dim=1)
	pred_class = classifier(X_pred, dest_u)

	trans_loss,ld,lr, lc = discounted_transformer_loss(X, X_pred,is_real, pred_class,Y,is_wasserstein)

	# gradients_of_transformer = trans_tape.gradient(trans_loss, transformer.trainable_variables)
	trans_loss.backward()

	transformer_optimizer.step()

	return trans_loss, ld, lr, lc


def train_discriminator_batch_wasserstein(X_old,source_u,dest_u, X_now, transformer, discriminator, discriminator_optimizer):
	
	discriminator_optimizer.zero_grad()
	X_pred = transformer(X_old,torch.cat([source_u.to(device),dest_u.to(device)],dim=1))
	
	# X_pred_old_domain_info = torch.cat([X_pred_old, domain_info], dim=1)

	is_real_old = discriminator(X_pred,dest_u)
	is_real_now = discriminator(X_now,source_u)
	
	disc_loss = discriminator_loss_wasserstein(is_real_now, is_real_old)

	disc_loss.backward()

	discriminator_optimizer.step()
	for p in discriminator.parameters():
		p.data.clamp_(-0.2, 0.2)
	return disc_loss

def train_discriminator_batch(X_old, X_now, transformer, discriminator, discriminator_optimizer):

	discriminator_optimizer.zero_grad()
	X_pred_old = transformer(X_old)
	domain_info = X_old[:,-1].view(-1,1)
	X_pred_old_domain_info = torch.cat([X_pred_old, domain_info], dim=1)

	is_real_old = discriminator(X_pred_old_domain_info)
	is_real_now = discriminator(X_now[:,0:-1])
	
	disc_loss = discriminator_loss(is_real_now, is_real_old)

	disc_loss.backward()
	discriminator_optimizer.step()

	return disc_loss


def train_classifier(X, source_u,dest_u, Y, classifier, transformer, classifier_optimizer):

	classifier_optimizer.zero_grad()
	# print(X.size(),source_u.size(),dest_u.size())
	X_pred = transformer(X,torch.cat([source_u.to(device),dest_u.to(device)],dim=1))
	# X_pred = transformer(X,U)
	# domain_info = U[:,-2:].view(-1,2)
	# X_pred_domain_info = torch.cat([X_pred, domain_info], dim=1)
	Y_pred = classifier(X_pred, dest_u)
	
	pred_loss = classification_loss(Y_pred, Y).mean()

	pred_loss.backward()
	classifier_optimizer.step()
	

	return pred_loss


def train_classifier_d(X, U, Y, classifier, classifier_optimizer,verbose=False):

	classifier_optimizer.zero_grad()
	Y_pred = classifier(X, U)
	pred_loss = classification_loss(Y_pred, Y).mean()
	# pred_loss = pred_loss.sum()
	pred_loss.backward()
	
	if verbose:
		# print(torch.cat([Y_pred, Y, Y*torch.log(Y_pred),
		# (Y*torch.log(Y_pred)).sum().unsqueeze(0).unsqueeze(0).repeat(BATCH_SIZE,1) ],dim=1))
		# for p in classifier.parameters():
		#     print(p.data)
		#     print(p.grad.data)
		#     print("____")
		# print(X.max())
		print(torch.cat([Y_pred.argmax(dim=1).view(-1,1).float(),Y.view(-1,1).float()],dim=1))
	classifier_optimizer.step()

	return pred_loss


EPOCH = 5
CLASSIFIER_EPOCHS = 20	
SUBEPOCH = 10
BATCH_SIZE = 128
DISC_BATCH_SIZE=64
SHUFFLE_BUFFER_SIZE=4096
IS_WASSERSTEIN = True
NUM_TRAIN_DOMAIN = 6
BIN_WIDTH = 30
def train(num_indices, source_indices, target_indices):

	I_d = np.eye(num_indices)



	transformer = Transformer(28**2 + 2*2, 256).to(device)
	discriminator = Discriminator(28**2 + 2, 256,IS_WASSERSTEIN).to(device)
	classifier = ClassifyNet(28**2 + 2,256,10).to(device)

	transformer_optimizer   = torch.optim.Adagrad(transformer.parameters(),5e-3)
	classifier_optimizer    = torch.optim.Adagrad(classifier.parameters(),5e-3)
	discriminator_optimizer = torch.optim.Adagrad(discriminator.parameters(),1e-3)

	# X_past = X_source[0]
	# U_past = U_source[0]
	# Y_past = Y_source[0]
	U_source = np.array(source_indices)
	U_target = np.array(target_indices)
	# writer = SummaryWriter(comment='{}'.format(time.time()))

	ot_maps = [[None for x in range(len(source_indices))] for y in range(len(source_indices))]

	# for i in range(len(source_indices)):
	#   for j in range(len(source_indices)):
	#       if i!=j:
	#           ot_sinkhorn = RegularizedSinkhornTransport(reg_e=0.5, alpha=10, max_iter=50, norm="median", verbose=False)
	#           ot_sinkhorn.fit(Xs=X_data[source_indices[i]]+1e-6, ys=Y_data[source_indices[i]]+1e-6, Xt=X_data[source_indices[j]]+1e-6, yt=Y_data[source_indices[j]]+1e-6, iteration=0)
	#           ot_maps[i][j] = ot_sinkhorn.transform(X_data[source_indices[i]]+1e-6)
	#       else:
	#           ot_maps[i][j] = X_data[source_indices[i]]
	# print(ot_maps)
	# # assert False
	# for class_index in range(1,len(X_source)):
	#   X_past = np.vstack([X_past, X_source[class_index]])
	#   Y_past = np.vstack([Y_past, Y_source[class_index]])
	#   U_past = np.hstack([U_past, U_source[class_index]])     
	
	# print(ot_maps)
	print("-------------TRAINING CLASSIFIER----------")
	class_step = 0
	mnist_ind = (np.arange(NUM_TRAIN_DOMAIN*10000))
	np.random.shuffle(mnist_ind)
	mnist_data = RotMNIST(indices=mnist_ind[:source_indices[-1]*10000],bin_width=BIN_WIDTH,bin_index=0,n_bins=NUM_TRAIN_DOMAIN)#len(source_indices))
	for epoch in range(CLASSIFIER_EPOCHS):
		past_dataset = torch.utils.data.DataLoader((mnist_data),BATCH_SIZE,True)
		class_loss = 0
		for batch_X, batch_U, batch_Y in tqdm(past_dataset):

			# batch_X = torch.cat([batch_X,batch_U.view(-1,2)],dim=1)

			l = train_classifier_d(batch_X,batch_U,batch_Y,classifier,classifier_optimizer,verbose=False)
			class_step += 1
			class_loss += l
		print("Epoch %d Loss %f"%(epoch,class_loss),flush=True)
	# print(batch_U)
	# # X_past = X_source[0]
	# # U_past = U_source[0]
	# # Y_past = Y_source[0]

	for index in range(1, len(source_indices)):

		print('Domain %d' %index)
		print('----------------------------------------------------------------------------------------------')

		past_data = (RotMNIST(indices=mnist_ind[:source_indices[index-1]*(60000//NUM_TRAIN_DOMAIN)],bin_width=BIN_WIDTH,bin_index=0,n_bins=6)) #,BATCH_SIZE,True)
		# present_dataset = torch.utils.data.Dataloader(torch.utils.data.TensorDataset(X_source[index], U_source[index], 
		#                   Y_source[index]),BATCH_SIZE,True,repeat(
		#                   math.ceil(X_past.shape[0]/X_source[index].shape[0])))

		num_past_batches = len(past_data) // BATCH_SIZE
		# X_past = np.vstack([X_past, X_source[index]])
		past_data = torch.utils.data.DataLoader(past_data,BATCH_SIZE,True)
		# Y_past = np.vstack([Y_past, Y_source[index]])
		# U_past = np.hstack([U_past, U_source[index]])
		
		p = RotMNIST(indices=mnist_ind[:source_indices[(index)]*(60000//NUM_TRAIN_DOMAIN)],bin_width=BIN_WIDTH,bin_index=0,n_bins=6)

		print(len(p))  # TODO
		all_data = torch.utils.data.DataLoader(p,
					BATCH_SIZE,True)            # for batch_X, batch_U, batch_Y, batch_transported in all_dataset:
		curr_data = torch.utils.data.DataLoader(RotMNIST(indices=mnist_ind[source_indices[index-1]*(60000//NUM_TRAIN_DOMAIN):source_indices[(index)]*(60000//NUM_TRAIN_DOMAIN)],bin_width=BIN_WIDTH,bin_index=index-1,n_bins=6),BATCH_SIZE,True,drop_last=True)
		num_all_batches  = len(p) // BATCH_SIZE
		all_steps_t = 0
		all_steps_d = 0
		step_c = 0

		for epoch in range(EPOCH):

			loss1, loss2 = 0,0
			step_t,step_d = 0,0

			all_dataset = iter(all_data)
			past_dataset = iter(past_data)
			curr_dataset = iter(curr_data)
			loop1 = True
			loop2 = True
			while (loop1 or loop2):
				if step_d < num_past_batches:
					batch_X, batch_U, batch_Y = next(past_dataset)
					batch_U = batch_U.view(-1,2)
					this_U = np.array([U_source[index]*BIN_WIDTH]*batch_U.shape[0]).reshape((batch_U.shape[0],1)) +\
							 np.random.randint(BIN_WIDTH,size=(batch_U.shape[0],1))
					this_U = np.hstack([np.array([U_source[index]]*batch_U.shape[0]).reshape((batch_U.shape[0],1)),
										this_U])
					this_U = torch.tensor(this_U).float().view(-1,2).to(device)
					# Do this in a better way


					# Better to shift this to the dataloader
					try:
						real_X,real_U,_ = next(curr_dataset)
					except:
						curr_dataset = iter(curr_data)
						real_X,real_U,_ = next(curr_dataset)
					if IS_WASSERSTEIN:
						loss_d = train_discriminator_batch_wasserstein(batch_X, batch_U,this_U,real_X, transformer, discriminator, discriminator_optimizer) #train_discriminator_batch(batch_X, real_X)
					else:
						loss_d = train_discriminator_batch(batch_X, batch_U,this_U,real_X, transformer, discriminator, discriminator_optimizer)
					loss2 += loss_d
					# writer.add_scalar('Loss/disc',loss_d.detach().numpy(),step_d+all_steps_d)
					step_d += 1
					loop2 = True
				else:
					loop2 = False
				if step_t < num_all_batches:
					batch_X, batch_U, batch_Y = next(all_dataset)
					batch_U = batch_U.view(-1,2)
					this_U = np.array([U_source[index]*BIN_WIDTH]*batch_U.shape[0]).reshape((batch_U.shape[0],1)) +\
							 np.random.randint(0,5,size=(batch_U.shape[0],1))
					this_U = np.hstack([np.array([U_source[index]]*batch_U.shape[0]).reshape((batch_U.shape[0],1)),
										this_U/(BIN_WIDTH * 6)])
					this_U = torch.tensor(this_U).float().view(-1,2).to(device)
					# print(batch_X.size(),batch_U.size(),this_U.size(),batch_transported.size())
					loss_t,ltd,lr,lc = train_transformer_batch(batch_X,batch_Y,batch_U,this_U,
									transformer,discriminator,classifier,
									transformer_optimizer,is_wasserstein=IS_WASSERSTEIN) #train_transformer_batch(batch_X)
					loss1 += loss_t
					# writer.add_scalar('Loss/transformer',loss_t.detach().numpy(),step_t+all_steps_t)
					# writer.add_scalar('Loss/transformer_disc',ltd.detach().numpy(),step_t+all_steps_t)
					# writer.add_scalar('Loss/transformer_rec',lr.detach().numpy(),step_t+all_steps_t)
					# writer.add_scalar('Loss/transformer_classifier',lc.detach().numpy(),step_t+all_steps_t)
					step_t += 1
					loop1 = True
				# for batch_X, batch_U, batch_Y in past_dataset:
				else:
					loop1 = False

			all_steps_d += step_d
			all_steps_t += step_t
			print('Epoch %d - %f, %f' % (epoch, loss1.detach().cpu().numpy(), loss2.detach().cpu().numpy()))

	
	print("___________TESTING____________")
	for i in range(len(target_indices)):
		print(U_target[i])

		source_dataset = torch.utils.data.DataLoader(RotMNIST(indices=mnist_ind[:source_indices[-1]*(60000//NUM_TRAIN_DOMAIN)],bin_width=BIN_WIDTH,bin_index=0,n_bins=6),BATCH_SIZE,True)



		step = 0
		for epoch in range(CLASSIFIER_EPOCHS):

			loss = 0
			
			for batch_X, batch_U, batch_Y in source_dataset:
				batch_U = batch_U.view(-1,2)
				this_U = np.array([U_target[i]*BIN_WIDTH]*batch_U.shape[0]).reshape((batch_U.shape[0],1)) +\
						 np.random.randint(0,5,size=(batch_U.shape[0],1))
				this_U = np.hstack([np.array([U_target[i]]*batch_U.shape[0]).reshape((batch_U.shape[0],1)),
									this_U/(BIN_WIDTH * 6)])
				this_U = torch.tensor(this_U).float().view(-1,2).to(device)
				# batch_X = torch.cat([batch_X, batch_U, this_U], dim=1)
				step += 1
				loss += train_classifier(batch_X,batch_U,this_U, batch_Y, classifier,transformer, classifier_optimizer)

			print('Epoch: %d - ClassificationLoss: %f' % (epoch, loss))

		
		#print(classifier.trainable_variables)
		index = target_indices[i]
		print(index)
		td = RotMNIST(indices=mnist_ind[(index-1)*(60000//NUM_TRAIN_DOMAIN):(index)*(60000//NUM_TRAIN_DOMAIN)],bin_width=BIN_WIDTH,bin_index=index-1,n_bins=6)
		print(len(mnist_ind[(index-1)*(60000//NUM_TRAIN_DOMAIN):(index)*(60000//NUM_TRAIN_DOMAIN)]))
		target_dataset = torch.utils.data.DataLoader(td,BATCH_SIZE,False,drop_last=True)
		Y_pred = []
		Y_label = []
		for batch_X, batch_U, batch_Y in tqdm(target_dataset):


			batch_U = batch_U.view(-1,2)
			batch_Y_pred = classifier(batch_X, batch_U).detach().cpu().numpy()
			Y_pred = Y_pred + [np.argmax(batch_Y_pred,axis=1)]
			Y_label = Y_label + [batch_Y.detach().cpu().numpy()]
		print(len(Y_pred),len(Y_label))
		# print(Y_pred[0].shape,Y_label[0].shape)
		Y_pred = np.hstack(Y_pred)
		Y_label = np.hstack(Y_label)
		print('shape: ',Y_pred.shape)
		print(accuracy_score(Y_label, Y_pred))
		print(confusion_matrix(Y_label, Y_pred))
		print(classification_report(Y_label, Y_pred))    
	return transformer,discriminator,classifier



def train_baselines(X_data,Y_data,U_data,num_indices, source_indices, target_indices):
	I_d = np.eye(num_indices)

	X_source = X_data[source_indices]
	Y_source = Y_data[source_indices]
	U_source = U_data[source_indices]

	X_target = X_data[target_indices]
	Y_target = Y_data[target_indices]
	U_target = U_data[target_indices]

	classifier = ClassifyNet(3,3,2)

	classifier_optimizer    = torch.optim.Adagrad(classifier.parameters(),5e-2)

	writer = SummaryWriter(comment='{}'.format(time.time()))

	## BASELINE 1- Sequential training with no adaptation ##
	for i in source_indices:
		X_past = X_source[0]
		U_past = U_source[0]
		Y_past = Y_source[0]
		
		past_data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_past).float(),
														torch.tensor(U_past).float(), torch.tensor(Y_past).float()),BATCH_SIZE,True)
		for epoch in range(EPOCH):
			loss = 0
			for batch_X,batch_U,batch_Y in past_data:
				batch_U = batch_U.view(-1,1)
				batch_X = torch.cat([batch_X, batch_U], dim=1)
				# step += 1
				loss += train_classifier_d(batch_X, batch_Y, classifier, classifier_optimizer, verbose=False)
			print('Epoch %d - %f' % (epoch, loss.detach().cpu().numpy()))

	print("___________TESTING____________")
	for i in range(len(X_target)):
		print(U_target[i])
		target_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_target[i]).float(), torch.tensor(U_target[i]).float(), torch.tensor(Y_target[i]).float()),BATCH_SIZE,False)
		Y_pred = []
		for batch_X, batch_U, batch_Y in target_dataset:

			batch_U = batch_U.view(-1,1)
			batch_X = torch.cat([batch_X, batch_U], dim=1)
			batch_Y_pred = classifier(batch_X).detach().cpu().numpy()

			Y_pred = Y_pred + [batch_Y_pred]  
		Y_pred = np.vstack(Y_pred)
		print('shape: ',Y_pred.shape)
		# print(Y_pred)
		Y_pred = np.array([0 if y[0] > y[1] else 1 for y in Y_pred])
		Y_true = np.array([0 if y[0] > y[1] else 1 for y in Y_target[i]])

		# print(Y_pred-Y_true)
		print(accuracy_score(Y_true, Y_pred))
		print(confusion_matrix(Y_true, Y_pred))
		print(classification_report(Y_true, Y_pred))    
	return trans,disc,classifier
 
	  
if __name__ == "__main__":
	# X_data, Y_data, U_data = load_moons(11)

	# train_baselines(X_data, Y_data, U_data, 11, [7,8], [9,10])
	t,d,c = train(3,[3,4],[5,6])
	torch.save({"trans":t.state_dict(),"disc":d.state_dict(),"classifier":c.state_dict()},"./model.pth")