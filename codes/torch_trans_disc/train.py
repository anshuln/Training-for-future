import numpy as np
import pandas as pd
import argparse
import math
import os
import matplotlib.pyplot as plt
#from transport import *
from sklearn.datasets import make_classification, make_moons
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from models import *
from data_loaders import *


def train_transformer_batch(X,transformer,discriminator,transformer_optimizer):

	X_pred = transformer(X)
	domain_info = X[:,-1].view(-1,1)
	X_pred_domain_info = torch.cat([X_pred, domain_info], dim=1)

	is_real = discriminator(X_pred_domain_info)

	trans_loss = discounted_transformer_loss(X, X_pred, is_real)

	# gradients_of_transformer = trans_tape.gradient(trans_loss, transformer.trainable_variables)
	trans_loss.backward()

	transformer_optimizer.step()

	return trans_loss

def train_discriminator_batch(X_old, X_now, transformer, discriminator, discriminator_optimizer):


	X_pred_old = transformer(X_old)
	domain_info = X_old[:,-1].view(-1,1)
	X_pred_old_domain_info = torch.cat([X_pred_old, domain_info], dim=1)

	is_real_old = discriminator(X_pred_old_domain_info)
	is_real_now = discriminator(X_now[:,0:-1])
	
	disc_loss = discriminator_loss(is_real_now, is_real_old)

	disc_loss.backward()
	discriminator_optimizer.step()

	return disc_loss


def train_classifier(X, Y, classifier, transformer, classifier_optimizer):


	X_pred = transformer(X)
	domain_info = tf.reshape(X[:,-2], [-1,1])
	X_pred_domain_info = tf.cat([X_pred, domain_info], axis=1)
	Y_pred = classifier(X_pred_domain_info)
	
	pred_loss = classification_loss(Y_pred, Y)

	pred_loss.backward()
	classifier_optimizer.step()
	

	return pred_loss


def train_classifier_d(X, Y, classifier, classifier_optimizer):

	Y_pred = classifier(X[:,0:-1])
	
	pred_loss = classification_loss(Y_pred, Y)

	pred_loss.backward()
	classifier_optimizer.step()

	return pred_loss


EPOCH = 200
SUBEPOCH = 10
BATCH_SIZE = 32
DISC_BATCH_SIZE=64
SHUFFLE_BUFFER_SIZE=4096

def train(X_data, Y_data, U_data, num_indices, source_indices, target_indices):

	I_d = np.eye(num_indices)

	X_source = X_data[source_indices]
	Y_source = Y_data[source_indices]
	U_source = U_data[source_indices]

	X_target = X_data[target_indices]
	Y_target = Y_data[target_indices]
	U_target = U_data[target_indices]


	transformer = Transformer(4, 3)
	discriminator = Discriminator(3, 3)
	classifier = ClassifyNet(3,3,2)

	transformer_optimizer   = torch.optim.Adagrad(transformer.parameters(),5e-2)
	classifier_optimizer    = torch.optim.Adagrad(discriminator.parameters(),5e-2)
	discriminator_optimizer = torch.optim.Adagrad(classifier.parameters(),5e-2)

	X_past = X_source[0]
	U_past = U_source[0]
	Y_past = Y_source[0]
	
	for index in range(1, len(X_source)):

		print('Domain %d' %index)
		print('----------------------------------------------------------------------------------------------')

		past_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_past).float(),
                                                     torch.tensor(U_past).float(), torch.tensor(Y_past).float()),BATCH_SIZE,True)
		# present_dataset = torch.utils.data.Dataloader(torch.utils.data.TensorDataset(X_source[index], U_source[index], 
		# 					Y_source[index]),BATCH_SIZE,True,repeat(
		# 					math.ceil(X_past.shape[0]/X_source[index].shape[0])))

		X_past = np.vstack([X_past, X_source[index]])
		Y_past = np.vstack([Y_past, Y_source[index]])
		U_past = np.hstack([U_past, U_source[index]])
		
		all_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_past).float(), torch.tensor(U_past).float(), torch.tensor(Y_past).float()),BATCH_SIZE,True)

		for epoch in range(EPOCH):

			loss1, loss2 = 0,0
			
			for batch_X, batch_U, batch_Y in all_dataset:

				# batch_X = tf.cast(batch_X, dtype=tf.float32)
				# batch_Y = tf.cast(batch_Y, dtype=tf.float32)
				# batch_U = tf.cast(batch_U, dtype=tf.float32)

				batch_U = batch_U.view(-1,1)
				this_U = torch.tensor([U_source[index][0]]*batch_U.shape[0]).float()
				this_U = this_U.view(-1,1)
				batch_X = torch.cat([batch_X, batch_U, this_U], dim=1)
				
				loss1 += train_transformer_batch(batch_X,transformer,discriminator,transformer_optimizer) #train_transformer_batch(batch_X)


			for batch_X, batch_U, batch_Y in past_dataset:

				# batch_X = tf.cast(batch_X, dtype=tf.float32)
				# batch_Y = tf.cast(batch_Y, dtype=tf.float32)
				# batch_U = tf.cast(batch_U, dtype=tf.float32)

				# batch_U = tf.reshape(batch_U, [-1,1])
				# this_U = tf.constant([U_source[index][0]]*batch_U.shape[0], dtype=tf.float32)
				# this_U = tf.reshape(this_U, [-1,1])
				# batch_X = tf.cat([batch_X, batch_U, this_U], axis=1)

				batch_U = batch_U.view(-1,1)
				this_U = torch.tensor([U_source[index][0]]*batch_U.shape[0])
				this_U = this_U.view(-1,1).float()
				batch_X = torch.cat([batch_X, batch_U, this_U], dim=1)
				# Do this in a better way

				indices = np.random.random_integers(0, X_source[index].shape[0]-1, batch_X.shape[0])

				# Better to shift this to the dataloader
				real_X = np.hstack([X_source[index][indices], U_source[index][indices].reshape(-1,1), 
							U_source[index][indices].reshape(-1,1)])

				real_X = torch.tensor(real_X).float()
				loss2 += train_discriminator_batch(batch_X, real_X, transformer, discriminator, discriminator_optimizer) #train_discriminator_batch(batch_X, real_X)

			print('Epoch %d - %f, %f' % (epoch, loss1.detach().cpu().numpy(), loss2.detach().cpu().numpy()))

	
	for i in range(len(X_target)):

		target_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_target[i]).float(), torch.tensor(U_target[i]).float(), torch.tensor(Y_target[i]).float()),BATCH_SIZE,True)
		source_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_past).float(), torch.tensor(U_past).float(), torch.tensor(Y_past).float()),BATCH_SIZE,True)

		for epoch in range(EPOCH):

			loss = 0
			
			for batch_X, batch_U, batch_Y in source_dataset:


				# batch_X = tf.cast(batch_X, dtype=tf.float32)
				# batch_Y = tf.cast(batch_Y, dtype=tf.float32)
				# batch_U = tf.cast(batch_U, dtype=tf.float32)

				# batch_U = tf.reshape(batch_U, [-1,1])
				# this_U = tf.constant([U_target[i][0]]*batch_U.shape[0], dtype=tf.float32)
				# this_U = tf.reshape(this_U, [-1,1])
				# batch_X = tf.cat([batch_X, batch_U, this_U], axis=1)
				batch_U = batch_U.view(-1,1)
				this_U = torch.tensor([U_source[index][0]]*batch_U.shape[0]).float()
				this_U = this_U.view(-1,1)
				batch_X = torch.cat([batch_X, batch_U, this_U], dim=1)

				loss += train_classifier_d(batch_X, batch_Y, classifier, classifier_optimizer)

			print('Epoch: %d - Loss: %f' % (epoch, loss))

		
		#print(classifier.trainable_variables)
		Y_pred = []
		for batch_X, batch_U, batch_Y in target_dataset:

			# batch_X = tf.cast(batch_X, dtype=tf.float32)
			# batch_Y = tf.cast(batch_Y, dtype=tf.float32)
			# batch_U = tf.cast(batch_U, dtype=tf.float32)

			# batch_U = tf.reshape(batch_U, [-1,1])
			# this_U = tf.constant([U_target[i][0]]*batch_U.shape[0], dtype=tf.float32)
			# this_U = tf.reshape(this_U, [-1,1])
			# batch_X = tf.cat([batch_X, batch_U, this_U], axis=1)

			batch_U = batch_U.view(-1,1)
			this_U = torch.tensor([U_source[index][0]]*batch_U.shape[0]).float()
			this_U = this_U.view(-1,1)
			batch_X = torch.cat([batch_X, batch_U, this_U], dim=1)
			#batch_X_pred = transformer(batch_X)
			#domain_info = tf.reshape(batch_X[:,-2], [-1,1])
			#X_pred_domain_info = tf.cat([batch_X_pred, domain_info], axis=1)
			batch_Y_pred = classifier(batch_X[:,0:-1]).detach().cpu().numpy()

			Y_pred = Y_pred + [batch_Y_pred]

		Y_pred = np.vstack(Y_pred)
		print('shape: ',Y_pred.shape)
		print(Y_pred)
		Y_pred = np.array([0 if y[0] > y[1] else 1 for y in Y_pred])
		Y_true = np.array([0 if y[0] > y[1] else 1 for y in Y_target[i]])
		print(accuracy_score(Y_true, Y_pred))
		print(confusion_matrix(Y_true, Y_pred))
		print(classification_report(Y_true, Y_pred))	



if __name__ == "__main__":
	X_data, Y_data, U_data = load_moons(11)

	classification(X_data, Y_data, U_data, 11, [7,8], [9,10])
	