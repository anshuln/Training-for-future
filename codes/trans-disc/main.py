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

EPOCH = 200
SUBEPOCH = 10
BATCH_SIZE = 32
DISC_BATCH_SIZE=64
SHUFFLE_BUFFER_SIZE=4096

def classification(X_data, Y_data, U_data, num_indices, source_indices, target_indices):

	# X_data is a list of num_indices length
	# Y_data is a list of num_indices length

	I_d = np.eye(num_indices)

	X_source = X_data[source_indices]
	Y_source = Y_data[source_indices]
	U_source = U_data[source_indices]

	X_target = X_data[target_indices]
	Y_target = Y_data[target_indices]
	U_target = U_data[target_indices]

	'''
	X_source = np.vstack(X_source)
	Y_source = np.vstack(Y_source)
	U_source = np.hstack(U_source)
	X_target = np.vstack(X_target)
	Y_target = np.vstack(Y_target)
	U_target = np.hstack(U_target)
	
	
	X_all = np.vstack([X_source, X_target])
	Y_all = np.vstack([Y_source, Y_target])
	U_all = np.hstack([U_source, U_target])
	
	perm = np.random.permutation(X_source.shape[0])
	X_source = X_source[perm]
	Y_source = Y_source[perm]
	U_source = U_source[perm]

	perm = np.random.permutation(X_all.shape[0])
	X_all = X_all[perm]
	Y_all = Y_all[perm]
	U_all = U_all[perm]

	'''	
	transformer = transformer_model(4, 3)
	discriminator = discriminator_model(3, 3)
	classifier = classification_model(3,3,2)

	transformer_optimizer = tf.keras.optimizers.Adagrad(5e-2)
	classifier_optimizer = tf.keras.optimizers.Adagrad(5e-2)
	discriminator_optimizer = tf.keras.optimizers.Adagrad(5e-2)
	
	@tf.function
	def train_transformer(X):

		with tf.GradientTape() as trans_tape:

			X_pred = transformer(X)
			domain_info = tf.reshape(X[:,-1], [-1,1])
			X_pred_domain_info = tf.concat([X_pred, domain_info], axis=1)

			is_real = discriminator(X_pred_domain_info)

			trans_loss = discounted_transformer_loss(X, X_pred, is_real)

		gradients_of_transformer = trans_tape.gradient(trans_loss, transformer.trainable_variables)

		transformer_optimizer.apply_gradients(zip(gradients_of_transformer, transformer.trainable_variables))

		return trans_loss

	
	
	@tf.function
	def train_discriminator(X_old, X_now):

		with tf.GradientTape() as disc_tape:

			X_pred_old = transformer(X_old)
			domain_info = tf.reshape(X_old[:,-1], [-1,1])
			X_pred_old_domain_info = tf.concat([X_pred_old, domain_info], axis=1)

			is_real_old = discriminator(X_pred_old_domain_info)
			is_real_now = discriminator(X_now[:,0:-1])
			
			disc_loss = discriminator_loss(is_real_now, is_real_old)

		gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

		discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

		return disc_loss


	@tf.function
	def train_classifier(X, Y):

		with tf.GradientTape() as pred_tape:

			X_pred = transformer(X)
			domain_info = tf.reshape(X[:,-2], [-1,1])
			X_pred_domain_info = tf.concat([X_pred, domain_info], axis=1)
			Y_pred = classifier(X_pred_domain_info)
			
			pred_loss = classification_loss(Y_pred, Y)

		gradients_of_classifier = pred_tape.gradient(pred_loss, classifier.trainable_variables)
		
		classifier_optimizer.apply_gradients(zip(gradients_of_classifier, classifier.trainable_variables))
		

		return pred_loss
	
	
	def train_classifier_d(X, Y):

		with tf.GradientTape() as pred_tape:

			Y_pred = classifier(X[:,0:-1])
			
			pred_loss = classification_loss(Y_pred, Y)

		gradients_of_classifier = pred_tape.gradient(pred_loss, classifier.trainable_variables)
		
		classifier_optimizer.apply_gradients(zip(gradients_of_classifier, classifier.trainable_variables))

		return pred_loss

	X_past = X_source[0]
	U_past = U_source[0]
	Y_past = Y_source[0]
	
	for index in range(1, len(X_source)):

		print('Domain %d' %index)
		print('----------------------------------------------------------------------------------------------')

		past_dataset = tf.data.Dataset.from_tensor_slices((X_past, U_past, Y_past)).batch(BATCH_SIZE).shuffle(SHUFFLE_BUFFER_SIZE)
		present_dataset = tf.data.Dataset.from_tensor_slices((X_source[index], U_source[index], 
							Y_source[index])).batch(BATCH_SIZE).shuffle(SHUFFLE_BUFFER_SIZE).repeat(
							math.ceil(X_past.shape[0]/X_source[index].shape[0]))

		X_past = np.vstack([X_past, X_source[index]])
		Y_past = np.vstack([Y_past, Y_source[index]])
		U_past = np.hstack([U_past, U_source[index]])
		
		all_dataset = tf.data.Dataset.from_tensor_slices((X_past, U_past, Y_past)).batch(BATCH_SIZE).shuffle(SHUFFLE_BUFFER_SIZE)

		for epoch in range(EPOCH):

			loss1, loss2 = 0,0
			
			for batch_X, batch_U, batch_Y in all_dataset:

				batch_X = tf.cast(batch_X, dtype=tf.float32)
				batch_Y = tf.cast(batch_Y, dtype=tf.float32)
				batch_U = tf.cast(batch_U, dtype=tf.float32)

				batch_U = tf.reshape(batch_U, [-1,1])
				this_U = tf.constant([U_source[index][0]]*batch_U.shape[0], dtype=tf.float32)
				this_U = tf.reshape(this_U, [-1,1])
				batch_X = tf.concat([batch_X, batch_U, this_U], axis=1)
				
				loss1 += train_transformer(batch_X)


			for batch_X, batch_U, batch_Y in past_dataset:

				batch_X = tf.cast(batch_X, dtype=tf.float32)
				batch_Y = tf.cast(batch_Y, dtype=tf.float32)
				batch_U = tf.cast(batch_U, dtype=tf.float32)

				batch_U = tf.reshape(batch_U, [-1,1])
				this_U = tf.constant([U_source[index][0]]*batch_U.shape[0], dtype=tf.float32)
				this_U = tf.reshape(this_U, [-1,1])
				batch_X = tf.concat([batch_X, batch_U, this_U], axis=1)

				# Do this in a better way

				indices = np.random.random_integers(0, X_source[index].shape[0]-1, batch_X.shape[0])
				real_X = np.hstack([X_source[index][indices], U_source[index][indices].reshape(-1,1), 
							U_source[index][indices].reshape(-1,1)])

				real_X = tf.constant(real_X)
				loss2 += train_discriminator(batch_X, real_X)

			print('Epoch %d - %f, %f' % (epoch, loss1.numpy(), loss2.numpy()))

	
	for i in range(len(X_target)):

		target_dataset = tf.data.Dataset.from_tensor_slices((X_target[i], U_target[i], Y_target[i])).batch(
							BATCH_SIZE).shuffle(SHUFFLE_BUFFER_SIZE)
		source_dataset = tf.data.Dataset.from_tensor_slices((X_past, U_past, Y_past)).batch(BATCH_SIZE).shuffle(
							SHUFFLE_BUFFER_SIZE)

		for epoch in range(EPOCH):

			loss = 0
			
			for batch_X, batch_U, batch_Y in source_dataset:


				batch_X = tf.cast(batch_X, dtype=tf.float32)
				batch_Y = tf.cast(batch_Y, dtype=tf.float32)
				batch_U = tf.cast(batch_U, dtype=tf.float32)

				batch_U = tf.reshape(batch_U, [-1,1])
				this_U = tf.constant([U_target[i][0]]*batch_U.shape[0], dtype=tf.float32)
				this_U = tf.reshape(this_U, [-1,1])
				batch_X = tf.concat([batch_X, batch_U, this_U], axis=1)

				loss += train_classifier_d(batch_X, batch_Y)

			print('Epoch: %d - Loss: %f' % (epoch, loss))

		
		#print(classifier.trainable_variables)
		Y_pred = []
		for batch_X, batch_U, batch_Y in target_dataset:

			batch_X = tf.cast(batch_X, dtype=tf.float32)
			batch_Y = tf.cast(batch_Y, dtype=tf.float32)
			batch_U = tf.cast(batch_U, dtype=tf.float32)

			batch_U = tf.reshape(batch_U, [-1,1])
			this_U = tf.constant([U_target[i][0]]*batch_U.shape[0], dtype=tf.float32)
			this_U = tf.reshape(this_U, [-1,1])
			batch_X = tf.concat([batch_X, batch_U, this_U], axis=1)

			#batch_X_pred = transformer(batch_X)
			#domain_info = tf.reshape(batch_X[:,-2], [-1,1])
			#X_pred_domain_info = tf.concat([batch_X_pred, domain_info], axis=1)
			batch_Y_pred = classifier(batch_X[:,0:-1]).numpy()

			Y_pred = Y_pred + [batch_Y_pred]

		Y_pred = np.vstack(Y_pred)
		print('shape: ',Y_pred.shape)
		print(Y_pred)
		Y_pred = np.array([0 if y[0] > y[1] else 1 for y in Y_pred])
		Y_true = np.array([0 if y[0] > y[1] else 1 for y in Y_target[i]])
		print(accuracy_score(Y_true, Y_pred))
		print(confusion_matrix(Y_true, Y_pred))
		print(classification_report(Y_true, Y_pred))



def main():

	#X_data, Y_data, U_data = load_twitter(['../Delhi_%d.csv' % i for i in range(5)])
	X_data, Y_data, U_data = load_moons(11)
	#X_data, Y_data, U_data = load_twitter(['../Fight_%d.csv' % i for i in range(5)])

	#ot_classifier(X_data, Y_data, [0,1,2,3,4,5,6,7,8], [9,10])
	classification(X_data, Y_data, U_data, 11, [7,8], [9,10])
	#cida_main_classification(X_data, Y_data, U_data, 11, list(range(7,9)), [9,10])
	#cida_main_classification(X_data, Y_data, U_data, 5, list(range(0,3)), [3, 4])
	#_dummy_train(X_data, Y_data, U_data, 11, list(range(0,9)), [9,10])
	

if __name__ == "__main__":

	main()
