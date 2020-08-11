import numpy as np
import pandas as pd
import argparse
import math
import os
import matplotlib.pyplot as plt
from transport import *
from sklearn.datasets import make_classification, make_moons
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pcida_model import *

EPOCH = 20
SUBEPOCH = 10
BATCH_SIZE = 32
DISC_BATCH_SIZE=64

def cida_main_classification(X_data, Y_data, U_data, num_indices, source_indices, target_indices):

	## Define and train classification model
	## Refactor this later

	# X_data, Y_data, U_data: list of data, labels, and domain info from all domains
	# source_indices: list of source indices
	# target_indices: list of target indices


	X_source = X_data[source_indices]
	Y_source = Y_data[source_indices]
	U_source = U_data[source_indices]

	X_target = X_data[target_indices]
	Y_target = Y_data[target_indices]
	U_target = U_data[target_indices]


	X_source = np.vstack(X_source)
	Y_source = np.vstack(Y_source)
	U_source = np.hstack(U_source)
	X_target = np.vstack(X_target)
	Y_target = np.vstack(Y_target)
	U_target = np.hstack(U_target)


	print(X_source.shape)
	print(Y_source.shape)
	print(U_source.shape)

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

	source_dataset = tf.data.Dataset.from_tensor_slices((X_source, Y_source, U_source)).batch(BATCH_SIZE)
	all_dataset = tf.data.Dataset.from_tensor_slices((X_all, Y_all, U_all)).batch(DISC_BATCH_SIZE)

	encoder = encoder_model()
	predictor = classification_model()
	discriminator_mean = discriminator_mean_model()
	discriminator_var = discriminator_var_model()

	encoder_optimizer = tf.keras.optimizers.Adam(1e-4)
	predictor_optimizer = tf.keras.optimizers.Adam(1e-4)
	discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
	loss = 0

	@tf.function
	def train_discriminator(X, U):

		with tf.GradientTape() as dics_tape_mu, tf.GradientTape() as dics_tape_sigma:

			E = encoder(X)				
			U_pred_mean = discriminator_mean(E)
			U_pred_var = discriminator_var(E)
			
			disc_loss = discriminator_loss(U_pred_mean, U_pred_var, U)

		gradients_of_discriminator_mu = dics_tape_mu.gradient(disc_loss, discriminator_mean.trainable_variables)
		gradients_of_discriminator_sigma = dics_tape_sigma.gradient(disc_loss, discriminator_var.trainable_variables)

		discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator_mu, discriminator_mean.trainable_variables))
		discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator_sigma, discriminator_var.trainable_variables))

		return disc_loss


	@tf.function
	def train_classifier(X, U, Y, l):

		with tf.GradientTape() as enc_tape, tf.GradientTape() as pred_tape, tf.GradientTape() as disc_tape:

			E = encoder(X)
			Y_pred = predictor(E)
			U_pred_mean = discriminator_mean(E)
			U_pred_var = discriminator_var(E)
			
			pred_loss = classification_loss(Y_pred, Y)
			disc_loss = discriminator_loss(U_pred_mean, U_pred_var, U)


		gradients_of_encoder_1 = enc_tape.gradient(pred_loss, encoder.trainable_variables)
		gradients_of_predictor = pred_tape.gradient(pred_loss, predictor.trainable_variables)
		gradients_of_encoder_2 = disc_tape.gradient(disc_loss, encoder.trainable_variables)

		gradients_of_encoder = []
		
		for j in range(len(gradients_of_encoder_1)):

			gradients_of_encoder.append(gradients_of_encoder_1[j] - l * gradients_of_encoder_2[j])
			
		#tf.print(gradients_of_encoder_1)
		
		encoder_optimizer.apply_gradients(zip(gradients_of_encoder, encoder.trainable_variables))
		predictor_optimizer.apply_gradients(zip(gradients_of_predictor, predictor.trainable_variables))	

		return pred_loss


	''' Training Loop'''
	for epoch in range(EPOCH):
		
		print ('Epoch %d - Discriminator' % epoch)

		for subepoch in range(SUBEPOCH):

			loss = 0
			
			for data in all_dataset:
				
				batch_X, batch_Y, batch_U = data

				batch_X = tf.cast(batch_X, dtype=tf.float32)
				batch_Y = tf.cast(batch_Y, dtype=tf.float32)
				batch_U = tf.cast(batch_U, dtype=tf.float32)

				batch_U = tf.reshape(batch_U, [-1,1])
				batch_X = tf.concat([batch_X, batch_U], axis=1)

				loss += train_discriminator(batch_X, batch_U)

			print('subepoch: %d, Loss: %f' % (subepoch, loss))
			
		''' Predictor Training '''

		print ('Epoch %d - Predictor' % epoch)

		for subepoch in range(SUBEPOCH):

			loss = 0

			for data in source_dataset:
			
				batch_X, batch_Y, batch_U = data

				batch_X = tf.cast(batch_X, dtype=tf.float32)
				batch_Y = tf.cast(batch_Y, dtype=tf.float32)
				batch_U = tf.cast(batch_U, dtype=tf.float32)

				batch_U = tf.reshape(batch_U, [-1,1])
				batch_X = tf.concat([batch_X, batch_U], axis=1)

				lambd = tf.constant(0, dtype=tf.float32)
				#lambd = tf.constant(epoch**2/EPOCH**2, dtype=tf.float32)
				
				loss += train_classifier(batch_X, batch_U, batch_Y, lambd)



			print('subepoch: %d, Loss: %f' % (subepoch, loss))


	X_target_merged = np.hstack([X_target, U_target.reshape(-1,1)])
	y_pred = predictor(encoder(X_target_merged))
	y_pred = np.array([0 if y[0] > y[1] else 1 for y in y_pred])
	Y_target = np.array([0 if y[0] > y[1] else 1 for y in Y_target])
	print(y_pred)
	print(accuracy_score(Y_target, y_pred))
	print(confusion_matrix(Y_target, y_pred))
	print(classification_report(Y_target, y_pred))

def cida_main_regression(X_data, Y_data, U_data, num_indices, source_indices, target_indices):

	## Define and train classification model
	## Refactor this later

	I_d = np.eye(num_indices)

	X_source = X_data[source_indices]
	Y_source = Y_data[source_indices]
	U_source = U_data[source_indices]

	X_target = X_data[target_indices]
	Y_target = Y_data[target_indices]
	U_target = U_data[target_indices]


	X_source = np.vstack(X_source)
	Y_source = np.hstack(Y_source)
	U_source = np.hstack(U_source)
	X_target = np.vstack(X_target)
	Y_target = np.hstack(Y_target)
	U_target = np.hstack(U_target)


	print(X_source.shape)
	print(Y_source.shape)
	print(U_source.shape)

	X_all = np.vstack([X_source, X_target])
	Y_all = np.hstack([Y_source, Y_target])
	U_all = np.hstack([U_source, U_target])
	
	perm = np.random.permutation(X_source.shape[0])
	X_source = X_source[perm]
	Y_source = Y_source[perm]
	U_source = U_source[perm]

	perm = np.random.permutation(X_all.shape[0])
	X_all = X_all[perm]
	Y_all = Y_all[perm]
	U_all = U_all[perm]

	source_dataset = tf.data.Dataset.from_tensor_slices((X_source, Y_source, U_source)).batch(BATCH_SIZE)
	all_dataset = tf.data.Dataset.from_tensor_slices((X_all, Y_all, U_all)).batch(DISC_BATCH_SIZE)

	encoder = encoder_model()
	predictor = regression_model()
	discriminator_mean = discriminator_mean_model()
	discriminator_var = discriminator_var_model()

	#encoder = cida.encoder_model(ip_shape=X_source.shape[1]+1, dims=2, dim_list=[8,4])
	#predictor = cida.predictor_model(ip_shape=4, out_shape=2)
	#discriminator = cida.discriminator_model(ip_shape=4)


	encoder_optimizer = tf.keras.optimizers.Adagrad(5e-2)
	predictor_optimizer = tf.keras.optimizers.Adagrad(5e-2)
	discriminator_optimizer = tf.keras.optimizers.Adagrad(1e-2)
	loss = 0

	#checkpoint_dir = './training_checkpoints'
	#checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	#checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
	                                 #discriminator_optimizer=discriminator_optimizer,
	                                 #generator=generator,
	                                 #discriminator=discriminator)

	@tf.function
	def train_discriminator(X, U):

		with tf.GradientTape() as dics_tape_mu, tf.GradientTape() as dics_tape_sigma:

			E = encoder(X)				
			U_pred_mean = discriminator_mean(E)
			U_pred_var = discriminator_var(E)
			
			disc_loss = discriminator_loss(U_pred_mean, U_pred_var, U)

		gradients_of_discriminator_mu = dics_tape_mu.gradient(disc_loss, discriminator_mean.trainable_variables)
		gradients_of_discriminator_sigma = dics_tape_sigma.gradient(disc_loss, discriminator_var.trainable_variables)

		discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator_mu, discriminator_mean.trainable_variables))
		discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator_sigma, discriminator_var.trainable_variables))

		return disc_loss


	@tf.function
	def train_regressor(X, U, Y, l):

		with tf.GradientTape() as enc_tape, tf.GradientTape() as pred_tape, tf.GradientTape() as disc_tape:

			E = encoder(X)
			Y_pred = predictor(E)
			U_pred_mean = discriminator_mean(E)
			U_pred_var = discriminator_var(E)

			pred_loss = regression_loss(Y_pred, Y)
			disc_loss = discriminator_loss(U_pred_mean, U_pred_var, U)


		gradients_of_encoder_1 = enc_tape.gradient(pred_loss, encoder.trainable_variables)
		gradients_of_predictor = pred_tape.gradient(pred_loss, predictor.trainable_variables)
		gradients_of_encoder_2 = disc_tape.gradient(disc_loss, encoder.trainable_variables)

		gradients_of_encoder = []
		
		for j in range(len(gradients_of_encoder_1)):

			gradients_of_encoder.append(gradients_of_encoder_1[j] - l * gradients_of_encoder_2[j])
			
		#tf.print(gradients_of_encoder_1)
		
		encoder_optimizer.apply_gradients(zip(gradients_of_encoder, encoder.trainable_variables))
		predictor_optimizer.apply_gradients(zip(gradients_of_predictor, predictor.trainable_variables))	

		return pred_loss


	''' Training Loop'''
	for epoch in range(EPOCH):
		
		print ('Epoch %d - Discriminator' % epoch)
		
		for subepoch in range(SUBEPOCH):

			loss = 0
			
			for data in all_dataset:
				
				batch_X, batch_Y, batch_U = data

				batch_X = tf.cast(batch_X, dtype=tf.float32)
				batch_Y = tf.cast(batch_Y, dtype=tf.float32)
				batch_U = tf.cast(batch_U, dtype=tf.float32)

				batch_U = tf.reshape(batch_U, [-1,1])
				batch_X = tf.concat([batch_X, batch_U], axis=1)

				loss += train_discriminator(batch_X, batch_U)

			print('subepoch: %d, Loss: %f' % (subepoch, loss))
		
		''' Predictor Training '''

		print ('Epoch %d - Predictor' % epoch)

		for subepoch in range(SUBEPOCH):

			loss = 0
			for data in source_dataset:
			
				batch_X, batch_Y, batch_U = data

				batch_X = tf.cast(batch_X, dtype=tf.float32)
				batch_Y = tf.cast(batch_Y, dtype=tf.float32)
				batch_U = tf.cast(batch_U, dtype=tf.float32)

				batch_U = tf.reshape(batch_U, [-1,1])
				batch_Y = tf.reshape(batch_Y, [-1,1])
				batch_X = tf.concat([batch_X, batch_U], axis=1)

				#lambd = tf.constant(epoch**2/EPOCH**2, dtype=tf.float32)
				lambd = tf.constant(0.0, dtype=tf.float32)
				
				loss += train_regressor(batch_X, batch_U, batch_Y, lambd)



			print('subepoch: %d, Loss: %f' % (subepoch, loss))


	X_target_merged = np.hstack([X_target, U_target.reshape(-1,1)])
	y_pred = predictor(encoder(X_target_merged))
	y_pred = y_pred.numpy()
	
	print(y_pred.shape)
	print(np.mean(abs(Y_target - y_pred)))
	print(np.max(abs(Y_target - y_pred)))
	print(np.min(abs(Y_target - y_pred)))
	print(np.std(abs(Y_target - y_pred)))
	
def ot_classifier(X_data, Y_data, source, target):

	print(X_data.shape)
	print(Y_data.shape)
	

	X_data = transform_samples_iter_reg(X_data, Y_data, 9)

	print(X_data.shape)
	print(Y_data.shape)
	
	X_source = X_data[source]
	Y_source = Y_data[source]
	X_target = X_data[target]
	Y_target = Y_data[target]

	X_source = np.vstack([X_source])
	Y_source = np.vstack([Y_source])

	X_target = np.vstack(X_target)
	Y_target = np.vstack(Y_target)


	source_dataset = tf.data.Dataset.from_tensor_slices((X_source, Y_source)).batch(BATCH_SIZE)

	encoder = encoder_model()
	predictor = regression_model()

	encoder_optimizer = tf.keras.optimizers.Adagrad(5e-2)
	predictor_optimizer = tf.keras.optimizers.Adagrad(5e-2)
	loss = 0

	@tf.function
	def train(X, Y):

		with tf.GradientTape() as enc_tape, tf.GradientTape() as pred_tape:

			E = encoder(X)
			Y_pred = predictor(E)

			pred_loss = regression_loss(Y_pred, Y)


		gradients_of_encoder = enc_tape.gradient(pred_loss, encoder.trainable_variables)
		gradients_of_predictor = pred_tape.gradient(pred_loss, predictor.trainable_variables)
		
		encoder_optimizer.apply_gradients(zip(gradients_of_encoder, encoder.trainable_variables))
		predictor_optimizer.apply_gradients(zip(gradients_of_predictor, predictor.trainable_variables))	

		return pred_loss

	for epoch in range(EPOCH):

		loss = 0
		for data in source_dataset:
		
			batch_X, batch_Y = data

			batch_X = tf.cast(batch_X, dtype=tf.float32)
			batch_Y = tf.cast(batch_Y, dtype=tf.float32)
			batch_Y = tf.reshape(batch_Y, [-1,1])
			
			loss += train(batch_X, batch_Y)

		print('Epoch: %d, Loss: %f' % (epoch, loss))

	X_target_merged = np.hstack([X_target, U_target.reshape(-1,1)])
	y_pred = predictor(encoder(X_target_merged))
	y_pred = np.array([0 if y[0] > y[1] else 1 for y in y_pred])
	Y_target = np.array([0 if y[0] > y[1] else 1 for y in Y_target])
	print(y_pred)
	print(accuracy_score(Y_target, y_pred))
	print(confusion_matrix(Y_target, y_pred))
	print(classification_report(Y_target, y_pred))


def load_twitter(files_list):

	domains = 5
	X_data, Y_data, U_data = [], [], []	

	for i, file in enumerate(files_list):

		df = pd.read_csv(file)
		Y_temp = df['y'].values
		X_temp = df.drop(['y', 'time', 'mean'], axis=1).values
		U_temp = np.array([i] * X_temp.shape[0])

		X_data.append(X_temp)
		Y_data.append(Y_temp)
		U_data.append(U_temp)

	return np.array(X_data), np.array(Y_data), np.array(U_data)

def load_news():

	domains = 5

	X_data, Y_data, U_data = [], [], []
	files_list = ['news_0.csv', 'news_1.csv', 'news_2.csv', 'news_3.csv', 'news_4.csv']
	
	# Assuming the last to be target, and first all to be source
	
	for i, file in enumerate(files_list):

		df = pd.read_csv(file)
		Y_temp = df[' shares'].values
		X_temp = df.drop([' timedelta', ' shares'], axis=1).values
		Y_temp = np.array([0 if d<=1400 else 1 for d in Y_temp])
		Y_temp = np.eye(2)[Y_temp]
		U_temp = np.array([i] * X_temp.shape[0])

		X_data.append(X_temp)
		Y_data.append(Y_temp)
		U_data.append(U_temp)


	return np.array(X_data), np.array(Y_data), np.array(U_data)

def load_moons(domains):

	X_data, Y_data, U_data = [], [], []
	for i in range(domains):

		angle = i*math.pi/(domains-1);
		X, Y = make_moons(n_samples=200, noise=0.1)
		rot = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
		X = np.matmul(X, rot)
		
		#plt.scatter(X[:,0], X[:,1], c=Y)
		#plt.savefig('moon_%d' % i)
		#plt.clf()

		Y = np.eye(2)[Y]
		U = np.array([i] * 200)

		X_data.append(X)
		Y_data.append(Y)
		U_data.append(U)

	return np.array(X_data), np.array(Y_data), np.array(U_data)


def main():

	#X_data, Y_data, U_data = load_twitter(['Delhi_%d.csv' % i for i in range(5)])
	#X_data, Y_data, U_data = load_news()
	#X_data, Y_data, U_data = load_twitter(['Fight_%d.csv' % i for i in range(5)])

	classification(X_data, Y_data, U_data, 11, [0,1,2,3,4,5,6,7,8], [9,10])
	#cida_main_regression(X_data, Y_data, U_data, 5, [3], [4])
	#cida_main_classification(X_data, Y_data, U_data, 11, list(range(7,9)), [9,10])
	#cida_main_classification(X_data, Y_data, U_data, 5, list(range(0,3)), [3, 4])
	#_dummy_train(X_data, Y_data, U_data, 11, list(range(0,9)), [9,10])
	

if __name__ == "__main__":

	main()
