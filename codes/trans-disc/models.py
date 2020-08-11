import tensorflow as tf
import numpy as np

def transformer_model(data_shape, latent_shape):

	model = tf.keras.Sequential()

	model.add(tf.keras.layers.Dense(latent_shape, input_shape=(data_shape,), activation='relu'))
	model.add(tf.keras.layers.Dense(data_shape-2))
	model.add(tf.keras.layers.LeakyReLU())

	return model

def discriminator_model(data_shape, hidden_shape):

	model = tf.keras.Sequential()

	model.add(tf.keras.layers.Dense(hidden_shape, input_shape=(data_shape,), activation='relu'))
	model.add(tf.keras.layers.Dense(1))

	return model

def classification_model(data_shape, hidden_shape, out_shape):

	model = tf.keras.Sequential()

	model.add(tf.keras.layers.Dense(hidden_shape, input_shape=(data_shape,), activation='relu'))
	model.add(tf.keras.layers.Dense(out_shape, activation='relu'))
	model.add(tf.keras.layers.Softmax())
	
	return model

def reconstruction_loss(data_true, data_predicted):

	return tf.reduce_sum((data_true-data_predicted)**2, axis=1)

def discriminator_loss(real_output, trans_output):

	bxe = tf.keras.losses.BinaryCrossentropy(from_logits=True)

	real_loss = bxe(tf.ones_like(real_output), real_output)
	trans_loss = bxe(tf.zeros_like(trans_output), trans_output)
	total_loss = real_loss + trans_loss
	
	return total_loss

def transformer_loss(trans_output):

	bxe = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
	return bxe(tf.ones_like(trans_output), trans_output)

def discounted_transformer_loss(real_data, trans_data, trans_output):

	time_diff = tf.exp(-(real_data[:,-1] - real_data[:,-2]))
	re_loss = reconstruction_loss(real_data[:,0:-2], trans_data)
	tr_loss = transformer_loss(trans_output)

	loss = tf.reduce_sum(time_diff * tr_loss + (1-time_diff) * re_loss)
	return loss

def classification_loss(y_true, y_pred):

	xe = tf.keras.losses.CategoricalCrossentropy()
	return xe(y_true, y_pred)