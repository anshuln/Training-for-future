import numpy as np
import tensorflow as tf

IP_SHAPE = 10
ENC_SHAPE_1 = 16
ENC_SHAPE_2 = 8
OUT_SHAPE = 1

def encoder_model():

	model = tf.keras.Sequential()

	model.add(tf.keras.layers.Dense(ENC_SHAPE_1, input_shape=(IP_SHAPE+1,), activation='relu'))
	model.add(tf.keras.layers.Dense(ENC_SHAPE_2))

	return model

def classification_model():

	model = tf.keras.Sequential()

	model.add(tf.keras.layers.Dense(OUT_SHAPE, input_shape=(ENC_SHAPE_2,), activation='relu'))
	model.add(tf.keras.layers.Softmax())

	return model

def regression_model():

	model = tf.keras.Sequential()

	model.add(tf.keras.layers.Dense(ENC_SHAPE_2/2, input_shape=(ENC_SHAPE_2,), activation='relu'))
	model.add(tf.keras.layers.Dense(OUT_SHAPE, activation='relu'))

	return model

def discriminator_mean_model():

	model = tf.keras.Sequential()

	model.add(tf.keras.layers.Dense(ENC_SHAPE_2/2, input_shape=(ENC_SHAPE_2,), activation='relu'))
	model.add(tf.keras.layers.Dense(1))

	return model

def discriminator_var_model():

	model = tf.keras.Sequential()

	model.add(tf.keras.layers.Dense(1, input_shape=(ENC_SHAPE_2,), activation='relu'))

	return model	

def classification_loss(y_pred, y_true):

	xe = tf.keras.losses.CategoricalCrossentropy()
	return xe(y_true, y_pred)

def discriminator_loss(u_pred_mean, u_pred_var, u_true):

	l_mu = (u_true - u_pred_mean)**2
	sig = tf.add(u_pred_var, tf.constant(1e-4))

	loss = 0.5 * tf.add(tf.divide(l_mu, sig), tf.math.log(sig))
	return tf.reduce_mean(loss)


def regression_loss(y_pred, y_true):

	return tf.reduce_mean((y_true - y_pred)**2)


