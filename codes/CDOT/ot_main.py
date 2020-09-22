import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from torch.utils.tensorboard import SummaryWriter
import time 
import torch
from transport import *
from models import *
from data_loaders import *

#dump all these to a config file
EPOCH = 100
BATCH_SIZE = 100
torch.set_num_threads(8)

def train_classifier_d(X, Y, classifier, classifier_optimizer, verbose=False):

	classifier_optimizer.zero_grad()
	Y_pred = classifier(X)
	pred_loss = classification_loss(Y_pred, Y)/BATCH_SIZE
	# pred_loss = pred_loss.sum()
	pred_loss.backward()
	
	if verbose:
		# print(torch.cat([Y_pred, Y, Y*torch.log(Y_pred),
		# (Y*torch.log(Y_pred)).sum().unsqueeze(0).unsqueeze(0).repeat(BATCH_SIZE,1) ],dim=1))
		for p in classifier.parameters():
			print(p.data)
			print(p.grad.data)
			print("____")
	classifier_optimizer.step()

	return pred_loss

def train(X_data, Y_data, U_data, num_indices, source_indices, target_indices):

	## BASELINE 1- Sequential training with no adaptation ##

	X_source = X_data[source_indices[0]]
	Y_source = Y_data[source_indices[0]]
	Y_source = np.array([0 if y[0] > y[1] else 1 for y in Y_source])
	print(Y_source.shape)
	X_aux = list(X_data[source_indices[1:]])
	Y_aux = list(Y_data[source_indices[1:]])
	Y_aux2 = []
	for i in range(len(Y_aux)):
		Y_aux2.append(np.array([0 if y[0] > y[1] else 1 for y in Y_aux[i]]))

	Y_aux = Y_aux2

	print(len(X_aux))
	print(len(Y_aux))

	X_target = X_data[target_indices[0]]
	Y_target = Y_data[target_indices[0]]
	Y_target = np.array([0 if y[0] > y[1] else 1 for y in Y_target])

	X_source, X_aux, X_target = transform_samples_reg_otda(X_source, Y_source, X_aux, Y_aux, X_target, Y_target)

	print(X_source.shape)
	print(Y_source.shape)
	print(X_target.shape)
	print(Y_target.shape)
	print(X_aux[0].shape)
	print(Y_aux[0].shape)

	X_source = np.vstack([X_source] + X_aux)
	Y_source = np.hstack([Y_source] + Y_aux)
	Y_source = np.eye(2)[Y_source]
	Y_target = np.eye(2)[Y_target]
	print(X_source.shape)
	print(Y_source.shape)
	print(X_target.shape)
	print(Y_target.shape)
	
	classifier = ClassifyNet(670,[256, 256, 128],2)

	classifier_optimizer = torch.optim.Adam(classifier.parameters(), 1e-3)

	writer = SummaryWriter(comment='{}'.format(time.time()))

	past_data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_source).float(), torch.tensor(Y_source).float()),BATCH_SIZE,False)    
	print('------------------------------------------------------------------------------------------')
	print('TRAINING - DOMAIN: %d' % i)
	print('------------------------------------------------------------------------------------------')
	for epoch in range(EPOCH):
		loss = 0
		for batch_X, batch_Y in past_data:
			loss += train_classifier_d(batch_X, batch_Y, classifier, classifier_optimizer, verbose=False)
		if epoch%10 == 0: print('Epoch %d - %f' % (epoch, loss.detach().cpu().numpy()))

	print('------------------------------------------------------------------------------------------')
	print('TESTING')
	print('------------------------------------------------------------------------------------------')
		
	target_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_target).float(), torch.tensor(Y_target).float()),BATCH_SIZE,False)
	Y_pred = []
	for batch_X, batch_Y in target_dataset:
		batch_Y_pred = classifier(batch_X).detach().cpu().numpy()

		Y_pred = Y_pred + [batch_Y_pred]  
	Y_pred = np.vstack(Y_pred)
	print('shape: ',Y_pred.shape)
	# print(Y_pred)
	Y_pred = np.array([0 if y[0] > y[1] else 1 for y in Y_pred])
	Y_true = np.array([0 if y[0] > y[1] else 1 for y in Y_target])

	# print(Y_pred-Y_true)
	print(accuracy_score(Y_true, Y_pred))
	print(confusion_matrix(Y_true, Y_pred))
	print(classification_report(Y_true, Y_pred))    
		
	

if __name__ == "__main__":

	X_data, Y_data, A_data, U_data = load_sleep2('shhs1-dataset-0.15.0.csv')
	X_data = preprocess_sleep2(X_data, [0, 1, 2, 3])
	#incremental_finetuning(X_data, Y_data, A_data, U_data, 5, [0, 1, 2, 3], [4])
	train(X_data, Y_data, U_data, 5, [0, 1, 2,3], [4])