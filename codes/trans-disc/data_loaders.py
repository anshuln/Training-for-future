import pandas as pd
import numpy as np
import math
from sklearn.datasets import make_classification, make_moons

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
