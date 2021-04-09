import numpy as np 
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	""" Arguments: arg """
	parser.add_argument('--type')
	
	args = parser.parse_args()
	
	arr = np.loadtxt("acc.txt")
	# arr = arr.

	if args.type == "acc":
		thresh = lambda x: x > 0.3
	elif args.type == "MSE":
		thresh = lambda x: x<2000
	elif args.type == "MAE":
		thresh = lambda x: x<50

	arr = [x for x in arr.tolist() if thresh(x)]
	arr = np.array(arr)

	print("Mean : {}, Std : {}, Number: {}".format(arr.mean(), arr.std(), len(arr)))


