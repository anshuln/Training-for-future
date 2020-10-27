'''Main script to run stuff

[description]
'''

import argparse
from trainer import *
from preprocess import *

device = "cuda:0"

def main(args):
	if args.use_cuda:
		args.device = "cuda:0"
	else:
		args.device = "cpu"

	if args.preprocess:
		if args.data == "mnist":
			load_Rot_MNIST(args.encoder)
		if args.data == "moons":
			load_moons(args)
		if args.data == "sleep":
			load_sleep2()

	if args.train_algo == "transformer":
		trainer = TransformerTrainer(args)
	elif args.train_algo == "grad":
		trainer = CrossGradTrainer(args)

	trainer.train()




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	""" Arguments: arg """
	parser.add_argument('--train_algo',help="String, needs to be one of grad or transformer")
	parser.add_argument('--data',help="String, needs to be one of mnist, sleep, moons")
	parser.add_argument('--epoch_transform',default=5,help="Needs to be int, number of epochs for transformer/ordinal classifier",type=int)
	parser.add_argument('--epoch_classifier',default=5,help="Needs to be int, number of epochs for classifier",type=int)
	parser.add_argument('--bs',default=256,help="Batch size",type=int)
	parser.add_argument('--wasserstein_disc',action='store_true',help="Should we use a wasserstein discriminator")
	parser.add_argument('--use_cuda',action='store_true',help="Should we use a GPU")
	parser.add_argument('--preprocess',action='store_true',help="Do we pre-process the data?")
	parser.add_argument('--encoder',action='store_true',help="Do we use encodings?")

	
	args = parser.parse_args()
	main(args)
