from losses import *

'''This file contains the configs for various datasets

'''

class Config():
	def __init__(self,args):
		'''
		args - The Namespace object passed through command line arguments. 
		'''
		self.epoch_classifier = args.epoch_classifier
		self.epoch_finetune = args.epoch_finetune 
		self.SUBEPOCHS = 1
		self.EPOCH = args.epoch_finetune // self.SUBEPOCHS
		self.bs = args.bs
		self.CLASSIFICATION_BATCH_SIZE = 100
		# self.PRETRAIN_EPOCH = 5
		self.data = args.data 
		self.update_num_steps = 1
		self.num_finetune_domains = 2

		if args.data == "house":
			self.epoch_classifier = args.epoch_classifier   # 15
			self.epoch_finetune = args.epoch_finetune # 10
			self.bs = args.bs  # 1000

			self.dataset_kwargs = {"root_dir":"../../data/HousePrice","device":args.device, "drop_cols":None}
			self.source_domain_indices = [6,7,8,9,10]
			self.target_domain_indices = [11]
			self.data_index_file = "../../data/HousePrice/indices.json"
			from models_GI import ClassifyNetHuge
			self.classifier = ClassifyNetHuge 
			self.model_kwargs =  {'time_conditioning':True,'task':'regression','use_time2vec':True,'leaky':True}
			self.lr = 5e-4
			self.classifier_loss_fn = reconstruction_loss
			self.loss_type = 'regression'
			self.encoder = None

			self.delta_lr=0.1
			self.delta_clamp=0.15
			self.delta_steps=10
			self.lambda_GI=0.5

		if args.data == "mnist":

			self.epoch_classifier = args.epoch_classifier   # 80
			self.epoch_finetune = args.epoch_finetune # 20
			self.bs = args.bs  # 250

			self.dataset_kwargs = {"root_dir":"../../data/MNIST/processed/","device":args.device, "drop_cols":None}
			self.source_domain_indices = [0,1,2,3]
			self.target_domain_indices = [4]
			self.data_index_file = "../../data/MNIST/processed/indices.json"
			from models_GI import ResNet, ResidualBlock
			self.classifier = ResNet 
			self.model_kwargs =  {
									"block": ResidualBlock,
									"layers": [2, 2, 2, 2]   
								}
			self.lr = 1e-4
			self.classifier_loss_fn = classification_loss
			self.loss_type = 'classification'
			self.encoder = None

			self.delta_lr=0.05
			self.delta_clamp=0.05
			self.delta_steps=5
			self.lambda_GI=1.0

		if args.data == 'moons':

			self.epoch_classifier = args.epoch_classifier   # 30
			self.epoch_finetune = args.epoch_finetune # 25
			self.bs = args.bs  # 250
		
			self.dataset_kwargs = {"root_dir":"../../data/moons","device":args.device, "drop_cols":None}
			self.source_domain_indices = [0,1, 2, 3, 4, 5, 6, 7, 8]
			self.target_domain_indices = [9]
			self.data_index_file = "../../data/moons/indices.json"
			from models_GI import PredictionModel
			self.classifier = PredictionModel
			self.model_kwargs =  {"data_shape":6, "hidden_shape":3, "out_shape":1, "time2vec":True}
			self.lr = 1e-3
			self.classifier_loss_fn = binary_classification_loss
			self.loss_type = 'classification'
			self.encoder = None

			self.delta_lr=0.05
			self.delta_clamp=0.5
			self.delta_steps=5
			self.lambda_GI=1.0



