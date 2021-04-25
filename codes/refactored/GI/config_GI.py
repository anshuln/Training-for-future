from losses import *



class Config():
	def __init__(self,args):
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

		self.delta = args.delta
		self.max_k = args.max_k

		log_file_name = 'debugging_no_curric_{}_{}'.format(args.data,args.model)
		if args.trelu_limit < 1000:
			log_file_name += '_trelulimit-{}'.format(args.trelu_limit)
		if args.single_trelu:
			log_file_name += '_single-trelu'
		if args.delta > 0.0 or args.delta < 0.0:
			log_file_name += '_delta-{}'.format(args.delta)
		if args.max_k > 1:
			log_file_name += '_max-k-{}'.format(args.max_k)			
		if args.time_softmax:
			log_file_name += '_time-softmax'
		if args.ensemble:
			log_file_name += '_ensemble'

		self.log = open(log_file_name,"a")
		print("seed - {}".format(args.seed),file=self.log)


		if args.data == "house":

			self.dataset_kwargs = {"root_dir":"../../data/HousePrice","device":args.device, "drop_cols":None}
			self.source_domain_indices = [6,7,8,9,10]
			self.target_domain_indices = [11]
			self.data_index_file = "../../data/HousePrice/indices.json"
			from models_GI import ClassifyNetHuge
			self.classifier = ClassifyNetHuge 
			self.model_kwargs =  {'time_conditioning':True,'task':'regression','use_time2vec':True,'leaky':True,"input_shape":31,"hidden_shapes":[400,400,400],"output_shape":1,'append_time':True, "trelu_limit" : args.trelu_limit, "single_trelu":args.single_trelu, "add_delta_f": args.model == "GI_v3", "time_softmax": args.time_softmax}
			self.lr = 5e-3
			self.classifier_loss_fn = reconstruction_loss
			self.loss_type = 'regression'
			self.encoder = None

			self.delta_lr=5e-4
			self.delta_clamp=0.15
			self.delta_steps=30
			self.lambda_GI=0.5

		if args.data == "mnist":

			self.dataset_kwargs = {"root_dir":"../../data/MNIST/processed/","device":args.device, "drop_cols":None}
			self.source_domain_indices = [0,1,2,3]
			self.target_domain_indices = [4]
			self.data_index_file = "../../data/MNIST/processed/indices.json"
			from models_GI import ResNet, ResidualBlock
			self.classifier = ResNet 
			self.model_kwargs =  {
									"block": ResidualBlock,
									"layers": [2, 2, 2, 2],
									"append_time": True,
									"time_conditioning": True,
									"use_time2vec": True,
									"trelu_limit" : args.trelu_limit,
									"single_trelu":args.single_trelu,
									"add_delta_f": args.model == "GI_v3",
									"time_softmax": args.time_softmax
								}
			self.lr = 1e-3
			self.classifier_loss_fn = classification_loss
			self.loss_type = 'classification'
			self.encoder = None

			self.delta_lr=1e-1
			self.delta_clamp=0.15
			self.delta_steps=15
			self.lambda_GI=1.0

		if args.data == 'moons':

			self.dataset_kwargs = {"root_dir":"../../data/Moons/processed","device":args.device, "drop_cols":None}
			self.source_domain_indices = [0,1, 2, 3, 4, 5, 6, 7, 8]
			self.target_domain_indices = [9]
			self.data_index_file = "../../data/Moons/processed/indices.json"
			from models_GI import PredictionModel
			self.classifier = PredictionModel
			self.model_kwargs =  {"data_shape":3, "hidden_shape":6, "out_shape":1,  "time2vec":True,"append_time":True,"time_conditioning":True}
			self.lr = 1e-3
			self.classifier_loss_fn = binary_classification_loss
			self.loss_type = 'classification'
			self.encoder = None

			self.delta_lr=0.05
			self.delta_clamp=0.5
			self.delta_steps=5
			self.lambda_GI=1.0



