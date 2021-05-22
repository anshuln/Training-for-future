'''Main script to run stuff

[description]
'''

import argparse
import os
import random
from trainer_GI import *
from preprocess import *

device = "cuda:0"

def main(args):

    if args.use_cuda:
        args.device = "cuda:0"
    else:
        args.device = "cpu"

    if args.preprocess:
        if args.data == "mnist":
            print("Preprocessing")
            load_Rot_MNIST(args.encoder)
        if args.data == "moons":
            load_moons(11,args.model)
        if args.data == "sleep":
            load_sleep2()
        if args.data == "cars":
            load_comp_cars()
        if args.data == "house":
            load_house_price(args.model)
        if args.data == "house_classifier":
            load_house_price_classification()             
    if args.seed is not None:
        seed = int(args.seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed) 
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # torch.random.manual_seed(int(args.seed))
        # np.random.seed(int(args.seed))

    trainer = GradRegTrainer(args)
    # if args.train_algo == "transformer":
    #   trainer = TransformerTrainer(args)
    # elif args.train_algo == "grad":
    #   trainer = CrossGradTrainer(args)
    # elif args.train_algo == "meta":
    #   trainer = MetaTrainer(args)
    # elif args.train_algo == "grad_reg":
    #   trainer = GradRegTrainer(args)
    # elif args.train_algo == "hybrid":
    #   trainer = HybridTrainer(args)
    trainer.train()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """ Arguments: arg """
    parser.add_argument('--model',help="String, needs to be one of baseline,tbaseline,goodfellow,GI")
    parser.add_argument('--data',help="String, needs to be one of mnist, sleep, moons, cars")
    parser.add_argument('--epoch_finetune',default=5,help="Needs to be int, number of epochs for transformer/ordinal classifier",type=int)
    parser.add_argument('--epoch_classifier',default=5,help="Needs to be int, number of epochs for classifier",type=int)
    parser.add_argument('--bs',default=100,help="Batch size",type=int)
    parser.add_argument('--aug_steps',default=10,help="Number of steps of data augmentation to do",type=int)
    parser.add_argument('--wasserstein_disc',action='store_true',help="Should we use a wasserstein discriminator")
    parser.add_argument('--early_stopping',action='store_true',help="Early Stopping for finetuning")
    parser.add_argument('--use_cuda',action='store_true',help="Should we use a GPU")
    parser.add_argument('--preprocess',action='store_true',help="Do we pre-process the data?")
    parser.add_argument('--encoder',action='store_true',help="Do we use encodings?")
    parser.add_argument('--goodfellow',action='store_true',help="Do we use goodfellow perturbations?")
    parser.add_argument('--delta',default=0.0,type=float)
    parser.add_argument('--max_k',default=1,type=float)

    parser.add_argument('--seed',default=None)
    parser.add_argument('--trelu_limit',default=1000,type=int)
    parser.add_argument('--single_trelu',action='store_true',help="Should each trelu output only a single param?")
    parser.add_argument('--ensemble',action='store_true',help="Should final prediction be an ensemble of linearized functions?")
    parser.add_argument('--time_softmax',action='store_true',help="Add a time dependent bias to the softmax?")
    # parser.add_argument('--ensemble',action='store_true',help="Should final prediction be an ensemble of linearized functions?")
    
    args = parser.parse_args()
    # print("seed - {} model - {}".format(args.seed,args.model),file=open('testing_{}_{}_{}.txt'.format(args.model,args.data,args.trelu_limit),'a'))
    main(args)
