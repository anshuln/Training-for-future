# Training with Gradient Interpolation

## Setup
* Needs `torch>=1.4` and corresponding `torchvision`. Also needs `tqdm`, `numpy`, `sklearn`. 
* For data setup, populate the `data` directory with files taken from the server

## Code overview
* The file `trainer_GI.py` contains the training algorithm in the function `adversarial_finetune`. It also implements a trainer class for boiler plate code.
* The file `config_GI.py` has the list of configs for different data sets. Change this for hyperparams
* `models_GI.py` has the model definitions, `main.py` is the entrypoint to the code

## Running the code
Run `python3 -W ignore main.py --data <DS> --epoch_classifier <NUM> --epoch_finetune <NUM> --bs <NUM> --use_cuda --early_stopping --algo grad_reg` with the dataset name. 

To run all the experiments, run the bash script `run_experiments_grad_reg.sh` followed by the bash script `gen_numbers.sh`. 
