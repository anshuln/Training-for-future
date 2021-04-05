rm -f results_grad.txt 

for seed in 1 2 3 4 5 6 7 8 9 10
do
	for delta in 0.0 #0.01 0.05 0.1 0.2 0.5
	do
		# CUDA_VISIBLE_DEVICES=1 python3 -W ignore main.py --train_algo grad_reg --data house --epoch_classifier 15 --epoch_transform 5 --use_cuda --bs 1000 --delta $delta --seed $seed --early_stopping> temp_log;
		CUDA_VISIBLE_DEVICES=1 python3 -W ignore main.py --model baseline  --data house --epoch_classifier 25  --epoch_finetune 10  --bs 1000  --seed $seed --use_cuda --early_stopping 
		CUDA_VISIBLE_DEVICES=1 python3 -W ignore main.py --model tbaseline  --data house --epoch_classifier 25  --epoch_finetune 10  --bs 1000  --seed $seed --use_cuda --early_stopping 
		CUDA_VISIBLE_DEVICES=1 python3 -W ignore main.py --model goodfellow  --data house --epoch_classifier 25  --epoch_finetune 10  --bs 1000  --seed $seed --use_cuda --early_stopping 
		CUDA_VISIBLE_DEVICES=1 python3 -W ignore main.py --model GI  --data house --epoch_classifier 25  --epoch_finetune 10  --bs 1000  --seed $seed --use_cuda --early_stopping --goodfellow
		# echo $seed $delta | tee -a results_grad.txt;
		# grep "MAE:" temp_log | tee -a results_grad.txt;
		# grep "MSE:" temp_log | tee -a results_grad.txt;
	done;
done;
