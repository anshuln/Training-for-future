

echo "MNIST Results -----------------------------------------------"

for model  in 'baseline' 'tbaseline' 'inc_finetune' 't_inc_finetune'
do  
	echo "$model"
	grep 'Accuracy: ' "results_""$model""_mnist.txt" | sed -n 's/Accuracy:  //p' > acc.txt 
	python3 compute_mean_std.py --type acc
done
for model  in 'GI' 'goodfellow' 't_goodfellow' 't_GI'
do  
	echo "$model"
	grep 'Accuracy: ' "results_""$model""_mnist.txt" | sed -n '1~2!p' | sed -n 's/Accuracy:  //p' > acc.txt 
	python3 compute_mean_std.py --type acc
	echo "$model""_baseline"
	grep 'Accuracy: ' "results_""$model""_mnist.txt" | sed -n '2~2!p' | sed -n 's/Accuracy:  //p' > acc.txt 
	python3 compute_mean_std.py --type acc
done



# echo "Moons Results -----------------------------------------------"

# for model  in 'baseline' 'tbaseline' 'inc_finetune' 't_inc_finetune'
# do  
# 	echo "$model"
# 	grep 'Accuracy: ' "results_""$model""_moons.txt" | sed -n 's/Accuracy:  //p' > acc.txt 
# 	python3 compute_mean_std.py --type acc
# done
# for model  in 'GI' 'goodfellow' 't_goodfellow' 't_GI'
# do  
# 	echo "$model"
# 	grep 'Accuracy: ' "results_""$model""_moons.txt" | sed -n '1~2!p' | sed -n 's/Accuracy:  //p' > acc.txt 
# 	python3 compute_mean_std.py --type acc
# done



echo "Housing Results -----------------------------------------------"
for model  in 'baseline' 'tbaseline' 'inc_finetune' 't_inc_finetune'
do  
	echo "$model"
	echo "MSE"
	# grep 'MSE: '  | sed -n 's/Accuracy:  //p' > acc.txt 
	grep 'MSE: ' "results_""$model""_house.txt"  | sed -n 's/MSE:  //p' | sed -n 's/\[//p' | sed -n 's/\]//p' > acc.txt
	python3 compute_mean_std.py --type MSE 
	echo "MAE"
	grep 'MAE: ' "results_""$model""_house.txt"  | sed -n 's/MAE:  //p' | sed -n 's/\[//p' | sed -n 's/\]//p' > acc.txt
	python3 compute_mean_std.py --type MAE 

done
for model  in 'GI' 'goodfellow' 't_goodfellow' 't_GI'
do  
	echo "$model"
	echo "MSE"
	# grep 'MSE: '  | sed -n 's/Accuracy:  //p' > acc.txt 
	grep 'MSE: ' "results_""$model""_house.txt" | sed -n '1~2!p' | sed -n 's/MSE:  //p' | sed -n 's/\[//p' | sed -n 's/\]//p' > acc.txt
	python3 compute_mean_std.py --type MSE 
	echo "MAE"
	grep 'MAE: ' "results_""$model""_house.txt" | sed -n '1~2!p' | sed -n 's/MAE:  //p' | sed -n 's/\[//p' | sed -n 's/\]//p' > acc.txt
	python3 compute_mean_std.py --type MAE 

	echo "$model""_baseline"
	echo "MSE"
	# grep 'MSE: '  | sed -n 's/Accuracy:  //p' > acc.txt 
	grep 'MSE: ' "results_""$model""_house.txt" | sed -n '2~2!p' | sed -n 's/MSE:  //p' | sed -n 's/\[//p' | sed -n 's/\]//p' > acc.txt
	python3 compute_mean_std.py --type MSE 
	echo "MAE"
	grep 'MAE: ' "results_""$model""_house.txt" | sed -n '2~2!p' | sed -n 's/MAE:  //p' | sed -n 's/\[//p' | sed -n 's/\]//p' > acc.txt
	python3 compute_mean_std.py --type MAE 

done
# grep 'MSE: ' results_GI_house.txt | sed -n '2~2!p' | sed -n 's/MSE:  //p' | sed -n 's/\[//p' | sed -n 's/\]//p' > acc.txt
# python3 compute_mean_std.py --type MSE

# grep 'MAE: ' results_GI_house.txt | sed -n '2~2!p' | sed -n 's/MAE:  //p' | sed -n 's/\[//p' | sed -n 's/\]//p' > acc.txt
# python3 compute_mean_std.py --type MAE
