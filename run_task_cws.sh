# Task type
task_type="cws"

# path of training data
path_data="./data/"

# path of conf file
path_aspect_conf="conf.cws-attributes"

# # Part1: Dataset Name
datasets[0]="ctb"

# # Part2: Model Name
models[0]="CrandBavgLstmCrf"
models[1]="Cw2vBavgLstmCrf"
models[2]="Cw2vBavgLstmMlp"
models[3]="Cw2vBavgCnnCrf"
models[4]="Cw2vBw2vLstmCrf"
models[5]="CelmBnonLstmMlp"
models[6]="CbertBnonLstmMlp"
models[7]="CbertBw2vLstmMlp"

# # Part3: Path of result files
resfiles[0]="ctb_results_8models/ctbtest_CrandBavgLstmCrf_08857307_9532_format.txt"
resfiles[1]="ctb_results_8models/ctbtest_Cw2vBavgLstmCrf_84074011_9508_format.txt"
resfiles[2]="ctb_results_8models/ctbtest_Cw2vBavgLstmMlp_52772725_9409_format.txt"
resfiles[3]="ctb_results_8models/ctbtest_Cw2vBavgCnnCrf_37186015_9472_format.txt"
resfiles[4]="ctb_results_8models/ctbtest_Cw2vBw2vLstmCrf_95751718_9514_format.txt"
resfiles[5]="ctb_results_8models/ctbtest_CelmBnonLstmMlp_81936000_9677.txt"
resfiles[6]="ctb_results_8models/ctb_CbertBnonLstmMlp_9768.txt"
resfiles[7]="ctb_results_8models/ctb_CbertBw2vLstmMlp_9761.txt"

path_preComputed="./preComputed"
path_fig=$task_type"-fig"
path_output_tensorEval="output_tensorEval/"$task_type/$model1"-"$model2


# -----------------------------------------------------
rm -fr $path_output_tensorEval/*
echo "${datasets[*]}"
python3 tensorEvaluation-cws.py \
	--path_data $path_data \
	--task_type $task_type  \
	--path_fig $path_fig \
	--data_list "${datasets[*]}"\
	--model_list "${models[*]}" \
	--path_preComputed $path_preComputed \
	--path_aspect_conf $path_aspect_conf \
	--resfile_list "${resfiles[*]}" \
	--path_output_tensorEval $path_output_tensorEval \
	# --path_sampsave $path_sampsave 
