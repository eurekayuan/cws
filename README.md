# Notes
目录里提供了8个模型（即 interpreCWS 论文里使用的）在数据集：ctb 上的结果文件。
结果文件存放的位置：./ctb_results_8models
结果文件的命名：例如“ctb_CbertBnonLstmMlp_9768.txt”， “ctb”表示数据集， “CbertBnonLstmMlp”表示模型的名字，“9768”表示“ctb”在模型“CbertBnonLstmMlp” 下的结果为“97.68”
结果文件的存储格式：每一个句子用空行隔开，每一行有三个token按照空格隔开，从左到右依次是：word, true_tag, predict_tag.



