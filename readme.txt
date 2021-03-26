1. 结果文件
目录里提供了8个模型（即 interpreCWS 论文里使用的）在数据集：ctb 上的结果文件。
结果文件存放的位置：./ctb_results_8models
结果文件的命名：例如“ctb_CbertBnonLstmMlp_9768.txt”， “ctb”表示数据集， “CbertBnonLstmMlp”表示模型的名字，“9768”表示“ctb”在模型“CbertBnonLstmMlp” 下的结果为“97.68”
结果文件的存储格式：每一个句子用空行隔开，每一行有三个token按照空格隔开，从左到右依次是：word, true_tag, predict_tag.

2. 属性的计算
interpreCWS论文里的7个属性的计算，在“tensorEvaluation-cws.py”文件里有提供，即new_metric() 函数内部的实现已经将每个词的7个属性值都计算好了的。

3. 这里给出了同一个数据集（ctb）在在八个模型下的结果文件，那么可以先复现下论文“A New Psychometric-inspired Evaluation Metric for Chinese Word Segmentation” 中提出来的 balanced F-score.
我们先复现下论文里的balanced F-score，就可以先不考虑属性。tensorEvaluation-cws.py代码可以参考的读取:
1）测试集合的读取：read_data_test2()
2）训练集合的读取：read_data()




