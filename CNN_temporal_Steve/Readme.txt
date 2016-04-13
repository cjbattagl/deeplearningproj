Command: qlua run.lua -b 10 -p cuda

notes:
1. You need to downoload the feature data and modify the path in the code "data.lua"
2. Now I am using "model" (2-layer architecture).
	TODO: 
	(1) model_1: 	1 layer
	(2) model_2D: 	2 layers with 2D kernels
	(3) model_3:	3 layers
3. best performance: in the folder "results_1L"
[25,250] + [11] + [2] (40 epochs)
training: 100
testing: 62.213

