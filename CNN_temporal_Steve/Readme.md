###Command: qlua run.lua -b 10 -p float -r 15e-5

1. You need to downoload the feature data and modify the path in the code "data.lua"
2. Now I am using "model_Res" (2-layer architecture).
   	* TODO: model_3 (3 layers)
3. best performance: in the folder "results_Res/results_Res_3/"
	* [16,32,256] + [3,11] + [2,2] 
	* 53 epochs
	* training: 100 
	* testing: 77.47

