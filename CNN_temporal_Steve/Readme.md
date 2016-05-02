#Command: qlua run.lua -b 10 -p float -r 15e-5

1. You need to downoload the feature data and modify the path in the code "data.lua"
2. Now I am using "model_Res" (2-layer architecture).
   	* TODO: model_3 (3 layers)
3. best performance: in the folder "results_Res/results_Res_3/"
	* [16,32,256] + [3,11] + [2,2] 
	* 53 epochs
	* training: 100 
	* testing: 77.47

=============================================================================
#lua files:
1. run.lua:
similar as the one we used for HW3, but set the maximum epoch as 100.

2. data.lua:
similar as the one we used for HW3, but use NIN features and randomly divide data into a 80/20 split (80% train/20% test)

3. data_final.lua:
similar as above but experimented on the UCF-101 split 1 with ResNet-101 features. No shuffling in order to generate the labels for demo easily

4. train.lua:
similar as the one we used for HW3, but change "data_augmentation". We randomly selected the starting point for cropping. (We used 48 of 50 frames for each video.)
We added two more optimization methods: adam and rmsprop.

5. test.lua: 
similar as the one we used for HW3, but we store the labels everytime we improve the testing 
accuracy. We store the best testing accuracy as well. 

6. model.lua:
similar as the one we used for HW3, but change some parameters. Please see the final report. This model is for NIN features.

7. model_1L.lua:
1-layer architecture for NIN features.

8. model_2D.lua:
2-layer architecture with 2D kernels for NIN features.

9. model_Res.lua:
2-layer architecture with 1D kernels for ResNet-101 features. ==> current best results for T-CNN

10. model_VGG.lua:
2-layer architecture with 1D kernels for VGG-M features. ==> slightly better than NIN features.

11. nameList_Hao.lua: 
use to generate the labels for the demo video.
	* command: th nameList_Hao.lua > labels_demo.txt

-----------------------------------------------------------------------------
# text files:
1. stats.txt, stats_VGG, stats_Res: 
some statistics of these three models.
2. labels_demo.txt:
labels for demo
