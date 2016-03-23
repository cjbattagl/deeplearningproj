run.lua: 
Load one video and do the following two things: 
1. generate the feature matrix (need 'gen_feature.lua')
2. make prediction frame-by-frame (need 'classify_video')
You can choose to turn the function on or off by yourself

command: qlua run.lua (-v your_video)

It will do the following steps:
1. load the video from '../Dataset/.....' (You need to download the UCF-11 first)
2. load the pre-trained model ("NIN" network) & the ImageNet labels
3. process the video

=============================================================================
run_UCF11.lua:
Load all the videos in UCF-11  

command: qlua run_UCF11.lua (-v your_video)

You need to put all the videos in the folder '../Dataset/.....'
=============================================================================
After running these codes, a new empty folder "out_frames" will be generated. You can ignore it. That's only for debugging.
