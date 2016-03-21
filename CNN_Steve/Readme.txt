command: qlua run.lua (-v your_video)

It will do the following steps:
1. download the videos from my dropbox (will modify it. There are only some test videos)
2. load the pre-trained model ("NIN" network) & the ImageNet labels
3. load the previously downloaded video
4. process the video & make prediction frame-by-frame