This is rather a note for me to remember what experiments have been done and what needs to be done in order to moving forward. 

---
TODO: 
- [ ] Experiment with different number of frames for training
- [x] Experiment with different batch size and learning rate
- [x] Train with different optimizers, like sgd, adam, adamax, rmsprop
- [x] Comparison between GRU and LSTM, but they should pretty much perform the same
- [ ] Have all the feature vectors from all frames for testing, not just 50 frames per video 
- [x] During testing, if you have at least two frames, input for the network should be feature vector from 1 to t-1
- [x] Feedforward from each time step and average across all frames over a video to make prediction


Note: 

Optimizer: tried SGD and Adam. It gives faster and so far the best convergence. it requires relatively smaller learning rate compared with SGD. I am using **LearningRate = 5e-4** for Adam. 

Hidden layers: using hidden number all the way to 128 (1024, 768, 512, 256, 128) seems to be overfitting the data. With the feature vector dimension to be 1024, the number of hidden layers (1024, 512, 256) can achieve 92% accuracy. 

UCF101 training and testing list: this training and testing list completely ruined the overall performance. Using hidden layers with (1024, 512, 256) only get closed to 60% accuracy. 


#### Contact: [Chih-Yao Ma](http://shallowdown.wix.com/chih-yao-ma) at <cyma@gatech.edu>
