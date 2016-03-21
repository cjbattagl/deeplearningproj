-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Video Classification		

-- It's a function version: 
-- 1. read the image from the input
-- 2. comment out most of the print functions
-- 3. set local variables


-- 1. Models: there are two kinds of pre-trained models:
-- 		1) use 'loadcaffe' to load a pre-trained model built in caffe (produce strange results now)
--		2) original Torch model (now I only have NIN)
-- 2. extract the intermediate feature (e.g. fc8)
-- 3. input: 	image
--	  output:	feature vector/prediction labels

-- author: Min-Hung Chen
-- contact: cmhungsteve@gatech.edu
-- Last updated: 03/20/2016

----------------------------------------------
-- 					Functions 				--
----------------------------------------------
-- Rescales and normalizes the image
function preprocess(im, img_mean, img_std)
  -- rescale the image
  local im3 = image.scale(im,224,224,'bilinear')
  -- subtract imagenet mean and divide by std
  for i=1,3 do im3[i]:add(-img_mean[i]):div(img_std[i]) end
  return im3
end


function classify_video(inFrame, net, synset_words)

require 'loadcaffe' 
require 'image'
require 'nn'

----------------------------------------------
-- 				Input images				--
----------------------------------------------
--print '==> Loading image'
--im = image.load(dir_image .. image_name)
--local im = inFrame.clone() ==> it show errors (bad argument #1 to 'clone')
local im = torch.Tensor(inFrame:size()):copy(inFrame)

----------------------------------------------
-- 			 Image Pre-processing			--
----------------------------------------------
--print '==> Preprocessing (need to add net.transform)'
--I = im.clone() ==> it show errors (bad argument #1 to 'clone')
--I = image.scale(torch.Tensor(im:size()):copy(im):float(),224,224,'bilinear')

-- mean & std
local img_mean = torch.Tensor({0.48462227599918, 0.45624044862054, 0.40588363755159})
local img_std = torch.Tensor({0.22889466674951, 0.22446679341259, 0.22495548344775})

-- rescale & normalization
--print('ImageNet')
--print(img_mean, img_std)
local I = preprocess(im, img_mean, img_std):view(1,3,224,224):float()

----------------------------------------------
-- 			 Forward Propagation			--
----------------------------------------------
-- 1. forward to get feature vectors
--print 'Obtain the final feature'
--feat = net:forward(I)
--print(feat)

-- 2. forward to get prediction labels
-- -- (1) Top 1 prediction
-- -- print 'Propagate through the model, show the best classes'
-- --local _,classes = net:forward(I):view(-1):sort(true)
-- local _,classes = net:forward(I):view(-1):max(1)
--   print('The predicted class is ', synset_words[classes[1] ])

-- Top 3 predictions
-- print 'Propagate through the model, sort outputs in decreasing order and show 3 best classes'
local _,classes = net:forward(I):view(-1):sort(true)
--_,classes = net:forward(I):sort(true)
--print(classes)
for i=1,3 do
  print('predicted class '..tostring(i)..': ', synset_words[classes[i] ])
end



end