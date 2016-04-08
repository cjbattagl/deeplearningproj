-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Deep Learning for Video Classification		

-- 1. Models: there are two kinds of pre-trained models:
-- 		1) use 'loadcaffe' to load a pre-trained model built in caffe (produce strange results now)
--		2) original Torch model (now I only have NIN)
-- 2. extract the intermediate feature (e.g. fc8)
-- 3. input: 	image
--	  output:	feature vector/prediction labels

-- author: Min-Hung Chen
-- contact: cmhungsteve@gatech.edu
-- Last updated: 04/08/2016

require 'loadcaffe' 
require 'image'
require 'nn'
require 'cunn'
--require 'cudnn'

-- data path
dir_image = './images/'
dir_model = './models/'

----------------------------------------------
-- 			User-defined parameters			--
----------------------------------------------
------ model selection ------
-- 1. NIN model (from Torch)
model_name = 'nin_nobn_final.t7'

-- 2. GoogleNet model (from Torch) ==> need cudnn
--model_name = 'GoogLeNet_v2.t7'

-- 3. ResNet model (from Torch) ==> need cudnn
--model_name = 'resnet-18.t7'

---- 4. NIN model (from caffe)
--prototxt = dir_model .. './solver.prototxt'
--binary = dir_model .. './nin_imagenet.caffemodel'

-- 5. VGG model (from caffe)
-- prototxt = dir_model .. './VGG_ILSVRC_19_layers_deploy.prototxt'
-- binary = dir_model .. './VGG_ILSVRC_19_layers.caffemodel'
prototxt = dir_model .. './VGG_CNN_M_deploy.prototxt'
binary = dir_model .. './VGG_CNN_M.caffemodel'

------ input image ------
--image_name = 'cat.jpg'
--image_name = 'leopard.jpg'
image_name = 'dog.png'
--image_name = 'Goldfish3.jpg'
--image_name = 'frame-000001.png'
--image_name = 'annas_hummingbird_sim_1.jpg'

model_path = dir_model..model_name
image_path = dir_image..image_name
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

----------------------------------------------
-- 					Models 					--
----------------------------------------------
------ Loading the model ------
print '==> Loading model'
-- 1. Torch model
net = torch.load(model_path):unpack():float()

-- 2. Caffe model
-- net = loadcaffe.load(prototxt, binary)
-- net:remove(24) 

--net:evaluate()

-- model modification
-- net:remove(24) 
-- net:add(nn.View(-1))
-- net:add(nn.SoftMax())

-- -- for ResNet
-- softMaxLayer = cudnn.SoftMax():cuda()
-- model:add(softMaxLayer)

print(net)

----------------------------------------------
-- 				Input images				--
----------------------------------------------
print '==> Loading image'
im = image.load(image_path, 3, 'float')

----------------------------------------------
-- 			Loading ImageNet Labels			--
----------------------------------------------
-- for prediction method (a) --
print '==> Loading synsets'
print 'Loads mapping from net outputs to human readable labels'
synset_words = {}
for line in io.lines'synset_words.txt' do table.insert(synset_words, line:sub(11)) end

-- for prediction method (b) --
imagenetLabel = require './imagenet'


-- ----------------------------------------------
-- -- 					GPU option				--
-- ----------------------------------------------
--net:cuda()
-- if arg[1] == 'cuda' then
--   net:cuda()
--   images = images:cuda()
-- else
--   net:float()
-- end

----------------------------------------------
-- 			 Image Pre-processing			--
----------------------------------------------
print '==> Preprocessing (normalization)'
--I = im.clone() ==> it show errors......
--I = image.scale(torch.Tensor(im:size()):copy(im):float(),224,224,'bilinear')

-- mean & std
img_mean = torch.Tensor({0.48462227599918, 0.45624044862054, 0.40588363755159})
img_std = torch.Tensor({0.22889466674951, 0.22446679341259, 0.22495548344775})

-- mean_name = 'ilsvrc_2012_mean.t7'
-- mean_path = dir_model..mean_name
-- img_mean = torch.load(mean_path).img_mean:transpose(3,1):float():div(255)

-- rescale & normalization
print('ImageNet')
--print(img_mean, img_std)
I = preprocess(im, img_mean, img_std):view(1,3,224,224):float()
--I = preprocess(im, img_mean, img_std):float()

-- -- rescale the image
-- im_sc = image.scale(im,224,224,'bilinear')
-- mean_sc = image.scale(img_mean,224,224,'bilinear')
-- -- subtract imagenet mean 
-- I = im_sc - mean_sc

----------------------------------------------
-- 			 Forward Propagation			--
----------------------------------------------
-- 1. forward to get feature vectors
--print 'Obtain the final feature'
--feat = net:forward(I)
--print(feat)

-- 2. forward to get prediction labels
print 'Propagate through the model, sort outputs in decreasing order and show 5 best classes'

-- -- (a) original prediction codes 
-- out = net:forward(I:cuda())
-- print(out)
--print(net)

_,classes = net:forward(I):view(-1):sort(true)
--print(classes)
for i=1,5 do
  print('predicted class '..tostring(i)..': ', synset_words[classes[i] ])
end

-- -- (b) prediction codes from ResNet
-- N = 5
-- -- Get the output of the softmax
-- output = net:forward(I:cuda()):squeeze()
-- -- Get the top 5 class indexes and probabilities
-- probs, indexes = output:topk(N, true, true)
-- print('Classes for', image_name)
-- for n=1,N do
-- 	print(probs[n], imagenetLabel[indexes[n]])
-- end
-- print('')

