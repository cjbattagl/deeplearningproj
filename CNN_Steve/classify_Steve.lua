-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Video Classification		

-- 1. Models: there are two kinds of pre-trained models:
-- 		1) use 'loadcaffe' to load a pre-trained model built in caffe (produce strange results now)
--		2) original Torch model (now I only have NIN)
-- 2. extract the intermediate feature (e.g. fc8)
-- 3. input: 	image
--	  output:	feature vector/prediction labels

-- author: Min-Hung Chen
-- contact: cmhungsteve@gatech.edu
-- Last updated: 03/19/2016

--require 'loadcaffe' 
require 'image'
require 'nn'

-- data path
dir_image = './images/'
dir_model = './models/'

----------------------------------------------
-- 			User-defined parameters			--
----------------------------------------------
------ model selection ------
-- 1. NIN model (from Torch)
model_name = dir_model .. 'nin_nobn_final.t7'

---- 2. GoogleNet model (from Torch) ==> need cuda
--model_name = dir_model .. 'GoogLeNet_v2.t7'

---- 3. NIN model (from caffe)
--prototxt = dir_model .. './solver.prototxt'
--binary = dir_model .. './nin_imagenet.caffemodel'

-- 4. VGG model (from caffe)
--prototxt = dir_model .. './VGG_ILSVRC_19_layers_deploy.prototxt'
--binary = dir_model .. './VGG_ILSVRC_19_layers.caffemodel'

------ input image ------
--image_name = 'Goldfish3.jpg'
image_name = 'frame-000001.png'

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
net = torch.load(model_name):unpack():float()

-- 2. Caffe model
--net = loadcaffe.load(prototxt, binary):float()

net:evaluate()

-- model modification
--net:remove(30)
--net:add(nn.View(-1))

print(net)

----------------------------------------------
-- 				Input images				--
----------------------------------------------
print '==> Loading image'
im = image.load(dir_image .. image_name)

----------------------------------------------
-- 			Loading ImageNet Labels			--
----------------------------------------------
print '==> Loading synsets'
print 'Loads mapping from net outputs to human readable labels'
synset_words = {}
for line in io.lines'synset_words.txt' do table.insert(synset_words, line:sub(11)) end

----------------------------------------------
-- 					GPU option				--
----------------------------------------------
if arg[1] == 'cuda' then
  net:cuda()
  images = images:cuda()
else
  net:float()
end

----------------------------------------------
-- 			 Image Pre-processing			--
----------------------------------------------
print '==> Preprocessing (need to add net.transform)'
--I = im.clone() ==> it show errors......
--I = image.scale(torch.Tensor(im:size()):copy(im):float(),224,224,'bilinear')

-- mean & std
img_mean = torch.Tensor({0.48462227599918, 0.45624044862054, 0.40588363755159})
img_std = torch.Tensor({0.22889466674951, 0.22446679341259, 0.22495548344775})

-- rescale & normalization
print('ImageNet')
print(img_mean, img_std)
I = preprocess(im, img_mean, img_std):view(1,3,224,224):float()

----------------------------------------------
-- 			 Forward Propagation			--
----------------------------------------------
-- 1. forward to get feature vectors
--print 'Obtain the final feature'
--feat = net:forward(I)
--print(feat)

-- 2. forward to get prediction labels
print 'Propagate through the model, sort outputs in decreasing order and show 5 best classes'
_,classes = net:forward(I):view(-1):sort(true)
--_,classes = net:forward(I):sort(true)
--print(classes)
for i=1,5 do
  print('predicted class '..tostring(i)..': ', synset_words[classes[i] ])
end
