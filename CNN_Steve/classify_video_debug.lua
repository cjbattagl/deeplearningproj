-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Deep Learning for Video Classification		

-- 1. Models: there are two kinds of pre-trained models:
-- 		1) use 'loadcaffe' to load a pre-trained model built in caffe
--		2) original Torch model 
-- 2. extract the intermediate feature (e.g. fc8)
-- 3. input: 	image
--	  output:	feature vector/prediction labels

-- author: Min-Hung Chen, modified by Hao Yan
-- contact: cmhungsteve@gatech.edu, yanhao@gatech.edu
-- Last updated: 04/14/2016

modelLang = 'torch' -- torch / caffe
typeCode = 2 -- 1. original codes; 2. from the ResNet sample codes
Ft = false -- true: extract feature; false: prediction

----------------------------------------------
--                Libraries                 --
----------------------------------------------

require 'image'
require 'nn'
require 'cunn'
require 'cudnn'

if modelLang == 'torch' then
  t = require './transforms'
elseif modelLang == 'caffe' then
  require 'loadcaffe' 
  matio = require 'matio'
end 

-- data path
dir_image = './images/'
dir_model = './models/'

----------------------------------------------
-- 			     User-defined parameters			  --
----------------------------------------------
------ model selection ------
if modelLang == 'torch' then
  -- 1. NIN model (from Torch)
  --model_name = 'nin_nobn_final.t7'

  -- 2. GoogleNet model (from Torch) ==> need cudnn
  --model_name = 'GoogLeNet_v2.t7'

  -- 3. ResNet model (from Torch) ==> need cudnn
  model_name = 'resnet-101.t7'

elseif modelLang == 'caffe' then
  ---- 4. NIN model (from caffe)
  --prototxt = dir_model .. './solver.prototxt'
  --binary = dir_model .. './nin_imagenet.caffemodel'

  -- 5. VGG model (from caffe)
  model_name = 'VGG'
  prototxt = dir_model .. './VGG_CNN_M_deploy.prototxt'
  binary = dir_model .. './VGG_CNN_M.caffemodel'

end

------ input image ------
--image_name = 'cat.jpg'
image_name = 'leopard.jpg'
--image_name = 'dog.png'
--image_name = 'Goldfish3.jpg'
--image_name = 'frame-000001.png'
--image_name = 'annas_hummingbird_sim_1.jpg'

model_path = dir_model..model_name
image_path = dir_image..image_name

----------------------------------------------
-- 					       Functions 				        --
----------------------------------------------
if typeCode == 1 then
  img_mean = torch.Tensor({0.48462227599918, 0.45624044862054, 0.40588363755159})
  img_std = torch.Tensor({0.22889466674951, 0.22446679341259, 0.22495548344775})

  -- Rescales and normalizes the image
  function preprocess(im, img_mean, img_std)
    -- rescale the image
    local im3 = image.scale(im,224,224,'bilinear')
    -- subtract imagenet mean and divide by std
    for i=1,3 do im3[i]:add(-img_mean[i]):div(img_std[i]) end
    return im3
  end

elseif typeCode == 2 then
  meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
  }

  transform = t.Compose{
     t.Scale(256),
     t.ColorNormalize(meanstd),
     t.CenterCrop(224),
  }
end  

----------------------------------------------
-- 					         Models 					      --
----------------------------------------------
------ Loading the model ------
print '==> Loading model'

if modelLang == 'torch' then
  if model_name == 'nin_nobn_final.t7' then
    net = torch.load(model_path):unpack():cuda()
  else
    net = torch.load(model_path):cuda()
  end
  
  -- model modification 
  if typeCode == 2 then
    if Ft then
      -- Remove the fully connected layer
      assert(torch.type(net:get(#net.modules)) == 'nn.Linear')
      net:remove(#net.modules)
    else
      softMaxLayer = cudnn.SoftMax():cuda()
      net:add(softMaxLayer)
    end
    
  end

elseif modelLang == 'caffe' then
  net = loadcaffe.load(prototxt, binary):cuda() 
  -- model modification 
  if FT then
    -- for VGG (use fc-6)
    net:remove()
    net:remove() 
    net:remove() 
    net:remove() 
    net:remove() 
    net:remove() 
    net:remove() 
  end
  
  
end

print(net)

----------------------------------------------
-- 			    Loading ImageNet Labels			    --
----------------------------------------------
if not Ft then
  if typeCode == 1 then
    print '==> Loading synsets'
    print 'Loads mapping from net outputs to human readable labels'
    synset_words = {}
    for line in io.lines'synset_words.txt' do table.insert(synset_words, line:sub(11)) end

  elseif typeCode == 2 then
    imagenetLabel = require './imagenet'

  end
end
----------------------------------------------
--                Input images              --
----------------------------------------------
print '==> Loading image'
im = image.load(image_path, 3, 'float')

----------------------------------------------
-- 			      Image Pre-processing			    --
----------------------------------------------
print '==> Preprocessing (normalization)'
--I = im.clone() ==> it show errors......
--I = image.scale(torch.Tensor(im:size()):copy(im):float(),224,224,'bilinear')

if modelLang == 'torch' then
  if typeCode == 1 then
    I = preprocess(im, img_mean, img_std):view(1,3,224,224):float()
  elseif typeCode == 2 then
    I = transform(im)
  end
  

elseif modelLang == 'caffe' then
  img_mean=matio.load(dir_model..'VGG_mean.mat').image_mean:transpose(3,1):float()
  im3 = image.scale(im,224,224,'bilinear')
  im3:resize(1,3,224,224)
  im3:mul(255)
  I = im3:view(1,3,224,224):float()
  I:add(-1,torch.repeatTensor(img_mean,I:size(1),1,1,1))
end

print('ImageNet')

----------------------------------------------
-- 			      Forward Propagation			      --
----------------------------------------------
if Ft then 
  -- 1. forward to get feature vectors
  print 'Obtain the final feature'

  if typeCode == 1 then
    feat = net:forward(I)
    print(feat)

  elseif typeCode == 2 then
    -- View as mini-batch of size 1
    I = I:view(1, table.unpack(I:size():totable()))
    -- Get the output of the layer before the (removed) fully connected layer
    feat = net:forward(I:cuda()):squeeze(1)
    print(feat)

  end

else
  -- 2. forward to get prediction labels
  print 'Propagate through the model, sort outputs in decreasing order and show 5 best classes'
  N = 5

  if typeCode == 1 then
    _,classes = net:forward(I:cuda()):view(-1):sort(true)
    --print(classes)
    for i=1,N do
      print('predicted class '..tostring(i)..': ', synset_words[classes[i] ])
    end  

  elseif typeCode == 2 then
    -- View as mini-batch of size 1
    batch = I:view(1, table.unpack(I:size():totable()))
    -- Get the output of the softmax
    output = net:forward(batch:cuda()):squeeze()
    -- Get the top 5 class indexes and probabilities
    probs, indexes = output:topk(N, true, true)
    print('Classes for', image_name)
    for n=1,N do
      print(probs[n], imagenetLabel[indexes[n]])
    end

  end
end

print('')
