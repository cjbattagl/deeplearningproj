-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Video Classification

-- Load one video & generate a feature matrix for this video
-- Select all the videos which have the frame numbers at least "numFrameMin"
-- No need to 

-- ffmpeg usage:
-- Video{
--     [path = string]          -- path to video
--     [width = number]         -- width  [default = 224]
--     [height = number]        -- height  [default = 224]
--     [zoom = number]          -- zoom factor  [default = 1]
--     [fps = number]           -- frames per second  [default = 30]
--     [length = number]        -- length, in seconds  [default = 2]
--     [seek = number]          -- seek to pos. in seconds  [default = 0]
--     [channel = number]       -- video channel  [default = 0]
--     [load = boolean]         -- loads frames after conversion  [default = true]
--     [delete = boolean]       -- clears (rm) frames after load  [default = true]
--     [encoding = string]      -- format of dumped frames  [default = png]
--     [tensor = torch.Tensor]  -- provide a packed tensor (NxCxHxW or NxHxW), that bypasses path
--     [destFolder = string]    -- destination folder  [default = out_frames]
--     [silent = boolean]       -- suppress output  [default = false]
-- }

-- author: Min-Hung Chen
-- contact: cmhungsteve@gatech.edu
-- Last updated: 04/05/2016

--#!/usr/bin/env torch

require 'xlua'
require 'torch'
--require 'qt'
--require 'qtwidget'
require 'imgraph'
require 'nnx'
require 'ffmpeg'
require 'image'
require 'nn'

require 'classify_video'
--require 'gen_feature_cuda'
--require 'gen_feature'

----------------------------------------------
-- 					Functions 				--
----------------------------------------------
-- Rescales and normalizes the image
function preprocess(im, img_mean, img_std)
  -- rescale the image
  --local im3 = image.scale(im,224,224,'bilinear')
  -- subtract imagenet mean and divide by std
  for i=1,3 do im[i]:add(-img_mean[i]):div(img_std[i]) end
  return im
end

----------------------------------------------
--         Input/Output information         --
----------------------------------------------
-- select the number of classes, groups & videos you want to use
numClass = 101
dimFeat = 1024

----------------------------------------------
-- 				Data paths				    --
----------------------------------------------
dirModel = './models/'
dirDatabase = '/home/cmhung/Desktop/UCF-101/'

nameClass = paths.dir(dirDatabase) 
numClassTotal = #nameClass -- 101 classes + "." + ".."

----------------------------------------------
-- 			User-defined parameters			--
----------------------------------------------
numFrameMin = 50
nameOutput = 'feat_label_UCF101_2.t7'

-- mean & std --
img_mean = torch.Tensor({0.48462227599918, 0.45624044862054, 0.40588363755159})
img_std = torch.Tensor({0.22889466674951, 0.22446679341259, 0.22495548344775})

------ model selection ------
-- 1. NIN model (from Torch)
modelName = dirModel .. 'nin_nobn_final.t7'

---- 2. GoogleNet model (from Torch) ==> need cuda
--modelName = dirModel .. 'GoogLeNet_v2.t7'

-- -- 3. NIN model (from caffe)
-- prototxt = dirModel .. './solver.prototxt'
-- binary = dirModel .. './nin_imagenet.caffemodel'

-- -- 4. VGG model (from caffe)
-- prototxt = dirModel .. './VGG_ILSVRC_19_layers_deploy.prototxt'
-- binary = dirModel .. './VGG_ILSVRC_19_layers.caffemodel'

-- ------ input image ------
-- --image_name = 'Goldfish3.jpg'
-- image_name = 'frame-000001.png'
--im = image.load(dir_image .. image_name)

----------------
-- parse args --
----------------
op = xlua.OptionParser('%prog [options]')
-- op:option{'-c', '--camera', action='store', dest='camidx',
--           help='camera index: /dev/videoIDX (if no video given)', 
--           default=0}
-- op:option{'-v', '--video', action='store', dest='video',
--           help='video file to process', default=videoPath}
op:option{'-f', '--fps', action='store', dest='fps',
          help='number of frames per second', default=30}
op:option{'-t', '--time', action='store', dest='seconds',
          help='length to process (in seconds)', default=2}
op:option{'-w', '--width', action='store', dest='width',
          help='resize video, width', default=224}
op:option{'-h', '--height', action='store', dest='height',
          help='resize video, height', default=224}
op:option{'-z', '--zoom', action='store', dest='zoom',
          help='display zoom', default=1}
op:option{'-fe', '--feat', action='store', dest='feat',
          help='option for generating features', default=true}
op:option{'-pr', '--pred', action='store', dest='pred',
          help='option for prediction', default=false}
op:option{'-p', '--type', action='store', dest='type',
          help='option for CPU/GPU', default='cuda'}
op:option{'-i', '--devid', action='store', dest='devid',
          help='device ID (if using CUDA)', default=1}      
opt,args = op:parse()

----------------------------------------------
-- 					models		        	--
----------------------------------------------
------ Loading the model ------
print ' '
print '==> Loading the model...'
-- 1. Torch model
net = torch.load(modelName):unpack():float()

-- -- 2. Caffe model
-- net = loadcaffe.load(prototxt, binary):float()

net:evaluate()

------ model modification ------
net:remove(30) -- process the model

print(net)
print ' '

----------------------------------------------
--  		       GPU option	 	        --
----------------------------------------------
if opt.type == 'cuda' then
  print(sys.COLORS.red ..  '==> switching to CUDA')
  require 'cunn'
  net:cuda()
  cutorch.setDevice(opt.devid)
  print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
  --images = images:cuda()
-- else
--   net:float()
end
print(sys.COLORS.white ..  ' ')

----------------------------------------------
-- 			    Loading ImageNet Labels			    --
----------------------------------------------
print '==> Loading the synsets...'
print 'Loads mapping from net outputs to human readable labels'
synset_words = {}
for line in io.lines'synset_words.txt' do table.insert(synset_words, line:sub(11)) end

print ' '
--====================================================================--
--                     Run all the videos in UCF-101                  --
--====================================================================--
print '==> Processing all the videos...'

-- featMats = torch.DoubleTensor(numVideo, dimFeat, numFrameMin):zero()
-- labels = torch.DoubleTensor(numVideo):zero()

-- Load the intermediate feature data or generate a new one --
if not paths.filep(nameOutput) then
	--out = {} -- output
	featMats = torch.DoubleTensor()
	labels = torch.DoubleTensor()
	countVideo = 0
	countClass = 0
	c_finished = 0 -- different from countClass since there are also "." and ".."
else
	dataTemp = torch.load(nameOutput) -- output
	featMats = dataTemp.featMats
	labels = dataTemp.labels
	countVideo = dataTemp.numVideo
	countClass = dataTemp.numClass
	c_finished = dataTemp.c_finished
	dataTemp = nil
end
collectgarbage()

timerAll = torch.Timer() -- count the whole processing time

if countClass == numClass then
	print('The feature data is already in your folder!!!!!!')
else
	for c=c_finished+1, numClassTotal do
		if nameClass[c] ~= '.' and nameClass[c] ~= '..' then
			print('Current Class: '..c..'. '..nameClass[c])
			countClass = countClass + 1
		  	------ Data paths ------
		  	local dirClass = dirDatabase..nameClass[c]..'/' 
		  	local nameSubVideo = paths.dir(dirClass)
		  	local numSubVideoTotal = #nameSubVideo -- videos + '.' + '..'
		  	--
		  	local timerClass = torch.Timer() -- count the processing time for one class
		  	--
		    for sv=1, numSubVideoTotal do
		      	--------------------
		      	-- Load the video --
		      	--------------------  
		      	if nameSubVideo[sv] ~= '.' and nameSubVideo[sv] ~= '..' then
		        	local videoName = nameSubVideo[sv]
		        	local videoPath = dirClass..videoName
		        	--
		        	--print('==> Loading the video: '..videoName)
		        	-- TODO --
		        	-- now:     fixed frame rate
		        	-- future:  fixed frame #
		        	--
		        	local video = ffmpeg.Video{path=videoPath, width=opt.width, height=opt.height, fps=opt.fps, length=opt.seconds, delete=true, destFolder='out_frames',silent=true}
		        	--
		        	-- --video:play{} -- play the video
		        	local vidTensor = video:totensor{} -- read the whole video & turn it into a 4D tensor
			        --
			        ------ Video prarmeters ------
			        local numFrame = vidTensor:size(1)
			        --
			        if numFrame >= numFrameMin then
			          	countVideo = countVideo + 1 -- save this video only when frame# >= min. frame#
			          	local featMatsVideo = torch.DoubleTensor(1,dimFeat,numFrameMin):zero()
			          	----------------------------------------------
			          	--           Process with the video         --
			          	----------------------------------------------
			          	if opt.pred then -- prediction (useless now)
			            	print '==> Begin predicting......'
			            	for f=1, numFrameTest do
			              		local inFrame = vidTensor[f]
			              		print('frame '..tostring(f)..': ')
			              		classify_video(inFrame, net, synset_words)      
			            	end
			          	end
			          	--
			          	if opt.feat then -- feature extraction
			            	--print '==> Generating the feature matrix......'
			            	for f=1, numFrameMin do
			              		local inFrame = vidTensor[f]
			              		--  
			              		--print('frame '..tostring(f)..'...')
			              		--local feat = gen_feature(inFrame, net, opt)
			              		--
			              		local I = preprocess(inFrame, img_mean, img_std):view(1,3,224,224):cuda()
			              		local feat = net:forward(I)
			              		--
			              		-- store the feature matrix for this video
						  		feat:resize(1,torch.numel(feat),1)
						  		featMatsVideo[{{},{},{f}}] = feat:double()
			            	end
			        		--
			            	-- store the feature and label for the whole dataset
			            	if countVideo == 1 then -- the first video
			            		featMats = featMatsVideo
			            		labels = torch.DoubleTensor(1):fill(countClass)
			            	else 					-- from the second or the following videos
			            		featMats = torch.cat(featMats,featMatsVideo,1)
			            		labels = torch.cat(labels,torch.DoubleTensor(1):fill(countClass),1)
			            	end
			            	-- print(videoName..' ==> feature dimension: '..
			            	--   '('..tostring(featMats:size(2))..', '..tostring(featMats:size(3))..'), '..
			            	--   'label: '..nameClass[c])
			          	end
			          	--print(video)
			        end
		      	end
		    end
		    out = {}
		    out.featMats = featMats
			out.labels = labels
			out.numVideo = countVideo
			out.numClass = countClass
			out.c_finished = c -- save the index
			print(out)
		  	print('The elapsed time for the class '..nameClass[c]..': ' .. timerClass:time().real .. ' seconds')
		  	torch.save(nameOutput, out)
		  	out = nil
		  	collectgarbage()
		  	print(' ')
		end
	end
end

print('The total elapsed time: ' .. timerAll:time().real .. ' seconds')
print('The total saved class numbers: ' .. countClass)
print('The total saved video numbers: ' .. countVideo)
print ' '

-- ------------------------------
-- --           Output         --
-- ------------------------------
-- out = {}
-- out.featMats = featMats
-- out.labels = labels
-- out.numVideo = countVideo
-- print(out)

-- torch.save('feat_label_UCF101_2.t7', out)
-- print ' '
