-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Deep Learning for Video Classification

-- Load all the videos & Generate a feature matrix for each video
-- Select all the videos which have the frame numbers at least "numFrameMin"
-- No need to specify the video number
-- Follow the split sets provided in the UCF-101 website
-- Generate the name list corresponding to each video as well
-- load caffe model (We use VGG-19 now)

-- Reference:
-- Khurram Soomro, Amir Roshan Zamir and Mubarak Shah, 
-- "UCF101: A Dataset of 101 Human Action Classes From Videos in The Wild.", 
-- CRCV-TR-12-01, November, 2012. 

-- ffmpeg usage:
-- Video{
--     [path = string]          -- path to video
--     [width = number]         -- width  [default = 224]
--     [height = number]        -- height  [default = 224]
--     [zoom = number]          -- zoom factor  [default = 1]
--     [fps = number]           -- frames per second  [default = 25]
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
-- Last updated: 04/09/2016

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
require 'loadcaffe' 

matio = require 'matio'

--require 'classify_video'
--require 'gen_feature_cuda'
--require 'gen_feature'

----------------------------------------------
-- 					Functions 				--
----------------------------------------------
-- -- Rescales and normalizes the image
-- function preprocess(im, img_mean, img_std)
--   -- rescale the image
--   --local im3 = image.scale(im,224,224,'bilinear')
--   -- subtract imagenet mean and divide by std
--   for i=1,3 do im[i]:add(-img_mean[i]):div(img_std[i]) end
--   return im
-- end

-- Rescales and normalizes the image
function preprocess(im, img_mean)
  -- rescale the image
  im3 = image.scale(im,224,224,'bilinear')
  im3:resize(1,3,224,224)
  im3:mul(255)
  -- subtract imagenet mean
  I = im3:view(1,3,224,224):float()
  I:add(-1,torch.repeatTensor(img_mean,I:size(1),1,1,1))
  return I
end

----------------------------------------------
--         Input/Output information         --
----------------------------------------------
-- select the number of classes, groups & videos you want to use
numClass = 101
dimFeat = 4096

----------------------------------------------
-- 				Data paths				    --
----------------------------------------------
dirModel = './models/'
dirDatabase = '/home/cmhung/Desktop/UCF-101/'

----------------------------------------------
-- 			User-defined parameters			--
----------------------------------------------
numFrameMin = 50
numSplit = 3

-- -- Input text files --
-- textClass = 'classInd.txt'
-- trainList = 'trainlist01.txt'
-- testList = 'testlist01.txt'

-- Train/Test split
groupSplit = {}
table.insert(groupSplit, {setTr = torch.Tensor({{8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}}),
setTe = torch.Tensor({{1,2,3,4,5,6,7}})})
table.insert(groupSplit, {setTr = torch.Tensor({{1,2,3,4,5,6,7,15,16,17,18,19,20,21,22,23,24,25}}),
setTe = torch.Tensor({{8,9,10,11,12,13,14}})})
table.insert(groupSplit, {setTr = torch.Tensor({{1,2,3,4,5,6,7,8,9,10,11,12,13,14,22,23,24,25}}),
setTe = torch.Tensor({{15,16,17,18,19,20,21}})})

-- Output information --
outTrain = {}
for sp=1,numSplit do
	table.insert(outTrain, {name = 'data_UCF101_train_'..sp..'.t7'})
end

outTest = {}
for sp=1,numSplit do
	table.insert(outTest, {name = 'data_UCF101_test_'..sp..'.t7'})
end

-- mean & std --
img_mean = matio.load(dirModel..'VGG_mean.mat').image_mean:transpose(3,1):float()

-- img_mean = torch.Tensor({0.48462227599918, 0.45624044862054, 0.40588363755159})
-- img_std = torch.Tensor({0.22889466674951, 0.22446679341259, 0.22495548344775})

------ model selection ------
-- -- 1. NIN model (from Torch)
-- modelName = dirModel .. 'nin_nobn_final.t7'

---- 2. GoogleNet model (from Torch) ==> need cuda
--modelName = dirModel .. 'GoogLeNet_v2.t7'

-- -- 3. NIN model (from caffe)
-- prototxt = dirModel .. './solver.prototxt'
-- binary = dirModel .. './nin_imagenet.caffemodel'

-- 4. VGG model (from caffe)
-- prototxt = dirModel .. './VGG_ILSVRC_19_layers_deploy.prototxt'
-- binary = dirModel .. './VGG_ILSVRC_19_layers.caffemodel'
prototxt = dirModel .. './VGG_CNN_M_deploy.prototxt'
binary = dirModel .. './VGG_CNN_M.caffemodel'

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
          help='number of frames per second', default=25}
op:option{'-t', '--time', action='store', dest='seconds',
          help='length to process (in seconds)', default=2}
op:option{'-w', '--width', action='store', dest='width',
          help='resize video, width', default=320}
op:option{'-h', '--height', action='store', dest='height',
          help='resize video, height', default=240}
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
-- 					Class		        	--
----------------------------------------------
-- -- Read textClass --
-- classAll = {}
-- for l in io.lines(dirDatabase..textClass) do
-- 	local n, c = l:match '(%S+)%s+(%S+)%s'
--     table.insert(classAll, {num = tonumber(n), name = c})
-- end

nameClass = paths.dir(dirDatabase) 
numClassTotal = #nameClass -- 101 classes + "." + ".."

----------------------------------------------
-- 					Models		        	--
----------------------------------------------
------ Loading the model ------
print ' '
print '==> Loading the model...'
-- -- 1. Torch model
-- net = torch.load(modelName):unpack():float()

-- 2. Caffe model
net = loadcaffe.load(prototxt, binary)

--net:evaluate()

------ model modification ------
if opt.feat then
	-- extract fc6 of VGGnet
	net:remove()
	net:remove() 
	net:remove() 
	net:remove() 
	net:remove() 
	net:remove() 
	net:remove() 
end
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
else
	net:float()
end
print(sys.COLORS.white ..  ' ')

----------------------------------------------
-- 			Loading ImageNet labels	  		--
----------------------------------------------
if opt.pred then
	print '==> Loading the synsets...'
	print 'Loads mapping from net outputs to human readable labels'
	synset_words = {}
	for line in io.lines'synset_words.txt' do table.insert(synset_words, line:sub(11)) end
end
print ' '
--====================================================================--
--                     Run all the videos in UCF-101                  --
--====================================================================--
print '==> Processing all the videos...'

-- featMats = torch.DoubleTensor(numVideo, dimFeat, numFrameMin):zero()
-- labels = torch.DoubleTensor(numVideo):zero()

-- Load the intermediate feature data or generate a new one --
for sp=1,numSplit do
	-- Training data --
	if not paths.filep(outTrain[sp].name) then
		Tr = {} -- output
		Tr.name = {}
		Tr.path = {}
		Tr.featMats = torch.DoubleTensor()
		Tr.labels = torch.DoubleTensor()
		Tr.countVideo = 0
		Tr.countClass = 0
		Tr.c_finished = 0 -- different from countClass since there are also "." and ".."
	else
		Tr = torch.load(outTrain[sp].name) -- output
	end

	-- Testing data --
	if not paths.filep(outTest[sp].name) then
		Te = {} -- output
		Te.name = {}
		Te.path = {}
		Te.featMats = torch.DoubleTensor()
		Te.labels = torch.DoubleTensor()
		Te.countVideo = 0
		Te.countClass = 0
		Te.c_finished = 0 -- different from countClass since there are also "." and ".."
	else
		Te = torch.load(outTest[sp].name) -- output
	end
	collectgarbage()

	timerAll = torch.Timer() -- count the whole processing time

	if Tr.countClass == numClass and Te.countClass == numClass then
		print('The feature data of split '..sp..' is already in your folder!!!!!!')
	else
		for c=Tr.c_finished+1, numClassTotal do
			if nameClass[c] ~= '.' and nameClass[c] ~= '..' then
				print('Current Class: '..c..'. '..nameClass[c])
				Tr.countClass = Tr.countClass + 1
				Te.countClass = Te.countClass + 1
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
				          	--countVideo = countVideo + 1 -- save this video only when frame# >= min. frame#
				          	local featMatsVideo = torch.DoubleTensor(1,dimFeat,numFrameMin):zero()
				          	----------------------------------------------
				          	--           Process with the video         --
				          	----------------------------------------------
				          	-- if opt.pred then -- prediction (useless now)
				           --  	print '==> Begin predicting......'
				           --  	for f=1, numFrameTest do
				           --    		local inFrame = vidTensor[f]
				           --    		print('frame '..tostring(f)..': ')
				           --    		classify_video(inFrame, net, synset_words)      
				           --  	end
				          	-- end
				          	--
				          	if opt.feat then -- feature extraction
				            	--print '==> Generating the feature matrix......'
				            	for f=1, numFrameMin do
				              		local inFrame = vidTensor[f]
				              		--  
				              		--print('frame '..tostring(f)..'...')
				              		--local feat = gen_feature(inFrame, net, opt)
				              		--
				              		
				          			------ Image pre-processing ------
				              		--local I = preprocess(inFrame, img_mean, img_std):view(1,3,224,224):cuda()
				              		local I = preprocess(inFrame, img_mean)
				              		if opt.type == 'cuda' then
				              			I = I:cuda()
				              		end
				              		local feat = net:forward(I)
				              		--
				              		-- store the feature matrix for this video
							  		feat:resize(1,torch.numel(feat),1)
							  		featMatsVideo[{{},{},{f}}] = feat:double()
				            	end
				        		--
				            	----------------------------------------------
				          		--          Train/Test feature split        --
				          		----------------------------------------------
				            	-- store the feature and label for the whole dataset
				            	local i,j = string.find(videoName,'_g') -- find the location of the group info in the string
				            	local videoGroup = tonumber(string.sub(videoName,j+1,j+2)) -- get the group#
				            	local videoPathLocal = nameClass[c]..'/'..videoName
				            	
				            	if groupSplit[sp].setTe:eq(videoGroup):sum() == 0 then -- training data
				            		Tr.countVideo = Tr.countVideo + 1
				            		Tr.name[Tr.countVideo] = videoName
				            		Tr.path[Tr.countVideo] = videoPathLocal
				            		if Tr.countVideo == 1 then -- the first video
				            			Tr.featMats = featMatsVideo
				            			Tr.labels = torch.DoubleTensor(1):fill(Tr.countClass)
				            		else 					-- from the second or the following videos
				            			Tr.featMats = torch.cat(Tr.featMats,featMatsVideo,1)
				            			Tr.labels = torch.cat(Tr.labels,torch.DoubleTensor(1):fill(Tr.countClass),1)
				            		end			            	
				            	else -- testing data
				            		Te.countVideo = Te.countVideo + 1
				            		Te.name[Tr.countVideo] = videoName
					            	Te.path[Tr.countVideo] = videoPathLocal
				            		if Te.countVideo == 1 then -- the first video
				            			Te.featMats = featMatsVideo
				            			Te.labels = torch.DoubleTensor(1):fill(Te.countClass)
				            		else 					-- from the second or the following videos
				            			Te.featMats = torch.cat(Te.featMats,featMatsVideo,1)
				            			Te.labels = torch.cat(Te.labels,torch.DoubleTensor(1):fill(Te.countClass),1)
				            		end			            	
				            	end
				            	
				          	end
				        end
			      	end
			      	collectgarbage()
			    end
			    
				Tr.c_finished = c -- save the index
				Te.c_finished = c -- save the index
				print('Split: '..sp)
				print('Finished class#: '..Tr.countClass)
				print('Generated training data#: '..Tr.countVideo)
				print('Generated testing data#: '..Te.countVideo)
			  	print('The elapsed time for the class '..nameClass[c]..': ' .. timerClass:time().real .. ' seconds')
			  	torch.save(outTrain[sp].name, Tr)
			  	torch.save(outTest[sp].name, Te)

			  	collectgarbage()
			  	print(' ')
			end
		end
	end

	print('The total elapsed time in the split '..sp..': ' .. timerAll:time().real .. ' seconds')
	print('The total training class numbers in the split'..sp..': ' .. Tr.countClass)
	print('The total training video numbers in the split'..sp..': ' .. Tr.countVideo)
	print('The total testing class numbers in the split'..sp..': ' .. Te.countClass)
	print('The total testing video numbers in the split'..sp..': ' .. Te.countVideo)
	print ' '

	Tr = nil
	Te = nil
	collectgarbage()
end
