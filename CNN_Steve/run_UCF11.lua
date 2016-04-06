-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Deep Learning for Video Classification

-- Load one video & generate a feature matrix for this video

-- TODO:
-- 1. frame#: I used the frame# of a short video now. We can also try to fix the frame#
--            and variable frame rate for different videos.
-- 2. video selection: In each group, I chose the first 4 videos now. We can either randomly 
--                     choose 4 videos, or use all the videos.

-- ffmpeg usage:
-- Video{
--     [path = string]          -- path to video
--     [width = number]         -- width  [default = 224]
--     [height = number]        -- height  [default = 224]
--     [zoom = number]          -- zoom factor  [default = 1]
--     [fps = number]           -- frames per second  [default = 30]
--     [length = number]        -- length, in seconds  [default = 10]
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
-- Last updated: 04/01/2016

--#!/usr/bin/env torch

require 'xlua'
require 'torch'
require 'qt'
require 'qtwidget'
require 'imgraph'
require 'nnx'
require 'ffmpeg'

require 'classify_video'
require 'gen_feature'

----------------------------------------------
--         Input/Output information         --
----------------------------------------------
-- select the number of classes, groups & videos you want to use
numClass = 11
numGroup = 25
numSubVideo = 4
numVideo = numClass*numGroup*numSubVideo
dimFeat = 1024

----------------------------------------------
-- 					        Data paths				      --
----------------------------------------------
dirModel = './models/'
dirDatabase = '../Dataset/UCF11_updated_mpg/'
nameClass = paths.dir(dirDatabase) -- nameClass[3] ~ nameClass[13] are the classes we want
numClassTotal = #nameClass - 2 -- 11 classes

----------------------------------------------
-- 			      User-defined parameters			  --
----------------------------------------------
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
opt,args = op:parse()

-- ----------------------------------------------
-- --          Extract video parameters        --
-- ----------------------------------------------
-- ------ Search for the shortest video ------
-- numFrameMin = 30000 -- set a really large number

-- for c=1, numClassTotal do
--   ------ Data paths ------
--   local dirClass = dirDatabase..nameClass[c+2]..'/' -- need to +2, since the first two are '.' & '..'
--   local nameGroup = paths.dir(dirClass)
--   local numGroupTotal = #nameGroup - 3

--   local timerClass = torch.Timer() -- count the processing time for one class

--   local numFrameMinClass = 30000

--   for g=1, numGroupTotal do
--     ------ Data paths ------
--     local dirGroup = dirClass..nameGroup[g+3]..'/' -- need to +2, since the first three are '.' & '..' & 'Annotation'
--     local nameSubVideo = paths.dir(dirGroup)
--     local numSubVideoTotal = #nameSubVideo - 2

--     for sv=1, numSubVideoTotal do
--       --------------------
--       -- Load the video --
--       --------------------  
--       local videoName = nameSubVideo[sv+2]
--       local videoPath = dirGroup..videoName

--       --print('==> Loading the video: '..videoName)
--       local video = ffmpeg.Video{path=videoPath, width=opt.width, height=opt.height, 
--                              fps=opt.fps, length=opt.seconds, delete=true, 
--                              destFolder='out_frames',silent=true}

--       -- --video:play{} -- play the video
--       local vidTensor = video:totensor{} -- read the whole video & turn it into a 4D tensor

--       ------ Video prarmeters ------
--       local numFrame  = vidTensor:size(1)

--       if numFrame < numFrameMinClass then
--         numFrameMinClass = numFrame
--         --dirMinVideo = dirGroup
--         --videoMinName = videoName
--         videoMinPathClass = videoPath
--       end

--       if numFrame < numFrameMin then
--         numFrameMin = numFrame
--         --dirMinVideo = dirGroup
--         --videoMinName = videoName
--         videoMinPath = videoPath
--       end

--         --print(video)
--     end
--   end

--   print('The serching time for the class '..nameClass[c+2]..': ' .. timerClass:time().real .. ' seconds')
--   print('The full path of the shortest video in the class '..nameClass[c+2]..': '..videoMinPathClass)
--   print('The smallest frame # of the class '..nameClass[c+2]..': '.. numFrameMinClass)

-- end

-- print(' ')
-- print('The full path of the shortest video in the dataset: '..videoMinPath)
-- print('The smallest frame #: '.. numFrameMin)

----------------------------------------------
--          Extract video parameters        --
----------------------------------------------
dirTestVideo = dirDatabase..'volleyball_spiking/v_spiking_14/'
videoTestName = 'v_spiking_14_02' -- a short video
videoTestPath = dirTestVideo..videoTestName..'.mpg'
videoTest = ffmpeg.Video{path=videoTestPath, width=opt.width, height=opt.height, 
                             fps=opt.fps, length=opt.seconds, delete=true, 
                             destFolder='out_frames',silent=true}

--video:play{} -- play the video
vidTensorTest = videoTest:totensor{} -- read the whole video & turn it into a 4D tensor

numFrameTest  = vidTensorTest:size(1)
numChnTest    = vidTensorTest:size(2)
vidHeightTest = vidTensorTest:size(3)
vidWidthTest  = vidTensorTest:size(4)

----------------------------------------------
-- 					        Models 					        --
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
-- ----------------------------------------------
-- -- 					       GPU option				       --
-- ----------------------------------------------
-- if arg[1] == 'cuda' then
--   net:cuda()
--   images = images:cuda()
-- else
--   net:float()
-- end

----------------------------------------------
-- 			    Loading ImageNet Labels			    --
----------------------------------------------
print '==> Loading the synsets...'
print 'Loads mapping from net outputs to human readable labels'
synset_words = {}
for line in io.lines'synset_words.txt' do table.insert(synset_words, line:sub(11)) end

print ' '
--====================================================================--
--                     Run all the videos in UCF-11                   --
--====================================================================--
print '==> Processing all the videos...'

------ output features & labels ------
featMats = torch.DoubleTensor(numVideo, dimFeat, numFrameTest):zero()
labels = torch.DoubleTensor(numVideo):zero()
countVideo = 0

timerAll = torch.Timer() -- count the whole processing time
for c=1, numClass do
  ------ Data paths ------
  local dirClass = dirDatabase..nameClass[c+2]..'/' -- need to +2, since the first two are '.' & '..'
  local nameGroup = paths.dir(dirClass)
  local numGroupTotal = #nameGroup - 3

  local timerClass = torch.Timer() -- count the processing time for one class

  for g=1, numGroupTotal do
    ------ Data paths ------
    local dirGroup = dirClass..nameGroup[g+3]..'/' -- need to +2, since the first three are '.' & '..' & 'Annotation'
    local nameSubVideo = paths.dir(dirGroup)
    local numSubVideoTotal = #nameSubVideo - 2
    local countSubVideo = 0 -- we only select 4 videos, so we need to count the video #

    for sv=1, numSubVideoTotal do
      --------------------
      -- Load the video --
      --------------------  
      if countSubVideo < numSubVideo then
        -- TODO --
        -- now:     choose the first 4 videos
        -- future:  probably randomly choose 4 videos

        local videoName = nameSubVideo[sv+2]
        local videoPath = dirGroup..videoName

        --print('==> Loading the video: '..videoName)
        -- TODO --
        -- now:     fixed frame rate
        -- future:  fixed frame #

        local video = ffmpeg.Video{path=videoPath, width=opt.width, height=opt.height, 
                               fps=opt.fps, length=opt.seconds, delete=true, 
                               destFolder='out_frames',silent=true}

        -- --video:play{} -- play the video
        local vidTensor = video:totensor{} -- read the whole video & turn it into a 4D tensor

        ------ Video prarmeters ------
        local numFrame  = vidTensor:size(1)
        -- local numChn     = vidTensor:size(2)
        -- local vidHeight  = vidTensor:size(3)
        -- local vidWidthT  = vidTensor:size(4)

        --if numFrame >= numFrameMin then
        if numFrame >= numFrameTest then
        
          countSubVideo = countSubVideo + 1
          countVideo = countVideo + 1
          ----------------------------------------------
          --           Process with the video         --
          ----------------------------------------------
          if opt.pred then
            print '==> Begin predicting......'
            for f=1, numFrameTest do
              local inFrame = vidTensor[f]
              print('frame '..tostring(f)..': ')
              classify_video(inFrame, net, synset_words)
              
            end
          end

          if opt.feat then
            --print '==> Generating the feature matrix......'
            for f=1, numFrameTest do
              local inFrame = vidTensor[f]
              --print('frame '..tostring(f)..'...')
              local feat = gen_feature(inFrame, net, synset_words)
              featMats[{{countVideo},{},{f}}] = feat
            end
            labels[countVideo] = c

            print(videoName..' ==> feature dimension: '..
              '('..tostring(featMats:size(2))..', '..tostring(featMats:size(3))..'), '..
              'label: '..nameClass[c+2])
          end

          --print(video)
        end
      end
    end
  end

  print('The elapsed time for the class '..nameClass[c+2]..': ' .. timerClass:time().real .. ' seconds')

end

print('The total elapsed time: ' .. timerAll:time().real .. ' seconds')
print ' '
------------------------------
--           Output         --
------------------------------
out = {}
out.featMats = featMats
out.labels = labels
print(out)
torch.save('feat_label_UCF11.t7', out)
print ' '
----------------------------------------------
--              Reference Codes             --
----------------------------------------------
-- -- setup GUI (external UI file)
-- --if not win or not widget then 
-- --   win = qtwidget.newwindow(opt.width*2*opt.zoom, opt.height*opt.zoom,
-- --                            'A simple mst-based cartoonizer')
-- --end

-- -- gaussian (a gaussian, really, is always useful)
-- gaussian = image.gaussian(3)

-- -- process function
-- function process()
--    -- (1) grab frame
--    frame = video:forward()

--    -- (2) compute affinity graph on input image
--    frame_smoothed = image.convolve(frame, gaussian, 'same')
--    graph = imgraph.graph(frame_smoothed)

--    -- (3) cut graph using min-spanning tree
--    mstsegm = imgraph.segmentmst(graph, 2, 20)

--    -- (4) pool the input frame into the segmentation
--    cartoon = imgraph.histpooling(frame:clone(), mstsegm)
-- end

-- -- display function
-- function display()
--    -- display input image + result
-- --   image.display{image={frame,cartoon}, win=win, zoom=opt.zoom}
-- end

-- setup gui
--timer = qt.QTimer()
--timer.interval = 10
--timer.singleShot = true
--qt.connect(timer,
--           'timeout()',
--           function()
--              process()
--              win:gbegin()
--              win:showpage()
--              display()
--              win:gend()
--              timer:start()
--           end)
--timer:start()
