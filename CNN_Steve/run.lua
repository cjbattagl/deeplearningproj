-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Video Classification


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
-- Last updated: 03/20/2016

--#!/usr/bin/env torch

require 'xlua'
require 'torch'
require 'qt'
require 'qtwidget'
require 'imgraph'
require 'nnx'

require 'classify_video'

----------------------------------------------
-- 			Download the video				--
----------------------------------------------
print '==> Downloading the video......'
videoUrl = 'https://www.dropbox.com/s/8kkilo6vkstwcnf/test.mpg'
-- videoUrl = 'https://www.dropbox.com/s/suyeymlof3gx6lz/v_riding_20_02.mpg'
-- videoUrl = 'https://www.dropbox.com/s/hsnaho7hcgytydj/v_spiking_20_02.mpg'
-- videoUrl = 'https://www.dropbox.com/s/3ixls6wqch0cnfv/v_swing_20_02.mpg'
-- videoUrl = 'https://www.dropbox.com/s/pzxbnaa2v8pjuj4/v_shooting_20_02.mpg'
-- videoUrl = 'https://www.dropbox.com/s/876kj0d4rcbdddj/v_tennis_20_02.mpg'
-- videoUrl = 'https://www.dropbox.com/s/v6iowtofkp0zqgr/v_walk_dog_20_02.mpg'
-- videoUrl = 'https://www.dropbox.com/s/os28z9ofxypq24t/v_biking_20_02.mpg'
-- videoUrl = 'https://www.dropbox.com/s/0nsz5qt7e0iq6ap/v_diving_20_02.mpg'
-- videoUrl = 'https://www.dropbox.com/s/sy07cjrm24v093w/v_golf_20_02.mpg'
-- videoUrl = 'https://www.dropbox.com/s/71zf2q40mzoames/v_juggle_20_02.mpg'
-- videoUrl = 'https://www.dropbox.com/s/nsetqtey8fgssg3/v_jumping_20_02.mpg'

videoName = paths.basename(videoUrl)
if not paths.filep(videoName) then os.execute('wget '..videoUrl) end

----------------
-- parse args --
----------------
op = xlua.OptionParser('%prog [options]')
op:option{'-c', '--camera', action='store', dest='camidx',
          help='camera index: /dev/videoIDX (if no video given)', 
          default=0}
op:option{'-v', '--video', action='store', dest='video',
          help='video file to process', default=videoName}
op:option{'-f', '--fps', action='store', dest='fps',
          help='number of frames per second', default=30}
op:option{'-t', '--time', action='store', dest='seconds',
          help='length to process (in seconds)', default=10}
op:option{'-w', '--width', action='store', dest='width',
          help='resize video, width', default=224}
op:option{'-h', '--height', action='store', dest='height',
          help='resize video, height', default=224}
op:option{'-z', '--zoom', action='store', dest='zoom',
          help='display zoom', default=1}
opt,args = op:parse()

-- data path --
dir_model = './models/'

----------------------------------------------
-- 			User-defined parameters			--
----------------------------------------------
------ model selection ------
-- 1. NIN model (from Torch)
model_name = dir_model .. 'nin_nobn_final.t7'

---- 2. GoogleNet model (from Torch) ==> need cuda
--model_name = dir_model .. 'GoogLeNet_v2.t7'

-- -- 3. NIN model (from caffe)
-- prototxt = dir_model .. './solver.prototxt'
-- binary = dir_model .. './nin_imagenet.caffemodel'

-- -- 4. VGG model (from caffe)
-- prototxt = dir_model .. './VGG_ILSVRC_19_layers_deploy.prototxt'
-- binary = dir_model .. './VGG_ILSVRC_19_layers.caffemodel'

-- ------ input image ------
-- --image_name = 'Goldfish3.jpg'
-- image_name = 'frame-000001.png'
--im = image.load(dir_image .. image_name)

----------------------------------------------
-- 					Models 					--
----------------------------------------------
------ Loading the model ------
--print '==> Loading model'
-- 1. Torch model
net = torch.load(model_name):unpack():float()

-- -- 2. Caffe model
-- net = loadcaffe.load(prototxt, binary):float()

net:evaluate()

-- model modification
--net:remove(30)
--net:add(nn.View(-1))

print(net)

-- ----------------------------------------------
-- -- 					GPU option				--
-- ----------------------------------------------
-- if arg[1] == 'cuda' then
--   net:cuda()
--   images = images:cuda()
-- else
--   net:float()
-- end

----------------------------------------------
-- 			Loading ImageNet Labels			--
----------------------------------------------
--print '==> Loading synsets'
--print 'Loads mapping from net outputs to human readable labels'
synset_words = {}
for line in io.lines'synset_words.txt' do table.insert(synset_words, line:sub(11)) end

--------------------
-- Load the video --
--------------------
print '==> Loading the video......'
if not opt.video then
   -- load camera
   require 'camera'
   video = image.Camera(opt.camidx, opt.width, opt.height)
else
   -- load video
   require 'ffmpeg'
   video = ffmpeg.Video{path=opt.video, width=opt.width, height=opt.height, 
                       fps=opt.fps, length=opt.seconds, delete=true, 
                       destFolder='out_frames',silent=false}
   -- video = ffmpeg.Video{path='test.mpg', width=224, height=224, fps=30, 
   --                      length=10, delete=false, destFolder='out_frame',
   --                      silent=false} -- use for debugging
end

--video:play{} -- play the video
vidTensor = video:totensor{} -- read the whole video & turn it into a 4D tensor
----------------------------------------------
--              Video prarmeters            --
----------------------------------------------
numFrame =  vidTensor:size(1)
numChn =    vidTensor:size(2)
vidHeight = vidTensor:size(3)
vidWidth =  vidTensor:size(4)

----------------------------------------------
--           Process with the video         --
----------------------------------------------
print '==> Begin predicting......'
for i=1, numFrame do
  local inFrame = vidTensor[i]
  print('frame '..tostring(i)..': ')
  classify_video(inFrame, net, synset_words)
end

--print(frame)
--print(video)

--classify_video(im, net, synset_words)

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
