-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Video Classification

-- Train a MLP on UCF11

-- TODO:
-- 1. cross-validation

-- modified by Hao Yan
-- contact: cmhungsteve@gatech.edu
-- Last updated: 04/03/2016


require 'pl'
require 'trepl'
require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> processing options')

opt = lapp[[
   -r,--learningRate       (default 5e-3)        learning rate
   -d,--learningRateDecay  (default 1e-6)        learning rate decay (in # samples)
   -w,--weightDecay        (default 1e-5)        L2 penalty on the weights
   -m,--momentum           (default 0.1)         momentum
   -d,--dropout            (default 0.5)         dropout amount
   -b,--batchSize          (default 256)         batch size
   -t,--threads            (default 1)           number of threads
   -p,--type               (default float)       float or cuda
   -i,--devid              (default 1)           device ID (if using CUDA)
   -s,--size               (default small)       dataset: small or full or extra
   -o,--save               (default results)     save directory
      --patches            (default all)         percentage of samples to use for testing'
      --visualize          (default false)        visualize dataset
      --model              (default Linear)         network model
      --optMethod          (default sgd)         optimization method
      --plot               (default false)       plot the training and test accuracies
      --dataAugment        (default false)       Enable dataAugmentation while training
]]
-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')
-- type:
if opt.type == 'cuda' then
   print(sys.COLORS.red ..  '==> switching to CUDA')
   require 'cunn'
   cutorch.setDevice(opt.devid)
   print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> load modules')

local data  = require 'data'
local train = require 'train'
local test  = require 'test'
--
------------------------------------------------------------------------
print(sys.COLORS.red .. '==> training!')
--

--for i=1,50 do 
while true do
    train(data.trainData)
    test(data.testData)
end

