-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Video Classification

-- This script implements a test procedure, to report accuracy on the test data

-- TODO:
-- 1. 
-- 2. 

-- modified by Hao Yan
-- contact: yanhao@gatech.edu
-- Last updated: 05/02/2016


require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'image'

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> defining some tools')

-- model:
local t = require 'model'
local model = t.model
local loss = t.loss
local nframe = t.nframe
local nfeature = t.nfeature

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes) 

-- Logger:
local testLogger = optim.Logger(paths.concat(opt.save,'testnew.log')) 
-- Batch test: 
local inputs = torch.Tensor(opt.batchSize, nfeature, nframe) -- get size from data
local targets = torch.Tensor(opt.batchSize)
local predarr = torch.Tensor(opt.batchSize,101,nframe)
-- local preds = torch.Tensor(opt.batchSize,101)
if opt.type == 'cuda' then 
   inputs = inputs:cuda()
   targets = targets:cuda()
end

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> defining test procedure')

-- test function
function test(testData)
   -- local vars
   local time = sys.clock()

   -- test over test data
   print(sys.COLORS.red .. '==> testing on test set:')
   for t = 1,testData:size(),opt.batchSize do
      -- disp progress
	  collectgarbage()
      xlua.progress(t, testData:size())

      -- batch fits?
      if (t + opt.batchSize - 1) > testData:size() then
         break
      end

      -- create mini batch
      local idx = 1
      for i = t,t+opt.batchSize-1 do
         inputs[idx] = testData.data[{i,{},{1,nframe}}]
         targets[idx] = testData.labels[i]
         idx = idx + 1
      end
      -- test sample
	  for i = 1,nframe do
       	 predarr[{{},{},i}] = model:forward(inputs[{{},{},i}])
	  end
      -- Voting Added 	  
	  preds = torch.mode(predarr)
      -- confusion
--	  print(targets[i],preds[{i,{},1}])
      for i = 1,opt.batchSize do
         confusion:add(preds[{i,{},1}], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end
   confusion:zero()
   
end

-- Export:
return test

