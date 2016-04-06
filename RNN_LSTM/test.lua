----------------------------------------------------------------
-- Georgia Tech 2016 Spring
-- Deep Learning for Perception
-- Final Project: LRCN model for Video Classification
--
-- 
-- This is a testing code for implementing the RNN model with LSTM 
-- written by Chih-Yao Ma. 
-- 
-- The code will take feature vectors (from CNN model) from contiguous 
-- frames and train against the ground truth, i.e. the labeling of video classes. 
-- 
-- Contact: Chih-Yao Ma at <cyma@gatech.edu>
----------------------------------------------------------------

require 'torch'
require 'sys'
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'

print(sys.COLORS.red .. '==> defining some tools')

-- model:
local m = require 'model'
local model = m.model
local criterion = m.criterion

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes) 

-- Logger:
local testLogger = optim.Logger(paths.concat(opt.save,'test.log'))

-- Batch test:
local inputs = torch.Tensor(opt.batchSize, TrainData:size(2), TrainData:size(3))
local targets = torch.Tensor(opt.batchSize)

if opt.cuda == true then
   inputs = inputs:cuda()
   targets = targets:cuda()
end


-- test function
function test(testData)

	-- local vars
	local time = sys.clock() 

	-- test over test data
	print(sys.COLORS.red .. '==> testing on test set:')

	for t = 1,TestData:size(1),opt.batchSize do
	  -- disp progress
	  xlua.progress(t, TestData:size(1))

	  -- batch fits?
	  if (t + opt.batchSize - 1) > TestData:size(1) then
	  	break
	  end

	  -- create mini batch
	  local idx = 1
	  for i = t,t+opt.batchSize-1 do
	  	inputs[idx] = TestData[i]
	  	targets[idx] = TestTarget[i]
	  	idx = idx + 1
	  end

	  -- test sample
	  local preds = model:forward(inputs)

	  -- confusion
	  for i = 1,opt.batchSize do
	  	confusion:add(preds[i], targets[i])
	  end
	end

	-- timing
	time = sys.clock() - time
	time = time / TestData:size(1)
	print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

  	-- print confusion matrix
  	print(confusion)

	if confusion.totalValid * 100 >= bestAcc then
		bestAcc = confusion.totalValid * 100
	end
	print(sys.COLORS.red .. '==> Best testing accuracy = ' .. bestAcc .. '%')

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