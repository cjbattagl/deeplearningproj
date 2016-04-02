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

require 'dp'
require 'rnn'
require 'sys'
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'

version = 1
c = sys.COLORS

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Language Model on PennTreeBank dataset using RNN or LSTM or GRU')
cmd:text('Example:')
cmd:text("recurrent-language-model.lua --cuda --useDevice 2 --progress --zeroFirst --cutoffNorm 4 --opt.rho 10")
cmd:text('Options:')
cmd:option('--startLearningRate', 0.05, 'learning rate at t=0')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--learningRateDecay', 1e-7, 'learningRateDecay')
cmd:option('--saturateEpoch', 400, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--weightDecay', 1e-5, 'weightDecay')
cmd:option('--maxOutNorm', -1, 'max l2-norm of each layer\'s output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('--batchSize', 8, 'number of examples per batch') -- how many examples per training 
cmd:option('--cuda', true, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 1000, 'maximum number of epochs to run')
cmd:option('--maxTries', 50, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--progress', true, 'print progress bar')
cmd:option('--silent', false, 'don\'t print anything to stdout')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')

-- recurrent layer 
cmd:option('--lstm', true, 'use Long Short Term Memory (nn.LSTM instead of nn.Recurrent)')
cmd:option('--gru', false, 'use Gated Recurrent Units (nn.GRU instead of nn.Recurrent)')
cmd:option('--rho', 5, 'back-propagate through time (BPTT) for opt.rho time-steps')
cmd:option('--hiddenSize', '{1024, 100}', 'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs/GRUs are stacked')
cmd:option('--zeroFirst', false, 'first step will forward zero through recurrence (i.e. add bias of recurrence). As opposed to learning bias specifically for first step.')
cmd:option('--dropout', false, 'apply dropout after each recurrent layer')
cmd:option('--dropoutProb', 0.5, 'probability of zeroing a neuron (dropout probability)')

-- data
cmd:option('--trainEpochSize', -1, 'number of train examples seen between each epoch')
cmd:option('--validEpochSize', -1, 'number of valid examples used for early stopping and cross-validation') 

cmd:option('--featFile', '', 'file of feature vectors')
cmd:option('--targFile', '', 'file of target vector')
dname,fname = sys.fpath()
cmd:option('-save', fname:gsub('.lua',''), 'subdirectory to save/log experiments in')

cmd:text()
opt = cmd:parse(arg or {})
opt.hiddenSize = dp.returnString(opt.hiddenSize)
if not opt.silent then
   table.print(opt)
end

-- type:
if opt.cuda == true then
   print(sys.COLORS.red ..  '==> switching to CUDA')
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
   print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
end

------------------------------------------------------------
-- Data
------------------------------------------------------------

nClass = 11 -- UCF11 has 11 categories

ds = {}
-- TODO: ds.size should correspondes to the number of samples(frames) 
ds.size = 1100
ds.FeatureDims = 1024 -- initial the dimension of feature vector
classes = {'1','2','3','4','5','6','7','8','9','10','11'}

F = torch.load('feat_label_UCF11.t7')
ds.input = F.featMats
ds.target = F.labels

if (false) then
-- input dimension = ds.size x ds.FeatureDims x opt.rho = 1000 x 4096
   if not (opt.featFile == '') then
      -- read feature file from command line
      print(' - - Reading external feature file . . .')
      file = torch.DiskFile(opt.featFile, 'r')
      ds.input = file:readObject()
   else
      -- generate random feature file
      print(c.red .. ' - - No --featFile specified. Generating random feature matrix . . . ' .. c.white)
      ds.input = torch.randn(ds.size, ds.FeatureDims, opt.rho)
   end

   -- target dimension = ds.size x 1 = 1000 x 1
   if not (opt.targFile == '') then
      -- read feature file from command line
      print(' - - Reading external target file . . .')
      file = torch.DiskFile(opt.targFile, 'r')
      ds.target = file:readObject()
   else
      print(c.red .. ' - - No --targFile specified. Generating random target vector . . . ' .. c.white)
      ds.target = torch.DoubleTensor(ds.size):random(nClass)
   end
end

------------------------------------------------------------
-- n-fold cross-validation
------------------------------------------------------------

nFolds = 5
-- shuffle the dataset
shuffle = torch.randperm(ds.size)
Index = torch.ceil(ds.size/nFolds)
-- extract test data
TestIndices = shuffle:sub(1,Index)
Test_ind = torch.LongTensor():resize(TestIndices:size()):copy(TestIndices)
TestData = ds.input:index(1,Test_ind)
TestTarget = ds.target:index(1,Test_ind)
-- extract train data
TrainIndices = shuffle:sub(Index+1,ds.size)
Train_ind = torch.LongTensor():resize(TrainIndices:size()):copy(TrainIndices)
TrainData = ds.input:index(1,Train_ind)
TrainTarget = ds.target:index(1,Train_ind)


------------------------------------------------------------
-- Model 
------------------------------------------------------------

-- Video Classification model
vc_rnn = nn.Sequential()

local inputSize = opt.hiddenSize[1]
for i,hiddenSize in ipairs(opt.hiddenSize) do 

   if i~= 1 and (not opt.lstm) and (not opt.gru) then
      vc_rnn:add(nn.Sequencer(nn.Linear(inputSize, hiddenSize)))
   end
   
   -- recurrent layer
   local rnn
   if opt.gru then
      -- Gated Recurrent Units
      rnn = nn.Sequencer(nn.GRU(inputSize, hiddenSize))
   elseif opt.lstm then
      -- Long Short Term Memory
      rnn = nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize))
   else
      -- simple recurrent neural network
      rnn = nn.Recurrent(
         hiddenSize, -- first step will use nn.Add
         nn.Identity(), -- for efficiency (see above input layer) 
         nn.Linear(hiddenSize, hiddenSize), -- feedback layer (recurrence)
         nn.Sigmoid(), -- transfer function 
         --99999 -- maximum number of time-steps per sequence
         opt.rho
      )
      if opt.zeroFirst then
         -- this is equivalent to forwarding a zero vector through the feedback layer
         rnn.startModule:share(rnn.feedbackModule, 'bias')
      end
      rnn = nn.Sequencer(rnn)
   end
   
   vc_rnn:add(rnn)

   if opt.dropout then -- dropout it applied between recurrent layers
      vc_rnn:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
   end
   
   inputSize = hiddenSize
end

------------------------------------------------------------
-- if only using simple Sequencer
------------------------------------------------------------
-- rnn = nn.Recurrent(
-- opt.hiddenSize[1], -- size of output
-- nn.Linear(ds.FeatureDims, opt.hiddenSize[1]), -- input layer
-- nn.Linear(opt.hiddenSize[1], opt.hiddenSize[1]), -- recurrent layer
-- nn.Sigmoid(), -- transfer function
-- opt.rho
-- )

-- vc_rnn = nn.Sequential()
-- -- vc_rnn:insert(nn.SplitTable(3,1), 1) -- tensor to table of tensors, which can't not be used in 'nn.Sequencer'
-- -- vc_rnn:add(rnn)
-- :add(nn.FastLSTM(ds.FeatureDims, opt.hiddenSize[1]))
-- :add(nn.FastLSTM(opt.hiddenSize[1], opt.hiddenSize[2]))
-- :add(nn.Linear(opt.hiddenSize[2], nClass))
-- :add(nn.LogSoftMax())

-- vc_rnn = nn.Sequencer(vc_rnn)




-- input layer 
-- vc_rnn:insert(nn.SplitTable(1,2), 1) -- tensor to table of tensors
 vc_rnn:insert(nn.SplitTable(3,1), 1) -- tensor to table of tensors

if opt.dropout then
   vc_rnn:insert(nn.Dropout(opt.dropoutProb), 1)
end

-- output layer
vc_rnn:add(nn.SelectTable(-1)) -- this selects the last time-step of the rnn output sequence
vc_rnn:add(nn.Linear(inputSize, nClass))
vc_rnn:add(nn.LogSoftMax())

-- vc_rnn:add(nn.Sequencer(nn.SelectTable(-1))) -- this selects the last time-step of the rnn output sequence
-- vc_rnn:add(nn.Sequencer(nn.Linear(inputSize, nClass)))
-- vc_rnn:add(nn.Sequencer(nn.LogSoftMax()))

if opt.uniform > 0 then
   for k,param in ipairs(vc_rnn:parameters()) do
      param:uniform(-opt.uniform, opt.uniform)
   end
end

-- will recurse a single continuous sequence
vc_rnn:remember((opt.lstm or opt.gru) and 'both' or 'eval')
print(vc_rnn)

-- build criterion
criterion = nn.ClassNLLCriterion()


if opt.cuda == true then
   vc_rnn:cuda()
   criterion:cuda()
end


------------------------------------------------------------
-- Initialization before training
------------------------------------------------------------
-- Initialize the input and target tensors
local inputs = torch.Tensor(opt.batchSize, TrainData:size(2), TrainData:size(3))
local targets = torch.Tensor(opt.batchSize)

if opt.cuda == true then
   inputs = inputs:cuda()
   targets = targets:cuda()
end


local indices = torch.LongTensor(opt.batchSize)
-- indices:resize(opt.batchSize) -- indices to be used later, so it is resized to batchsize


-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes)

-- Log results to files
local trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))


-- Pass learning rate from command line
opt.learningRate = opt.startLearningRate

local optimState = {
   learningRate = opt.learningRate,
   momentum = opt.momentum,
   weightDecay = opt.weightDecay,
   learningRateDecay = opt.learningRateDecay
}


-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
local w,dE_dw = vc_rnn:getParameters()

------------------------------------------------------------
-- Train
------------------------------------------------------------
for iteration = 1, opt.maxEpoch do

   local time = sys.clock()

   -- shuffle at each epoch
   local shuffle = torch.randperm(TrainData:size(1))

   -- do one epoch
   print(sys.COLORS.green .. '==> doing epoch on training data:') 
   print("==> online epoch # " .. iteration .. ' [batchSize = ' .. opt.batchSize .. ']')

   for t = 1,TrainData:size(1),opt.batchSize do


      if opt.progress == true then
         -- disp progress
         xlua.progress(t, TrainData:size(1))
      end
      collectgarbage()

      -- batch fits?
      if (t + opt.batchSize - 1) > TrainData:size(1) then
         break
      end

      -- create mini batch
      local idx = 1
      for i = t,t+opt.batchSize-1 do
         inputs[idx] = TrainData[shuffle[i]]
         targets[idx] = TrainTarget[shuffle[i]]
         idx = idx + 1
      end

      --------------------------------------------------------
      -- My defined training and update process
      --------------------------------------------------------
      -- [[forward sequence through vc_rnn
      -- vc_rnn:zeroGradParameters() 

      -- local outputs = vc_rnn:forward(inputs)
      -- local err = criterion:forward(outputs, targets)
      
      -- print(string.format("Iteration %d ; NLL err = %f ", iteration, err))


      -- -- backward sequence through vc_rnn (i.e. backprop through time)
      -- local gradOutputs = criterion:backward(outputs, targets)
      -- local gradInputs = vc_rnn:backward(inputs, gradOutputs)


      -- -- update confusion
      -- for i = 1,opt.batchSize do
      --    confusion:add(outputs[i],targets[i])
      -- end


      -- -- update parameters
      -- vc_rnn:updateParameters(opt.learningRate)

      -- -- learning rate decay
      -- opt.learningRate = opt.learningRate + (opt.minLR - opt.startLearningRate)/opt.saturateEpoch
      -- opt.learningRate = math.max(opt.minLR, opt.learningRate)
      -- if not opt.silent then
      --    print("learning rate = ", opt.learningRate)
      -- end

      --------------------------------------------------------
      -- Using optim package for training
      --------------------------------------------------------
      local eval_E = function(w)
         -- reset gradients
         dE_dw:zero()

         -- evaluate function for complete mini batch
         local outputs = vc_rnn:forward(inputs)
         local E = criterion:forward(outputs,targets)

         -- estimate df/dW
         local dE_dy = criterion:backward(outputs,targets)   
         vc_rnn:backward(inputs,dE_dy)

         -- update confusion
         for i = 1,opt.batchSize do
            confusion:add(outputs[i],targets[i])
         end

         -- return f and df/dX
         return E,dE_dw
      end

      -- optimize on current mini-batch
      optim.sgd(eval_E, w, optimState)
   end

   -- time taken
   time = sys.clock() - time
   time = time / TrainData:size(1)
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end
   -- next epoch
   confusion:zero()

end


--[[
local answer
repeat
   io.write("continue with this operation (y/n)? ")
   io.flush()
   answer=io.read()
until answer=="y" or answer=="n"
--]]
