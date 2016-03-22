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

version = 1



--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Language Model on PennTreeBank dataset using RNN or LSTM or GRU')
cmd:text('Example:')
cmd:text("recurrent-language-model.lua --cuda --useDevice 2 --progress --zeroFirst --cutoffNorm 4 --opt.rho 10")
cmd:text('Options:')
cmd:option('--startLearningRate', 0.05, 'learning rate at t=0')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 400, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxOutNorm', -1, 'max l2-norm of each layer\'s output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('--batchSize', 8, 'number of examples per batch') -- how many examples per training 
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 1000, 'maximum number of epochs to run')
cmd:option('--maxTries', 50, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'don\'t print anything to stdout')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')

-- recurrent layer 
cmd:option('--lstm', true, 'use Long Short Term Memory (nn.LSTM instead of nn.Recurrent)')
cmd:option('--gru', false, 'use Gated Recurrent Units (nn.GRU instead of nn.Recurrent)')
cmd:option('--rho', 5, 'back-propagate through time (BPTT) for opt.rho time-steps')
cmd:option('--hiddenSize', '{4096, 800, 200}', 'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs/GRUs are stacked')
cmd:option('--zeroFirst', false, 'first step will forward zero through recurrence (i.e. add bias of recurrence). As opposed to learning bias specifically for first step.')
cmd:option('--dropout', false, 'apply dropout after each recurrent layer')
cmd:option('--dropoutProb', 0.5, 'probability of zeroing a neuron (dropout probability)')

-- data
cmd:option('--trainEpochSize', -1, 'number of train examples seen between each epoch')
cmd:option('--validEpochSize', -1, 'number of valid examples used for early stopping and cross-validation') 

cmd:text()
opt = cmd:parse(arg or {})
opt.hiddenSize = dp.returnString(opt.hiddenSize)
if not opt.silent then
   table.print(opt)
end


------------------------------------------------------------
-- Data
------------------------------------------------------------

nClass = 11 -- UCF11 has 11 categories

ds = {}
-- TODO: ds.size should correspondes to the number of samples(frames) 
ds.size = 1000
ds.FeatureDims = 4096 -- initial the dimension of feature vector


-- input dimension = ds.size x ds.FeatureDims x opt.rho = 1000 x 4096
ds.input = torch.randn(ds.size, ds.FeatureDims, opt.rho)

-- target dimension = ds.size x 1 = 1000 x 1
ds.target = torch.DoubleTensor(ds.size):random(nClass)


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




-- input layer (i.e. word embedding space)

-- vc_rnn:insert(nn.SplitTable(1,2), 1) -- tensor to table of tensors
 vc_rnn:insert(nn.SplitTable(3,1), 1) -- tensor to table of tensors

if opt.dropout then
   vc_rnn:insert(nn.Dropout(opt.dropoutProb), 1)
end

-- TODO: LookupTable can only take 1D or 2D input 
-- lookup = nn.LookupTable(nClass, opt.hiddenSize[1])
-- lookup.maxOutNorm = -1 -- disable maxParamNorm on the lookup table
-- vc_rnn:insert(lookup, 1)


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


------------------------------------------------------------
-- Train
------------------------------------------------------------
-- local inputs, targets = torch.LongTensor(), torch.LongTensor()
-- local inputs, targets = {}, {}
local inputs, targets = torch.Tensor(), torch.Tensor()

local indices = torch.LongTensor(opt.batchSize)
-- indices:resize(opt.batchSize) -- indices to be used later, so it is resized to batchsize

-- build criterion
criterion = nn.ClassNLLCriterion()

opt.learningRate = opt.startLearningRate

for iteration = 1, opt.maxEpoch do
   -- [[create a sequence of opt.rho time-steps

   indices:random(1,ds.size) -- choose some random samples for training
   inputs:index(ds.input, 1,indices)
   targets:index(ds.target, 1,indices)


   ------------------------------------------------------------
   -- If SplitTable in Sequencer doesn't work, process the input first
   ------------------------------------------------------------
   -- Convert tensor to table of tensors
   -- mlp = nn.SplitTable(3,1)
   -- inputs = mlp:forward(inputs)

   -- Naive way to convert tensor to table of tensors
   -- for step = 1, opt.rho do
   --    -- batch of inputs
   --    inputs[step] = inputs[step] or ds.input.new()
   --    -- inputs[step]:index(ds.input:select(3,step), 1, indices)
   --    inputs[step] = (ds.input:select(3,step))
   -- end

   -- [[forward sequence through vc_rnn
   
   vc_rnn:zeroGradParameters() 

   local outputs = vc_rnn:forward(inputs)
   local err = criterion:forward(outputs, targets)
   
   print(string.format("Iteration %d ; NLL err = %f ", iteration, err))


   -- backward sequence through vc_rnn (i.e. backprop through time)
   
   local gradOutputs = criterion:backward(outputs, targets)

   local gradInputs = vc_rnn:backward(inputs, gradOutputs)
   
   -- update parameters
   
   vc_rnn:updateParameters(opt.learningRate)

   -- learning rate decay
   opt.learningRate = opt.learningRate + (opt.minLR - opt.startLearningRate)/opt.saturateEpoch
   opt.learningRate = math.max(opt.minLR, opt.learningRate)
   if not opt.silent then
      print("learning rate = ", opt.learningRate)
   end

   -- empty 'inputs' tensor
   -- TODO: collectgarbage(?)
   inputs = torch.Tensor()

end


--[[
local answer
repeat
   io.write("continue with this operation (y/n)? ")
   io.flush()
   answer=io.read()
until answer=="y" or answer=="n"
--]]
