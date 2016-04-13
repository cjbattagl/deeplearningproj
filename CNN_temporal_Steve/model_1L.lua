-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Video Classification

-- Create CNN and loss to optimize.
-- 1 conv layer

-- TODO:
-- 1. change nstate
-- 2. change convsize, convstep, poolsize, poolstep

-- modified by Min-Hung Chen
-- contact: cmhungsteve@gatech.edu
-- Last updated: 04/13/2016


require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers
--require 'Dropout' -- Hinton dropout technique
require 'sys'


if opt.type == 'cuda' then
   nn.SpatialConvolutionMM = nn.SpatialConvolution
end

----------------------------------------------------------------------
print '==> processing options'
----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> define parameters')

-- 11 classes problem
local noutputs = 101

-- input dimensions: 
local nframe = 48
local nfeature = 1024

-- hidden units, filter sizes (for ConvNet only):
local nstates = {25,250} 		     -- exp. 1
local convsize = {11} 			     -- exp. 1
local convstep = {1}
local convpad  = {(convsize[1]-1)/2}
local poolsize = {2} 			-- exp. 1
local poolstep = {2} 			-- exp. 1

----------------------------------------------------------------------
local classifier = nn.Sequential()
local model = nn.Sequential()
local model_name ="model.net"

if opt.model == 'CNN' then
   print(sys.COLORS.red ..  '==> construct CNN')
   ---- TODO --------
   --Create a CNN network as mentioned in the write-up
   --followed by a 2 layer fully connected layers
   --Use ReLU as yoru activations
   model_name = 'CNN.net'

   local CNN = nn.Sequential()
   
   -- stage 1: conv -> ReLU -> Pooling
   CNN:add(nn.SpatialConvolutionMM(1,nstates[1],1,convsize[1],1,convstep[1],0,convpad[1])) -- 48*1024
   local sizeConv1_w = nfeature
   local sizeConv1_h = (nframe-convsize[1]+2*convpad[1])/convstep[1]+1

   CNN:add(nn.ReLU())
   CNN:add(nn.SpatialMaxPooling(1,poolsize[1],1,poolstep[1])) -- 12*1024
   local sizePool1_w = sizeConv1_w
   local sizePool1_h = (sizeConv1_h-poolsize[1])/poolstep[1]+1

   CNN:add(nn.Dropout(opt.dropout)) -- dropout

   -- stage 3: linear -> ReLU -> linear
   local ninputFC = sizePool1_w*sizePool1_h*nstates[1] -- exp. 1

   CNN:add(nn.Reshape(ninputFC))
   CNN:add(nn.Linear(ninputFC,nstates[2]))
   CNN:add(nn.ReLU())

   --CNN:add(nn.Dropout(opt.dropout)) -- dropout

   CNN:add(nn.Linear(nstates[2],noutputs))

   -- stage 4 : log probabilities
   CNN:add(nn.LogSoftMax())

   model:add(CNN)

end
   
-- Loss: NLL
loss = nn.ClassNLLCriterion()

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> here is the network:')
print(model)

if opt.type == 'cuda' then
   model:cuda()
   loss:cuda()
end

-- return package:
return {
   model = model,
   loss = loss,
   nfeature = 1024,
   nframe = 48,
   model_name = model_name,
}
