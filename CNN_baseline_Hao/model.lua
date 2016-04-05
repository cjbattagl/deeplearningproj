-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Video Classification

-- Create MLP and loss to optimize.

-- TODO:
-- 1. change nstate

-- modified by Hao Yan
-- contact: cmhungsteve@gatech.edu
-- Last updated: 04/03/2016


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

-- hidden units, filter sizes (for mlp only):
local nstates = 256

----------------------------------------------------------------------
local classifier = nn.Sequential()
local model = nn.Sequential()
local model_name ="model.net"

if opt.model == 'Linear' then
   print(sys.COLORS.red ..  '==> construct mlp')
   ---- TODO --------
   --Create a CNN network as mentioned in the write-up
   --followed by a 2 layer fully connected layers
   --Use ReLU as yoru activations
   model_name = 'mlp.net'

   local mlp = nn.Sequential()
   -- stage 1 : FC layer 
   mlp:add(nn.Linear(nfeature,nstates))
   mlp:add(nn.ReLU())
   mlp:add(nn.Linear(nstates,noutputs))

   -- stage 2 : log probabilities
   mlp:add(nn.LogSoftMax())
   model:add(mlp)

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

