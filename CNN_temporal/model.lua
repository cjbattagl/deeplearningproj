----------------------------------------------------------------------
-- Create CNN and loss to optimize.
--
-- Hao Yan
----------------------------------------------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers
--require 'Dropout' -- Hinton dropout technique
require 'sys'
----------------------------------------------------------------------
print '==> processing options'
----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> define parameters')

-- 11 classes problem
local noutputs = 11

-- input dimensions: 
local nframe = 48
local nfeature = 1024

-- hidden units, filter sizes (for ConvNet only):
local nstates = {16,32,1000}
local filtsize = {3, 5}
local poolsize = {4,2}

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
   CNN:add(nn.SpatialConvolution(1,nstates[1], 1,filtsize[1],1,1,0,(filtsize[1]-1)/2)) -- 48*1024
   CNN:add(nn.ReLU())
   CNN:add(nn.SpatialMaxPooling(1,poolsize[1],1,poolsize[1])) -- 12*1024

   -- conv -> ReLU -> Pooling   
   CNN:add(nn.SpatialConvolution(nstates[1], nstates[2],  1,filtsize[2],1,1,0,(filtsize[2]-1)/2,1)) -- 12*1024
   CNN:add(nn.ReLU()) 
   CNN:add(nn.SpatialMaxPooling(1,poolsize[2],1,poolsize[2]))  -- 6*1024
   CNN:add(nn.Dropout(opt.dropout))

   -- Linear 
   CNN:add(nn.Reshape(nstates[2]*6*1024))
   CNN:add(nn.Linear(nstates[2]*6*1024,nstates[3]))
   CNN:add(nn.ReLU())
   CNN:add(nn.Linear(nstates[3],noutputs))
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

