----------------------------------------------------------------------
-- Load Data and separate training and testing samples
--
-- Hao Yan
----------------------------------------------------------------------


require 'torch'   -- torch

dataall = torch.load('feat_label_UCF11.t7')
dataall.labels:eq(0):nonzero()
dataall.labels[454] = 5
dataall.labels[925] = 10

ndata = (#dataall.labels)[1]
local labelsShuffle = torch.randperm(ndata)
local trsize = torch.round(0.8*ndata)
local tesize = ndata - trsize
-- create train set:
trainData = {
   data = torch.Tensor(trsize, 1024, 57),
   labels = torch.Tensor(trsize),
   size = function() return trsize end
}

testData = {
      data = torch.Tensor(tesize, 1024, 57),
      labels = torch.Tensor(tesize),
      size = function() return tesize end
   }

classes = {'basketball','biking','diving','golf_swing','horse_riding','soccer_juggling','swing','tennis_swing','trampoline_jumping','volleyball_spiking','walking'}

for i = 1,trsize do
    trainData.data[i] = dataall.featMats[labelsShuffle[i]]:clone()
    trainData.labels[i] = dataall.labels[labelsShuffle[i]]
end


for i=trsize+1,tesize+trsize do
   testData.data[i-trsize] = dataall.featMats[labelsShuffle[i]]:clone()
   testData.labels[i-trsize] = dataall.labels[labelsShuffle[i]]
end

print(trainData)

dataall = nil
collectgarbage()


return {
   trainData = trainData,
   testData = testData,
   classes = classes
}

