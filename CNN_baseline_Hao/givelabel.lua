require 'torch'
require 'nn'
dirDatabase = './'

testData = torch.load(dirDatabase..'data_UCF101_test_1-2.t7')
tesize = (#testData.labels)[1]
dimFeat = testData.featMats:size(2)
nframe = testData.featMats:size(3)
   --require 'cunn'
   --cutorch.setDevice(1)

testData.data = testData.featMats
testData.featMats = nil
testData.size =  function() return tesize end
collectgarbage()
local predarr = torch.Tensor(tesize,101,nframe)

model = torch.load(dirDatabase..'mlp.net')

torch.setdefaulttensortype('torch.FloatTensor')

for t = 1,testData:size() do
    print(t)
    for i =1,nframe do
        input = testData.data[{t,{},i}]
        --input:reshape(1,dimFeat)
	binput = torch.FloatTensor(1,2048)
	binput:copy(input)
        print(binput:type())
	print(model:forward(binput):size())
	predarr[{t,{},i}] = model:forward(binput)
    end
end

torch.save('predarrmlp.t7',predarr)

model = torch.load(dirDatabase..'mlpnew.net')

for t = 1,testData:size() do
    for i =1,nframe do
        input = testData.data[{t,{},i}]
	binput = torch.FloatTensor(1,2048)
	binput:copy(input)
        predarr[{t,{},i}] = model:forward(binput)
    end
end

torch.save('predarrmlpnew.t7',predarr)

