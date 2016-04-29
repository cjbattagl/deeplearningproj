require 'torch'
require 'nn'
dirDatabase = '.'

model_name = 'mlp.net' 


testData = torch.load('data_UCF101_test_1-2.t7')
tesize = (#testData.labels)[1]
dimFeat = testData.featMats:size(2)
nframe = testData.featMats:size(3)
testData.data = testData.featMats
testData.featMats = nil
testData.size =  function() return tesize end
collectgarbage()
local predarr = torch.Tensor(tesize,101)

model = torch.load(model_name)

torch.setdefaulttensortype('torch.FloatTensor')

for t = 1,testData:size() do
    for i =1,nframe do
        input = testData.data[{t,{},i}]
        binput = torch.FloatTensor(1,2048)
        binput:copy(input)
	predarr[{t,{}}] = model:forward(inputnew)
    end
end

sorted,indices=torch.sort(predarr,2,true)
predlabels=indices[{{},1}]

classes = {
"BoxingSpeedBag", "Surfing", "FloorGymnastics", "IceDancing", "Lunges", "Swing", "SkyDiving", "MilitaryParade", "PlayingPiano", "Punch",
"HulaHoop", "VolleyballSpiking", "Skijet", "JavelinThrow", "LongJump", "Mixing", "Shotput", "BandMarching", "Kayaking", "StillRings",
"PushUps", "Archery", "FieldHockeyPenalty", "BoxingPunchingBag", "PlayingCello", "FrontCrawl", "Billiards", "Rowing", "ApplyLipstick", "TrampolineJumping",
"CuttingInKitchen", "BodyWeightSquats", "JugglingBalls", "Nunchucks", "JumpRope", "PlayingViolin", "PlayingGuitar", "YoYo", "SumoWrestling", "SoccerJuggling",
"CliffDiving", "CricketBowling", "PlayingDhol", "HorseRiding", "BabyCrawling", "PlayingSitar", "TaiChi", "BenchPress", "PommelHorse", "BrushingTeeth",
"Hammering", "PlayingTabla", "HandstandWalking", "Typing", "CleanAndJerk", "TennisSwing", "CricketShot", "BlowDryHair", "HeadMassage", "BalanceBeam",
"TableTennisShot", "MoppingFloor", "Drumming", "PlayingFlute", "FrisbeeCatch", "ApplyEyeMakeup", "SkateBoarding", "BaseballPitch", "SoccerPenalty", "ThrowDiscus",
"RopeClimbing", "HorseRace", "HighJump", "PullUps", "Diving", "BreastStroke", "ParallelBars", "WalkingWithDog", "PizzaTossing", "BlowingCandles",
"GolfSwing", "PoleVault", "UnevenBars", "HandstandPushups", "JumpingJack", "WallPushups", "WritingOnBoard", "Skiing", "Bowling", "BasketballDunk",
"SalsaSpin", "ShavingBeard", "Basketball", "Knitting", "RockClimbingIndoor", "Haircut", "Biking", "Fencing", "Rafting", "PlayingDaf",
"HammerThrow"
}
predlabeltxt = {}
for i = 1,3754 do
    predlabeltxt[i] = classes[predlabels[{i,1}]]
end
torch.save('labels.txt',predlabeltxt,'ascii')
