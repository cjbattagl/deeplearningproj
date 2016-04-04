-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Video Classification

-- Load Data and separate training and testing samples

-- TODO:
-- 1. subtract by mean (?)
-- 2. cross-validation 

-- modified by Min-Hung Chen
-- contact: cmhungsteve@gatech.edu
-- Last updated: 04/03/2016


require 'torch'   -- torch

----------------------------------------------
-- 				         Data paths		            --
----------------------------------------------
dirDatabase = '../Dataset/'

----------------------------------------------
-- 			      User-defined parameters		    --
----------------------------------------------
ratioTrain = 0.8

----------------------------------------------
-- 					       Load Data		    	      --
----------------------------------------------
-- Load all the feature matrices & labels
--dataall = torch.load(dirDatabase..'feat_label_UCF11.t7')
dataall = torch.load(dirDatabase..'feat_label_UCF101.t7')
-- dataall.labels:eq(0):nonzero()
-- dataall.labels[454] = 5
-- dataall.labels[925] = 10

-- information for the data
ndata = (#dataall.labels)[1]
dimFeat = dataall.featMats:size(2)
numFrame = dataall.featMats:size(3)

----------------------------------------------
--   Separate the training & testing sets   --
----------------------------------------------

local labelsShuffle = torch.randperm(ndata)
local trsize = torch.round(ratioTrain*ndata)
local tesize = ndata - trsize

-- create the train set:
trainData = {
   data = torch.Tensor(trsize, dimFeat, numFrame),
   labels = torch.Tensor(trsize),
   size = function() return trsize end
}

-- create testing set:
testData = {
      data = torch.Tensor(tesize, dimFeat, numFrame),
      labels = torch.Tensor(tesize),
      size = function() return tesize end
   }

-- classes in UCF-11
-- classes = {'basketball','biking','diving','golf_swing','horse_riding','soccer_juggling',
-- 			'swing','tennis_swing','trampoline_jumping','volleyball_spiking','walking'}

-- classes in UCF-101
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

