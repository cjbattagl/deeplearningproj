-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Video Classification

-- Load Data and separate training and testing samples

-- TODO:
-- 1. subtract by mean (?)

-- modified by Hao Yan
-- contact: yanhao@gatech.edu
-- Last updated: 04/04/2016


require 'torch'   -- torch

----------------------------------------------
-- 				         Data paths		            --
----------------------------------------------
dirDatabase = './'

----------------------------------------------
-- 			      User-defined parameters		    --
----------------------------------------------
ratioTrain = 0.8

----------------------------------------------
-- 					       Load Data		    	      --
----------------------------------------------
-- Load all the feature matrices & labels
--dataall = torch.load(dirDatabase..'feat_label_UCF11.t7')
trainData = torch.load(dirDatabase..'data_UCF101_train_1-2.t7')
testData = torch.load(dirDatabase..'data_UCF101_test_1-2.t7')


-- information for the data
trsize = (#trainData.labels)[1]
tesize = (#testData.labels)[1]
dimFeat = trainData.featMats:size(2)
numFrame = trainData.featMats:size(3)
print(trsize,tesize,dimFeat,numFrame)
----------------------------------------------
--   Separate the training & testing sets   --
----------------------------------------------

--local labelsShuffle = torch.randperm(ndata)

-- create the train set:
trainData.data = trainData.featMats
trainData.featMats = nil

collectgarbage()
trainData.size =  function() return trsize end
testData.data = testData.featMats
testData.featMats = nil
testData.size =  function() return tesize end
collectgarbage()

print('finishing train test')

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

return {
   trainData = trainData,
   testData = testData,
   classes = classes
}

