require 'torch'
require 'cutorch'
require 'nn'
-- require 'cunn'
require 'cudnn'

-- Load the datasets
-- TODO: Return random permutations
local trainset = torch.load('train.t7')
local testset = torch.load('test.t7')

-- Set metatable to return [i]th sample and func for size

-- Normalize data

-- Define model
--[[
   Use VGG configuration D or E: conv3 layers with ReLU and maxpooling
   No FC layers, image output
   Softmax loss
   Use dropout and batch normalization
--]]

-- Define loss function

-- Train the network

-- Validate

-- Test
