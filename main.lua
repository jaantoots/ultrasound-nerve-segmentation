require 'torch'
require 'cutorch'
require 'nn'
-- require 'cunn'
require 'cudnn'

-- Load the datasets
-- TODO: Return random permutations
trainset = torch.load('train.t7'):cuda()
testset = torch.load('test.t7'):cuda()
