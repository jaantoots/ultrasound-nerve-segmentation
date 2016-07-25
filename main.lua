local torch = require "torch"
require "cutorch"
local nn = require "nn"
local cudnn = require "cudnn"

-- Dataset handling methods
-- TODO: Return random permutations
local data = require "data"

-- Set metatable to return [i]th sample and func for size

-- Normalize data

-- Network and loss function
local net = require "model"
local criterion = cudnn.SpatialCrossEntropyCriterion()
criterion = criterion:cuda()

-- Train the network

-- Validate

-- Test
