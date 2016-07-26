require "torch"
require "cutorch"
require "nn"
local cudnn = require "cudnn"
local optim = require "optim"

-- Dataset handling methods
-- TODO: Return random permutations
local data = require "data"
-- TODO: Calculate weights
local weights = {}

-- Set metatable to return [i]th sample and func for size

-- Normalize data

-- Network and loss function
local net = require "model"
local criterion = cudnn.SpatialCrossEntropyCriterion(weights)
criterion = criterion:cuda()

-- Train the network
net:training()
local maxIterations = 1000
local batchSize = 8
local config = {
  learningRate = 1e-1,
  alpha = 0.99,
  epsilon = 1e-6
}
local params, gradParams = net:getParameters()
for i = 1, maxIterations do
  -- Get the minibatch
  local batch = data.batch(batchSize)
  local batchInputs = batch.inputs:cuda()
  local batchLabels = batch.labels:cuda()
  -- TODO: Check weights initialization

  local function feval (_)
    -- For optim, outputs f(X): loss and df/dx: gradients
    gradParams:zero()
    local outputs = net:forward(batchInputs)
    local loss = criterion:forward(outputs, batchLabels)
    local gradLoss = criterion:backward(outputs, batchLabels)
    net:backward(batchInputs, gradLoss)
    print(i, loss)
    return loss, gradParams
  end
  optim.rmsprop(feval, params, config)
end

-- Validate

-- Test
