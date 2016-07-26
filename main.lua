local torch = require "torch"
require "cutorch"
require "nn"
require "cunn"
local cudnn = require "cudnn"
local optim = require "optim"

-- Enable these for final training
-- cudnn.benchmark = true
-- cudnn.fastest = true

-- Dataset handling methods
local data = require "data"
data.init('train', 200, 280)
-- TODO: Downscaled images should fit to memory: faster not to read from disk
-- TODO: Calculate weights
local weights = torch.Tensor{0.5, 0.5}

-- TODO: Normalize data

-- Network and loss function
local net = require "model"
local criterion = cudnn.SpatialCrossEntropyCriterion(weights)
criterion = criterion:cuda()

-- Train the network
net:training()
-- TODO: Create argparser and config file
local maxIterations = 1000
local batchSize = 8
local config = {
  learningRate = 1e-1,
  alpha = 0.99,
  epsilon = 1e-6
}

local params, gradParams = net:getParameters() -- optim requires 1D tensors
print("Check parameters:", params:mean(), params:std())
print("==> Start training: " .. params:nElement() .. " parameters")

local logger = optim.Logger('out/accuracy.log')
logger:setNames{'Iteration', 'Loss'}
-- TODO: Add accuracy function
local lossWindow = torch.Tensor(10):zero()

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
  local _, fs = optim.rmsprop(feval, params, config)

  -- Log loss
  lossWindow[math.fmod(i, 10) + 1] = fs[1]
  if i >= 10 then
    logger:add{i, lossWindow:mean()}
  end

  -- Save model
  if math.fmod(i, 1000) == 0 then
    net:clearState()
    torch.save('out' .. '/model_' .. i .. '.bin', net)
  end
end

-- TODO: Validate after maxIterations
