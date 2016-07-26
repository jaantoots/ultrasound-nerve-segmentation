local torch = require "torch"
require "cutorch"
require "nn"
local cudnn = require "cudnn"
local optim = require "optim"
local itorch = require "itorch"

-- Dataset handling methods
local data = require "data"
-- TODO: Calculate weights
local weights = torch.Tensor{0.5, 0.5}

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
print("Start training: " .. params:nElement() .. " parameters")
for i = 1, maxIterations do
  -- Get the minibatch
  local batch = data.batch(batchSize)
  local batchInputs = batch.inputs:cuda()
  local batchLabels = batch.labels:cuda()
  print(batchInputs:size(), batchLabels:size())
  -- TODO: Check weights initialization

  local function feval (_)
    -- For optim, outputs f(X): loss and df/dx: gradients
    gradParams:zero()
    print("Forward pass")
    local outputs = net:forward(batchInputs)
    print("Loss")
    local loss = criterion:forward(outputs, batchLabels)
    print("Loss gradient")
    local gradLoss = criterion:backward(outputs, batchLabels)
    print("Backpropagation")
    net:backward(batchInputs, gradLoss)
    print(i, loss)
    if math.fmod(i, 10) then
      local _, predLabels = torch.max(outputs[1]:squeeze(), 1)
      itorch.image({batchInputs[1], batchLabels[1] - 1, predLabels - 1})
    end
    return loss, gradParams
  end
  optim.rmsprop(feval, params, config)
end

net:clearState()
torch.save('out' .. '/model_' .. maxIterations .. '.bin', net)

-- Validate

-- Test
