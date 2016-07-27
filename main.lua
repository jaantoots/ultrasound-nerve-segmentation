local torch = require "torch"
require "cutorch"
require "nn"
require "cunn"
local cudnn = require "cudnn"
local optim = require "optim"
local paths = require "paths"
local json = require "json"
local helpers = require "helpers"

-- Enable these for final training
-- cudnn.benchmark = true
-- cudnn.fastest = true

-- Parse arguments & load configuration
local parser = helpers.parser()
local args = parser:parse()
local opts = helpers.opts(args)

-- Dataset handling
local data = require "data"
data.init(opts.dataDir, opts.height, opts.width)
opts.weights = opts.weights or data.weights()
opts.mean, opts.std = data.normalize(opts.mean, opts.std)

-- Network and loss function
local net
local criterion = cudnn.SpatialCrossEntropyCriterion(torch.Tensor(opts.weights))
criterion = criterion:cuda()
-- Load network from file if provided
local startIteration = 0
if args.model then
  net = torch.load(args.model)
  startIteration = string.match(args.model, '_(%d+)%.bin$') or startIteration
else
  net = require "model"
end

-- Prepare output
opts.maxIterations = (startIteration + args.iter) or opts.maxIterations or
  (startIteration + 10000)
paths.mkdir(opts.outDir)
json.save(opts.outDir .. '/conf.json', opts)
local logger = optim.Logger(opts.outDir .. '/accuracy.log')
logger:setNames{'Iteration', 'Loss', 'Score'}

-- Train the network
net:training()
local params, gradParams = net:getParameters() -- optim requires 1D tensors
local lossWindow = torch.Tensor(10):zero()
print("Check parameters:", params:mean(), params:std())
print("==> Start training: " .. params:nElement() .. " parameters")
for i = (startIteration + 1), opts.maxIterations do
  -- Get the minibatch
  local batch = data.batch(opts.batchSize)
  local batchInputs = batch.inputs:cuda()
  local batchLabels = batch.labels:cuda()
  local diceValue

  local function feval (_)
    -- For optim, outputs f(X): loss and df/dx: gradients
    gradParams:zero()
    -- Forward pass
    local outputs = net:forward(batchInputs)
    local loss = criterion:forward(outputs, batchLabels)
    -- Backpropagation
    local gradLoss = criterion:backward(outputs, batchLabels)
    net:backward(batchInputs, gradLoss)
    -- Statistics
    diceValue = helpers.dice(outputs, batchLabels)
    print(i, loss, diceValue:mean())
    return loss, gradParams
  end
  local _, fs = optim.rmsprop(feval, params, opts.config)

  -- Log loss
  lossWindow[math.fmod(i, 10) + 1] = fs[1]
  if i >= 10 then
    logger:add{i, lossWindow:mean(), diceValue:mean()}
  end
  -- Save model
  if math.fmod(i, 1000) == 0 then
    net:clearState()
    torch.save(opts.outDir .. '/model_' .. i .. '.bin', net)
  end
end

-- TODO: Validate after maxIterations
