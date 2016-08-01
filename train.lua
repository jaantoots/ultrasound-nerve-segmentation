local torch = require "torch"
require "cutorch"
require "nn"
require "cunn"
local cudnn = require "cudnn"
local optim = require "optim"
local paths = require "paths"
local json = require "json"
local data = require "data"
local helpers = require "helpers"

-- Enable these for final training
cudnn.benchmark = true
cudnn.fastest = true

-- Parse arguments & load configuration
local parser = helpers.trainParser()
local args = parser:parse()
local opts = helpers.opts(args)

-- Initialize and normalize training data
local trainData = data.new(opts.train,
  opts.height, opts.width, opts.validationSubjects)
opts.mean, opts.std = trainData:normalize(opts.mean, opts.std)

-- Network and loss function
local net = require "model"
local criterion = cudnn.SpatialCrossEntropyCriterion(torch.Tensor(opts.weights))
criterion = criterion:cuda()
-- Load network from file if provided
local startIteration = 0
if args.model then
  net = torch.load(args.model)
  startIteration = string.match(args.model, '_(%d+)%.t7$') or startIteration
end

-- Prepare output
opts.maxIterations = args.iter and (startIteration + args.iter) or
  opts.maxIterations or (startIteration + 10000)
paths.mkdir(opts.output)
json.save(opts.output .. '/conf.json', opts)
local logger = optim.Logger(opts.output .. '/accuracylog.txt')
logger:setNames{'Iteration', 'Loss', 'Score'}
local lossWindow = torch.Tensor(10):zero()
local diceWindow = torch.Tensor(10):zero()

-- Train the network
net:training()
local params, gradParams = net:getParameters() -- optim requires 1D tensors
print("==> Start training: " .. params:nElement() .. " parameters")
for i = (startIteration + 1), opts.maxIterations do
  -- Get the minibatch
  local batch = trainData:batch(opts.batchSize)
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
    print(i, loss, diceValue[1])
    return loss, gradParams
  end
  local _, fs = optim.adam(feval, params, opts.config)

  -- Log loss
  lossWindow[math.fmod(i, 10) + 1] = fs[1]
  diceWindow[math.fmod(i, 10) + 1] = diceValue:mean()
  if i >= 10 then
    logger:add{i, lossWindow:mean(), diceWindow:mean()}
  end
  -- Save model
  if math.fmod(i, 1000) == 0 then
    net:clearState()
    torch.save(opts.output .. '/model_' .. i .. '.t7', net)
  end
end
