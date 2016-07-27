local torch = require "torch"
require "cutorch"
require "nn"
require "cunn"
local cudnn = require "cudnn"
local optim = require "optim"
local paths = require "paths"
local json = require "json"
local argparse = require "argparse"
local helpers = require "helpers"

-- Enable these for final training
-- cudnn.benchmark = true
-- cudnn.fastest = true

-- Load configuration
local opts
if paths.filep('conf.json') then
  opts = json.load('conf.json')
else
  opts = {}
end
opts.dataDir = opts.dataDir or 'train'
opts.outDir = opts.outDir or 'out/2016-07-27-test'
opts.height = opts.height or 200
opts.width = opts.width or 280

-- Dataset handling
local data = require "data"
data.init(opts.dataDir, opts.height, opts.width)
opts.weights = opts.weights or data.weights()
opts.mean, opts.std = data.normalize(opts.mean, opts.std)

-- Network and loss function
local net = require "model"
local criterion = cudnn.SpatialCrossEntropyCriterion(torch.Tensor(opts.weights))
criterion = criterion:cuda()

-- Train the network
net:training()
-- TODO: Create argparser and config file
opts.maxIterations = opts.maxIterations or 1000
opts.batchSize = opts.batchSize or 8
opts.config = opts.config or {
  learningRate = 1e-1,
  alpha = 0.99,
  epsilon = 1e-6
}

local params, gradParams = net:getParameters() -- optim requires 1D tensors
print("Check parameters:", params:mean(), params:std())
print("==> Start training: " .. params:nElement() .. " parameters")

paths.mkdir(opts.outDir)
json.save(opts.outDir .. '/conf.json', opts)
local logger = optim.Logger(opts.outDir .. '/accuracy.log')
logger:setNames{'Iteration', 'Loss', 'Score'}
-- TODO: Add accuracy function
local lossWindow = torch.Tensor(10):zero()

for i = 1, opts.maxIterations do
  -- Get the minibatch
  local batch = data.batch(opts.batchSize)
  local batchInputs = batch.inputs:cuda()
  local batchLabels = batch.labels:cuda()
  -- TODO: Check weights initialization
  local diceValue

  local function feval (_)
    -- For optim, outputs f(X): loss and df/dx: gradients
    gradParams:zero()
    local outputs = net:forward(batchInputs)
    local loss = criterion:forward(outputs, batchLabels)
    local gradLoss = criterion:backward(outputs, batchLabels)
    net:backward(batchInputs, gradLoss)
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
