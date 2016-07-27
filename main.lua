local torch = require "torch"
require "cutorch"
require "nn"
require "cunn"
local cudnn = require "cudnn"
local optim = require "optim"
local paths = require "paths"
local json = require "json"

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
opts.dir = opts.dir or 'train'
opts.height = opts.height or 200
opts.width = opts.width or 280

-- Dataset handling methods
local data = require "data"
data.init(opts.dir, opts.height, opts.width)
opts.weights = opts.weights or data.weights()
opts.mean, opts.std = data.normalize(opts.mean, opts.std)

-- Network and loss function
local net = require "model"
local criterion = cudnn.SpatialCrossEntropyCriterion(opts.weights)
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

json.save('out/conf.json', opts)
local logger = optim.Logger('out/accuracy.log')
logger:setNames{'Iteration', 'Loss'}
-- TODO: Add accuracy function
local lossWindow = torch.Tensor(10):zero()

for i = 1, opts.maxIterations do
  -- Get the minibatch
  local batch = data.batch(opts.batchSize)
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
  local _, fs = optim.rmsprop(feval, params, opts.config)

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

-- TODO: Calculate accuracy score as Dice coefficient

-- TODO: Validate after maxIterations
