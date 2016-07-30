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
local parser = helpers.validateParser()
local args = parser:parse()
local opts = helpers.opts(args)

-- Initialize and normalize training and validation data
local trainData = data.new(opts.train,
  opts.height, opts.width, opts.validationSubjects)
trainData:normalize(opts.mean, opts.std)
local validateData = data.new(opts.validate,
  opts.height, opts.width, opts.validationSubjects, true)
validateData:normalize(opts.mean, opts.std)

-- Load network from file
local net = torch.load(args.model)
local modelName = string.match(args.model, '(.*)%.t7$')

-- Prepare output
paths.mkdir(opts.output)
local trainLogger = optim.Logger(
  opts.output .. '/' .. modelName .. '-train.txt')
trainLogger:setNames{'Name', 'Score'}
local validateLogger = optim.Logger(
  opts.output .. '/' .. modelName .. '-validate.txt')
trainLogger:setNames{'Name', 'Score'}

-- Evaluate the network
net:evaluate()
local function evaluate (dataset, logger)
  local scores = {}
  for i = 1, math.ceil(dataset.size/opts.batchSize) do
    -- Get the minibatch without random permutation
    local batch, names = dataset:batch(opts.batchSize, true)
    local batchInputs = batch.inputs:cuda()
    local batchLabels = batch.labels:cuda()
    -- Forward pass and score
    local outputs = net:forward(batchInputs)
    local diceValue = helpers.dice(outputs, batchLabels)
    -- Output
    for j, name in pairs(names) do
      if (i - 1)*opts.batchSize + j > dataset.size then break end
      logger:add{name, diceValue[j]}
      scores[#scores + 1] = diceValue[j]
    end
  end
  return torch.Tensor(scores):mean()
end
print("==> Start validation")
local trainScore = evaluate(trainData, trainLogger)
print("Training data score:", trainScore)
local validateScore = evaluate(validateData, validateLogger)
print("Validation data score:", validateScore)
json.save(opts.output .. '/' .. modelName .. '.json',
  {trainScore, validateScore})
