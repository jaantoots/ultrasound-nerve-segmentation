local torch = require "torch"
require "cutorch"
require "nn"
require "cunn"
local cudnn = require "cudnn"
local paths = require "paths"
local gm = require "graphicsmagick"
local data = require "data"
local helpers = require "helpers"

-- Enable these for final training and evaluation
cudnn.benchmark = true
cudnn.fastest = true

-- Parse arguments & load configuration
local parser = helpers.testParser()
local args = parser:parse()
local opts = helpers.opts(args)

-- Option for running full images through the trained network
if args.no_resize then
  opts.height = nil
  opts.width = nil
  opts.batchSize = opts.altBatchSize
end

-- Load network from file
local net = torch.load(args.model)
local modelName = string.match(args.model, '(.*)%.t7$')
-- Evaluate the network
-- net:evaluate() -- For some reason, the model does not work in evaluate mode
local function predict (dataset, suffix)
  -- Output predictions for a dataset
  local fileName = modelName .. '-' .. suffix .. '.csv'
  if paths.filep(fileName) then
    return false
  end
  local file = io.open(fileName, 'w')

  for i = 1, math.ceil(dataset.size/opts.batchSize) do
    -- Get the minibatch without random permutation
    local batch, names = dataset:batch(opts.batchSize, true, true)
    local batchInputs = batch.inputs:cuda()
    -- Forward pass
    local outputs = net:forward(batchInputs)
    local _, predictions = outputs:max(outputs:dim() - 2)
    predictions = predictions:squeeze():int() - 1

    -- Output
    for j, name in pairs(names) do
      if (i - 1)*opts.batchSize + j > dataset.size then break end
      local pred = predictions[j]
      local predStretch = torch.Tensor(1, pred:size(1), pred:size(2))
      predStretch[1] = pred
      -- Resize image to original size
      if not args.no_resize then
        local img = gm.Image(predStretch, 'I', 'DHW'):
          size(dataset.owidth, dataset.oheight)
        pred = img:toTensor('double', 'I', 'HW'):round():int()
      end
      -- Different name style for file output for test data
      if string.match(name, '%d+_%d+') then
        local subject, image = string.match(name, '(%d+)_(%d+)')
        name = subject .. ',' .. image
      end
      -- Output run length encoded data
      file:write(name, ',', helpers.encode(pred), '\n')
    end
  end
  file:close()
  return true
end

-- Initialize and normalize data, and run test
local trainData = data.new(opts.trainDir, opts.height, opts.width)
trainData:normalize(opts.mean, opts.std)
print("==> Predict training data")
predict(trainData, 'train_masks')

local testData = data.new(opts.testDir, opts.height, opts.width)
testData:normalize(opts.mean, opts.std)
print("==> Predict test data")
predict(testData, 'test_masks')
