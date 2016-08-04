local torch = require "torch"
local paths = require "paths"
local json = require "json"
local argparse = require "argparse"

local helpers = {}

function helpers.trainParser ()
  -- Return argparse object
  local parser = argparse("train.lua",
    "Train a VGG net for ultrasound nerve segmentation.")
  parser:option("-c --conf", "Configuration file.", "conf.json")
  parser:option("-o --output", "Output directory.")
  parser:option("-b --batch", "Batch size.")
  parser:option("-i --iter", "Number of iterations to train.")
  parser:option("-m --model", "Saved model, if continuing training.")
  return parser
end

function helpers.validateParser ()
  -- Return argparse object
  local parser = argparse("validate.lua",
    "Validate a model for ultrasound nerve segmentation.")
  parser:argument("model", "Model file, or a directory with multiple models.")
  parser:option("-c --conf", "Configuration file.", "conf.json")
  parser:option("-b --batch", "Batch size.")
  return parser
end

function helpers.testParser ()
  -- Return argparse object
  local parser = argparse("test.lua",
    "Test a model for ultrasound nerve segmentation.")
  parser:argument("model", "Model file.")
  parser:flag("-r --no-resize", "Do not resize images for forward pass.")
  parser:option("-c --conf", "Configuration file.", "conf.json")
  parser:option("-b --batch", "Batch size.")
  return parser
end

function helpers.opts (args)
  -- Return opts for training
  local opts
  if paths.filep(args.conf) then
    opts = json.load(args.conf)
  else
    opts = {}
  end
  opts.train = opts.train or 'train'
  opts.validate = opts.validate or 'train'
  opts.trainDir = opts.trainDir or 'train'
  opts.testDir = opts.testDir or 'test'
  opts.output = args.output or opts.output or
    'out/' .. os.date('%Y-%m-%d-%H-%M-%S')
  opts.height = opts.height or 200
  opts.width = opts.width or 280
  opts.weights = opts.weights or {1, 1}
  opts.batchSize = args.batch or opts.batchSize or 8
  opts.altBatchSize = args.batch or opts.altBatchSize or 2
  opts.config = opts.config or {
    learningRate = 1e-1,
    alpha = 0.99,
    epsilon = 1e-6
  }
  return opts
end

function helpers.dice (outputs, targets)
  --[[ Calculate accuracy score as Dice coefficient

  Parameters
  ----------
  outputs: torch.Tensor of size [batchSize] x 2 x height x width, network output
      containing class scores for each pixel
  targets: torch.Tensor of size [batchSize] x height x width, ground truth
      labels containing class indices for each pixel

  Returns
  -------
  coeffs: torch.Tensor of size [batchSize] (or double) containing the Dice
      scores for each image in the batch
  --]]
  local _, predictions = outputs:max(outputs:dim() - 2)
  predictions = predictions:squeeze():double() - 1
  targets = targets:double() - 1
  -- Numerator
  local nums = torch.cmul(predictions, targets)
  nums = nums:sum(nums:dim()):squeeze()
  nums = nums:sum(nums:dim())
  -- Denominator
  local dens = predictions + targets
  dens = dens:sum(dens:dim()):squeeze()
  dens = dens:sum(dens:dim())
  -- Coefficient
  local coeffs = 2*torch.cdiv(nums, dens):squeeze()
  coeffs[coeffs:ne(coeffs)] = 1 -- by definition if both sets are zero
  return coeffs
end

function helpers.encode (mask)
  -- Run-length encoder for the mask
  local size = mask:size()
  local output = ""
  local runcount = 0
  for i = 1, size[2] do
    for j = 1, size[1] do
      if mask[j][i] == 1 then
        if runcount == 0 then
          output = output .. (i - 1)*size[1] + j .. ' '
        end
        runcount = runcount + 1
      else
        if runcount > 0 then
          output = output .. runcount .. ' '
          runcount = 0
        end
      end
    end
  end
  if runcount > 0 then
    output = output .. runcount .. ' '
  end
  return output
--  return string.match(output, '^(.-)%s?$')
end

return helpers
