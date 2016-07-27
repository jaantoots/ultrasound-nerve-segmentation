local torch = require "torch"
local paths = require "paths"
local json = require "json"
local argparse = require "argparse"

local helpers = {}

function helpers.parser ()
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

function helpers.opts (args)
  -- Return opts for training
  local opts
  if paths.filep(args.conf) then
    opts = json.load(args.conf)
  else
    opts = {}
  end
  opts.train = opts.train or 'train.t7'
  opts.output = args.output or opts.output or 'out/2016-07-27-serial'
  opts.height = opts.height or 200
  opts.width = opts.width or 280
  opts.batchSize = args.batch or opts.batchSize or 8
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

return helpers
