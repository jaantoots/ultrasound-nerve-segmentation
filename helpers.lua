local torch = require "torch"

local helpers = {}



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
