local torch = require "torch"

local helpers = {}

function helpers.dice (outputs, targets)
  -- Calculate accuracy score as Dice coefficient
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
  local coeff = 2*torch.cdiv(nums, dens):squeeze()
  coeff[coeff:ne(coeff)] = 1 -- by definition if both sets are zero
  return coeff
end

return helpers
