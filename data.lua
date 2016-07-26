local torch = require "torch"
local paths = require "paths"
local gm = require "graphicsmagick"

local data = {} -- Return variable
local dir = 'train'
data.height = 200
data.width = 280
data.train = {}
data.validation = {}

-- Vanilla validation set definition
data.validationSubjects = {}
for i = 39, 47 do
  data.validationSubjects[i] = true
end

-- Get the images and dimensions in the data.training set
for file in paths.iterfiles(dir) do
  local subject = tonumber(string.match(file, '%d+'))
  if not string.match(file, 'mask') then
    if data.validationSubjects[subject] then
      data.validation[#data.validation + 1] = string.match(file, '%d+_%d+')
    else
      data.train[#data.train + 1] = string.match(file, '%d+_%d+')
    end
  end
end
print("Train", "Valid.", "Width", "Height")
print(#data.train, #data.validation, data.width, data.height)

-- Initialize variables for returning images from a random permutation
local shuffle
local iteration = #data.train

local function nextImage ()
  -- Return the next image from the data.training dataset
  if iteration >= #data.train then -- Move to next epoch as necessary
    shuffle = torch.randperm(#data.train)
    iteration = 0
  end
  iteration = iteration + 1
  return data.train[shuffle[iteration]]
end

function data.batch (batchSize)
  --[[Return a minibatch of data.training data

  Parameters
  ----------
  batchSize: number of images in minibatch

  Returns
  -------
  batch: table
    inputs: torch.Tensor of size batchSize x width x height containing the input
        image for data.training
    labels: torch.Tensor of the same size containing class indices
  --]]
  local inputs = torch.Tensor(batchSize, 1, data.height, data.width)
  local labels = torch.Tensor(batchSize, data.height, data.width)
  for i = 1, batchSize do
    local image = dir .. '/' .. nextImage()
    inputs[i][1] = gm.Image(image .. '.tif'):size(
      data.width, data.height):toTensor('double', 'I', 'HW')
    labels[i] = gm.Image(image .. '_mask.tif'):size(
      data.width, data.height):toTensor('double', 'I', 'HW')
  end
  labels = labels + 1 -- ClassNLLCriterion expects class labels starting at 1
  return {inputs = inputs, labels = labels}
end

return data
