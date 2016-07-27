local torch = require "torch"
local paths = require "paths"
local gm = require "graphicsmagick"

local data = {} -- Return variable

-- Vanilla validation set definition
data.validationSubjects = {}
for i = 39, 47 do
  data.validationSubjects[i] = true
end

function data.init (dir, height, width)
  --[[Initialize data to read from dir and resize images to height and width

  Parameters
  ----------
  dir: directory containing the training images
  height: height for resizing
  width: width for resizing

  Returns
  -------
  Print number of images in training and validation sets and the specified size
  --]]
  data.dir = dir
  data.train = {}
  data.validation = {}
  for file in paths.iterfiles(data.dir) do
    local subject = tonumber(string.match(file, '%d+'))
    if not string.match(file, 'mask') then
      -- Divide according to subject number
      if data.validationSubjects[subject] then
        data.validation[#data.validation + 1] = string.match(file, '%d+_%d+')
      else
        data.train[#data.train + 1] = string.match(file, '%d+_%d+')
      end
    end
  end
  data.height = height
  data.width = width

  print("Train", "Valid.", "Width", "Height")
  print(#data.train, #data.validation, data.width, data.height)
end

function data.weights ()
  -- Calculate proportion of positive area as data is very unbalanced
  local means = {}
  for file in paths.iterfiles(data.dir) do
    if string.match(file, 'mask') then
      local mask = gm.Image(data.dir .. '/' .. file):size(
        data.width, data.height):toTensor('double', 'I', 'HW')
      means[#means + 1] = mask:mean()
    end
  end
  local mean = torch.Tensor(means):mean()
  -- Weights for classes in the corresponding order
  return torch.Tensor{1, (1 - mean)/mean}
end

-- Initialize variables for nextImage
local shuffle
local iteration
local function nextImage ()
  -- Return the next image from a random permutation of the training dataset
  if iteration == nil or iteration >= #data.train then
    -- Move to next epoch as necessary
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
    local image = data.dir .. '/' .. nextImage()
    inputs[i][1] = gm.Image(image .. '.tif'):size(
      data.width, data.height):toTensor('double', 'I', 'HW')
    labels[i] = gm.Image(image .. '_mask.tif'):size(
      data.width, data.height):toTensor('double', 'I', 'HW')
  end
  labels = labels + 1 -- ClassNLLCriterion expects class labels starting at 1
  return {inputs = inputs, labels = labels}
end

return data
