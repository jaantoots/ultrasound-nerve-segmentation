local torch = require "torch"
local paths = require "paths"
local gm = require "graphicsmagick"

local data = {} -- Return variable

-- Vanilla validation set definition
data.validationSubjects = {}
for i = 39, 47 do
  data.validationSubjects[i] = true
end

local function iterImages (dname)
  -- Return an iterator over the TIFF images in directory `dname`
  return paths.files(dname,
    function (file) return string.match(file, '%.tif$') end)
end

function data._init (dir, height, width)
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
  for file in iterImages(data.dir) do
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

  print("Train", "Valid.", "Height", "Width")
  print(#data.train, #data.validation, data.height, data.width)
end

function data.serialize (file, names)
  -- Serialize images in `names` to `file`
  local inputs = torch.ByteTensor(#names, data.height, data.width)
  local labels = torch.ByteTensor(#names, data.height, data.width)
  -- Iterate through images to store them
  for i, name in pairs(names) do
    local image = data.dir .. '/' .. name
    inputs[i] = gm.Image(image .. '.tif'):
      size(data.width, data.height):toTensor('byte', 'I', 'HW')
    labels[i] = gm.Image(image .. '_mask.tif'):
      size(data.width, data.height):toTensor('byte', 'I', 'HW')
  end
  -- Store data in file
  local out = {index = names, inputs = inputs, labels = labels}
  torch.save(file, out)
end

function data._load (file)
  -- Load images from `file`
  data.data = torch.load(file)
  local size = data.data.inputs:size()
  data.size = size[1]
  data.height = size[2]
  data.width = size[3]
  print("Train", "Height", "Width")
  print(size[1], data.height, data.width)
end

function data.load (file, height, width)
  --[[ Load the data

  Parameters
  ----------
  file: file containing serialized images or directory containing the data
  height: height for resizing, ignored if loading from serialized file
  width: width for resizing, ignored if loading from serialized file
  --]]
  if paths.filep(file) then
    data._load(file)
  else
    data._init(file, height, width)
  end
end

function data.weights ()
  -- Calculate proportion of positive area as data is very unbalanced
  local means = {}
  for file in iterImages(data.dir) do
    if string.match(file, 'mask') then
      local mask = gm.Image(data.dir .. '/' .. file):
        size(data.width, data.height):toTensor('double', 'I', 'HW')
      means[#means + 1] = mask:mean()
    end
  end
  local mean = torch.Tensor(means):mean()
  -- Weights for classes in the corresponding order
  return {1, (1 - mean)/mean}
end

function data.normalize (mean, std)
  --[[ Normalize the images based on the training data

  Parameters
  ----------
  mean: (optional) mean to subtract from data, will be calculated if nil
  std: (optional) std to divide data, will be calculated if nil
  --]]
  if not mean or not std then
    local means = {}
    local stds = {}
    for file in iterImages(data.dir) do
      if not string.match(file, 'mask') then
        local img = gm.Image(data.dir .. '/' .. file):
          size(data.width, data.height):toTensor('double', 'I', 'HW')
        means[#means + 1] = img:mean()
        stds[#stds + 1] = img:std()
      end
    end
    mean = torch.Tensor(means):mean()
    std = torch.Tensor(stds):mean()
  end
  data.mean = mean
  data.std = std
  return data.mean, data.std
end

-- Initialize variables for nextImage
local shuffle
local iteration
local function nextImageDisk ()
  -- Return the next image from a random permutation of the training dataset
  if iteration == nil or iteration >= #data.train then
    -- Move to next epoch as necessary
    shuffle = torch.randperm(#data.train)
    iteration = 0
  end
  iteration = iteration + 1
  -- Load the images from files and normalize
  local image = data.dir .. '/' .. data.train[shuffle[iteration]]
  local input = gm.Image(image .. '.tif'):size(data.width, data.height):
    toTensor('double', 'I', 'HW'):add(-data.mean):div(data.std)
  local label = gm.Image(image .. '_mask.tif'):size(data.width, data.height):
    toTensor('double', 'I', 'HW')
  return input, label
end
local function nextImageMemory ()
  -- Return the next image from a random permutation of the training dataset
  if iteration == nil or iteration >= data.size then
    -- Move to next epoch as necessary
    shuffle = torch.randperm(data.size)
    iteration = 0
  end
  iteration = iteration + 1
  -- Load the images from files and normalize
  local i = shuffle[iteration]
  local input = data.data.inputs[i]:double():add(-data.mean):div(data.std)
  local label = data.data.labels[i]:double()
  return input, label
end

-- TODO: Downscaled images should fit to memory: faster not to read from disk
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
    if data.data then
      inputs[i][1], labels[i] = nextImageMemory()
    else
      inputs[i][1], labels[i] = nextImageDisk()
    end
  end
  labels = labels + 1 -- ClassNLLCriterion expects class labels starting at 1
  return {inputs = inputs, labels = labels}
end

return data
