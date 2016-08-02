local torch = require "torch"
local paths = require "paths"
local gm = require "graphicsmagick"

local Data = torch.class('Data')

function Data:__init (file, height, width, validationSubjects, isValidate)
  --[[ Load the data

  This high-level function handles loading images from either a serialized file
  or a directory containing the original data, depending on the first argument.

  Parameters
  ----------
  file: file containing serialized images or directory containing the data
  The following parameters are ignored if `file` is an existing file:
  height: height for resizing
  width: width for resizing
  validationSubjects: list of subjects in the validation set
  --]]
  if paths.filep(file) then
    self:_loadFile(file)
  else
    -- Initialize validation subjects definition
    self.validationSubjects = {}
    for _, subject in pairs(validationSubjects) do
      self.validationSubjects[subject] = true
    end
    self:_loadDir(file, height, width)
  end
  self.isValidate = isValidate or false
  -- Initialize variables for nextImage
  self.shuffle = nil
  self.iteration = self.size
end

local function iterImages (dname)
  -- Return an iterator over the TIFF images in directory `dname`
  return paths.files(dname,
    function (file) return string.match(file, '%.tif$') end)
end

function Data:_loadDir (dir, height, width)
  --[[Initialize data from `dir` and resize images to `height` and `width`

  Parameters
  ----------
  dir: directory containing the training images
  height: height for resizing
  width: width for resizing

  Returns
  -------
  Print number of images in training and validation sets and the specified size
  --]]
  self.dir = dir
  -- Create lists of filenames
  self.train = {}
  self.validate = {}
  for file in iterImages(self.dir) do
    local subject = tonumber(string.match(file, '%d+'))
    if not string.match(file, 'mask') then
      -- Divide according to subject number
      if self.validationSubjects[subject] then
        self.validate[#self.validate + 1] = string.match(file, '%d+_%d+')
      else
        self.train[#self.train + 1] = string.match(file, '%d+_%d+')
      end
    end
  end
  self.size = #self.train
  self.height = height
  self.width = width

  print("Train", "Valid.", "Height", "Width")
  print(#self.train, #self.validate, self.height, self.width)
end

function Data:serialize (file, names)
  -- Serialize images in `names` to `file`
  local inputs = torch.Tensor(#names, self.height, self.width)
  local labels = torch.Tensor(#names, self.height, self.width)
  -- Iterate through images to store them
  for i, name in pairs(names) do
    local image = self.dir .. '/' .. name
    inputs[i] = gm.Image(image .. '.tif'):
      size(self.width, self.height):toTensor('double', 'I', 'HW')
    labels[i] = gm.Image(image .. '_mask.tif'):
      size(self.width, self.height):toTensor('double', 'I', 'HW')
  end
  -- Store data in file
  local out = {index = names, inputs = inputs, labels = labels}
  torch.save(file, out)
end

function Data:_loadFile (file)
  -- Load images from `file`
  self.data = torch.load(file)
  local size = self.data.inputs:size()
  self.size = size[1]
  self.height = size[2]
  self.width = size[3]
  print("Images", "Height", "Width")
  print(self.size, self.height, self.width)
end

function Data:weights ()
  -- Calculate proportion of positive area in the data
  local means = {}
  for file in iterImages(self.dir) do
    if string.match(file, 'mask') then
      means[#means + 1] = gm.Image(self.dir .. '/' .. file):
        size(self.width, self.height):toTensor('double', 'I', 'HW'):mean()
    end
  end
  local mean = torch.Tensor(means):mean()
  -- Weights for classes in the corresponding order
  return {mean, 1 - mean}
end

function Data:normalize (mean, std)
  --[[ Normalize the images based on the training data

  Parameters
  ----------
  mean: (optional) mean to subtract from data, will be calculated if nil
  std: (optional) std to divide data, will be calculated if nil
  --]]
  if not mean or not std then
    local means = {}
    local stds = {}
    local function addImg (img)
        means[#means + 1] = img:mean()
        stds[#stds + 1] = img:std()
    end
    if self.data then
      for i = 1, self.size do
        addImg(self.data.inputs[i])
      end
    else
      for file in iterImages(self.dir) do
        if not string.match(file, 'mask') then
          addImg(gm.Image(self.dir .. '/' .. file):
            size(self.width, self.height):toTensor('double', 'I', 'HW'))
        end
    end
    end
    mean = torch.Tensor(means):mean()
    std = torch.Tensor(stds):mean()
  end
  self.mean = mean
  self.std = std
  return self.mean, self.std
end

function Data:_nextImageDisk ()
  -- Load the images from disk and normalize
  local image
  if self.isValidate then
    image = self.validate[self.shuffle[self.iteration]]
  else
    image = self.train[self.shuffle[self.iteration]]
  end
  local input = gm.Image(self.dir .. '/' .. image .. '.tif')
    :size(self.width, self.height):toTensor('double', 'I', 'HW')
    :add(-self.mean):div(self.std)
  local label = gm.Image(self.dir .. '/' .. image .. '_mask.tif')
    :size(self.width, self.height):toTensor('double', 'I', 'HW')
  return input, label, image
end

function Data:_nextImageMemory ()
  -- Return normalized images from loaded file
  local i = self.shuffle[self.iteration]
  local input = self.data.inputs[i]:add(-self.mean):div(self.std)
  local label = self.data.labels[i]
  local name = self.data.index[i]
  return input, label, name
end

function Data:batch (batchSize, noShuffle)
  --[[Return a minibatch of training data

  Parameters
  ----------
  batchSize: number of images in minibatch

  Returns
  -------
  batch: table
    inputs: torch.Tensor of size batchSize x width x height containing the input
        image for training
    labels: torch.Tensor of the same size containing class indices
  --]]
  local inputs = torch.Tensor(batchSize, 1, self.height, self.width)
  local labels = torch.Tensor(batchSize, self.height, self.width)
  local names = {}
  for i = 1, batchSize do
    -- Get the next image from a random permutation of the training dataset
    if self.iteration >= self.size then
      -- Move to next epoch as necessary
      if noShuffle then
        self.shuffle = torch.range(1, self.size)
      else
        self.shuffle = torch.randperm(self.size)
      end
      self.iteration = 0
    end
    self.iteration = self.iteration + 1
    if self.data then
      inputs[i][1], labels[i], names[i] = self:_nextImageMemory()
    else
      inputs[i][1], labels[i], names[i] = self:_nextImageDisk()
    end
  end
  labels = labels + 1 -- ClassNLLCriterion expects class labels starting at 1
  return {inputs = inputs, labels = labels}, names
end

return Data
