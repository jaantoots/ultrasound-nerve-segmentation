local torch = require 'torch'
local paths = require 'paths'
local gm = require 'graphicsmagick'

-- Get the number and the dimensions of images in the training set
local size = 0
local width, height
for file in paths.iterfiles('train') do
   if width == nil then
      width, height = gm.Image('train/' .. file):size()
   end
   size = size + 1
end
print(size, width, height)

-- Initialise storage
local images = torch.ByteTensor(size/2, width, height)
local masks = torch.ByteTensor(size/2, width, height)
local subjects = torch.Tensor(size/2)

-- Provide storage indexing for iterating through images
local nextIndex = 0
local function newIndex(index, file)
   nextIndex = nextIndex + 1
   index[file] = nextIndex
   return nextIndex
end
local index = {}
setmetatable(index, {__index = newIndex})

-- Iterate through images to store them in Tensors
for file in paths.iterfiles('train') do
   local image = gm.Image('train/' .. file)
   local i = index[string.match(file, '%d+_%d+')]
   if string.match(file, 'mask') then
      masks[i] = image:toTensor('byte', 'I')
   else
      images[i] = image:toTensor('byte', 'I')
      subjects[i] = string.match(file, '%d+')
   end
end

-- Store data in serialised file
local data = {data = images, mask = masks, subject = subjects}
local file = torch.DiskFile('train.t7', 'w')
file:writeObject(data)
file:close()
