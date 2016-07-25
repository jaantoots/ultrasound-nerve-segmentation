local nn = require "nn"
local cudnn = require "cudnn"

local net = nn.Sequential()

function net:ConvolutionLayer (nInputPlane, nOutputPlane)
  -- SpatialConvolution layer with 3x3 kernel, stride 1, padding 1
  self:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
  -- Batch normalization with default arguments
  self:add(nn.SpatialBatchNormalization(nOutputPlane))
  -- ReLU activation
  self:add(nn.ReLU())
end

function net:DilatedConvolutionLayer (nInputPlane, nOutputPlane, dilation)
  -- SpatialConvolution layer with 3x3 kernel, stride 1, padding, and dilation
  local kernel = 3
  local pad = math.floor(dilation*kernel/2 - 1)
  self:add(nn.SpatialDilatedConvolution(nInputPlane, nOutputPlane,
      kernel, kernel, 1, 1, pad, pad, dilation, dilation))
  -- Batch normalization with default arguments
  self:add(nn.SpatialBatchNormalization(nOutputPlane))
  -- ReLU activation
  self:add(nn.ReLU())
end

-- Based on VGG ConvNet configuration D
-- 580x420
net:ConvolutionLayer(1, 64)
net:ConvolutionLayer(64, 64)
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- 290x210
net:ConvolutionLayer(64, 128)
net:ConvolutionLayer(128, 128)
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- 145x105
net:ConvolutionLayer(128, 256)
net:ConvolutionLayer(256, 256)
net:ConvolutionLayer(256, 256)
net:add(nn.Dropout(0.5))
-- as before
net:DilatedConvolutionLayer(256, 512, 2)
net:DilatedConvolutionLayer(512, 512, 2)
net:DilatedConvolutionLayer(512, 512, 2)
net:add(nn.Dropout(0.5))
-- as before
net:DilatedConvolutionLayer(512, 512, 4)
net:DilatedConvolutionLayer(512, 512, 4)
net:DilatedConvolutionLayer(512, 512, 4)
net:add(nn.Dropout(0.5))

-- Classification
net:add(nn.SpatialConvolution(512, 2, 1, 1))
net:add(nn.SpatialFullConvolution(2, 2, 8, 8, 4, 4, 2, 2))
-- back to 580x420

-- Weights initialization for convolutional layers
local function InitializeWeights (name)
  -- Initialize layer of type `name`
  for _, module in pairs(net:findModules(name)) do
    -- see arXiv:1502.01852 [cs.CV]
    local n = module.kW * module.kH * module.nOutputPlane
    module.weight:normal(0, math.sqrt(2/n))
    module.bias:zero()
  end
end
-- Initialize used types
InitializeWeights("nn.SpatialConvolution")
InitializeWeights("nn.SpatialDilatedConvolution")
InitializeWeights("nn.SpatialFullConvolution")

-- Move to GPU
cudnn.convert(net, cudnn)
net = net:cuda()

return net
