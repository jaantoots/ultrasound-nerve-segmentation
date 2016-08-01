local torch = require "torch"
require "cutorch"
require "nn"
require "cunn"
require "cudnn"
local argparse = require "argparse"

-- Parse arguments
local parser = argparse("asciimodel.lua",
  "Convert a binary model to ASCII.")
parser:argument("input", "Model to load.")
parser:argument("output", "Output model file.")
local args = parser:parse()

-- Load network from file
local net = torch.load(args.input)
torch.save(args.output, net, 'ascii')
