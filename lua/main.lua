--rovernet top-level script

require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'image'

--What the saved network is named
network = torch.load('network.t7')

--Network init
print('[rovernet] hello from within Torch/Lua environment (time=' .. 
os.clock() .. ')')

torch.setdefaulttensortype('torch.FloatTensor')

print('[rovernet] loading previously trained network (time=' .. 
os.clock() .. ')')
Agent = torch.load(network)

--Check agent
print(Agent)
Agent:forward(image)


end
