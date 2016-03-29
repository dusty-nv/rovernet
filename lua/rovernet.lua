-- rover top-level script

require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'image'


--
-- work init
--
print('[rover]  hello from within Torch/Lua environment (time=' .. os.clock() .. ')')
torch.setdefaulttensortype('torch.FloatTensor')
histLen = 4
width = 256
height = 256
nChannels = 1
hiddenSize = 32
outputs = 3
--Really large number
maxSteps = 500000

reward_counts = {}
episode_counts = {}
v_history = {}
qmax_history = {}
td_history = {}
reward_history = {}
step = 0
time_history[1] = 0

--give preprocessed lidar, reward, and terminal flag
screen, reward, terminal = rover:getState()



local model = nn.Sequential()
model:add(nn.View(histLen, height, width))
model:add(nn.SpatialConvolution(histLen*nChannels, 32, 5, 5, 2, 2, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.SpatialConvolution(32,32,4,4,2,2))
model:add(nn.ReLU(true))
local convOutputSize = torch.prod(torch.Tensor(model:forward(torch.Tensor(torch.LongStorage({histLen*nChannels, height, width}))):size():totable()))
model:add(nn.View(convOutputSize))


local head = nn.Sequential()
head:add(nn.Linear(convOutputSize, hiddenSize))
head:add(nn.ReLU(true))
head:add(nn.Linear(hiddenSize, outputs))

local headConcat = nn.ConcatTable()
headConcat:add(head)
model:add(headConcat)




function update_work( imu_tensor, goal_tensor, output_tensor )

	print(output_tensor)


	print('output tensor')
end
