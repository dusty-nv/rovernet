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


function update_work( imu_tensor, goal_tensor, output_tensor )

	print(output_tensor)


	print('output tensor')
end
