-- rovernet top-level script

require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'image'
require 'rnn'

--
-- network init
-- 
print('[rovernet]  hello from within Torch/Lua environment (time=' .. os.clock() .. ')')

torch.setdefaulttensortype('torch.FloatTensor')



--
-- run the next iteration of the network (called from C main loop)
--
function update_network( imu_tensor, goal_tensor, output_tensor )

	--print('[rovernet]  user_epoch(' .. os.clock() .. ')')
	print( imu_tensor:size(1) .. ' ' .. goal_tensor:size(1) .. ' ' .. output_tensor:size(1) )
	
	bearing = imu_tensor[1][1]
	
	print( 'bearing:  ' .. bearing )

	--print( imu_tensor )
	--print( output_tensor )

	
end

