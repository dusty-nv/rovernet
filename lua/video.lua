-- rovernet top-level script


require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'image'


--
-- network init
-- 
print('[rovernet]  hello from within Torch/Lua environment (time=' .. os.clock() .. ')')

torch.setdefaulttensortype('torch.FloatTensor')


img_width   = 1280	
img_height  = 360


rovernet = nn:Sequential()
	rovernet:add(nn.Reshape(img_height*img_width))
	rovernet:add(nn.Linear(img_height*img_width, 100))
	rovernet:add(nn.ReLU())
	rovernet:add(nn.Linear(100, 9))
	rovernet:cuda()

criterion = nn.MSECriterion()
criterion:cuda()

--
-- run the next iteration of the network (called from C main loop)
--
function update_network( img_tensor )

	print('[rovernet]  user_epoch(' .. os.clock() .. ')')

	image.save("/home/ubuntu/test.jpg", img_tensor)

	--img_dim    = img_tensor:dim()
	--img_width  = img_tensor:size(1)	-- indexes start at 1 ;)
	--img_height = img_tensor:size(2)
	--img_width2 = img_width / 2
	--img_height2 = img_height / 2
	
	--print('[rovernet]  ' .. img_dim .. ' dims  (' .. img_width .. ' x ' .. img_height .. ')')
	--ffi.C.printf('[rovernet]  dims %0.0f width %0.0f height %0.0f\n', img_dim, img_width, img_height)
	--print(img_tensor)
	--print(nn)
	
	x = torch.ones(9)
	x:cuda()
	y = torch.CudaTensor(9); y:copy(x:narrow(1,1,9))
	--y = torch.FloatTensor(9); y:copy(x:narrow(1,1,9))	-- y = ideal motor states

	print('criterion:forward')
	output_states = rovernet:forward(img_tensor)
	criterion:forward(output_states, y)

	print('zeroGradParameters')
	rovernet:zeroGradParameters()

	--output2 = rovernet.output:float()
	print(rovernet.output)					-- rovernet.out = output_states
	print(y)

	print('criterion: backward')
	c = criterion:backward(rovernet.output, y)	

	print('rovernet:backward')
	rovernet:backward(img_tensor, c)

	print('rovernet:updateParameters')
	rovernet:updateParameters(0.0001)

	print('completed rovernet updates')
	
	--rovernet:backward(reward)
end

