-- rovernet top-level script

require 'torch'
require 'cutorch'
require 'rnn'


--
-- network init
-- 
print('[rovernet]  hello from within Torch/Lua environment (time=' .. os.clock() .. ')')

torch.setdefaulttensortype('torch.FloatTensor')



--
-- run the next iteration of the network (called from C main loop)
--
function update_network( img_tensor )

	print('[rovernet]  user_epoch(' .. os.clock() .. ')')

	img_dim    = img_tensor:dim()
	img_width  = img_tensor:size(1)	-- indexes start at 1 ;)
	img_height = img_tensor:size(2)

	print('[rovernet]  ' .. img_dim .. ' dims  (' .. img_width .. ' x ' .. img_height .. ')')
	--ffi.C.printf('[rovernet]  dims %0.0f width %0.0f height %0.0f\n', img_dim, img_width, img_height)
	print(img_tensor)
	
end

