-- rovernet top-level script

require 'torch'
require 'rnn'

local ffi = require 'ffi'

ffi.cdef[[
	int printf(const char *fmt, ...);
]]


--
-- network init
-- 
print('[rovernet]  running rovernet.lua init')
torch.setdefaulttensortype('torch.FloatTensor')


--
-- run the next iteration of the network (called from C)
--
function update_network( img_tensor )

	ffi.C.printf('[roverNet]  user_epoch(%f)\n', os.clock())

	img_dim    = img_tensor:dim()
	img_width  = img_tensor:size(1)	-- indexes start at 1 ;)
	img_height = img_tensor:size(2)

	ffi.C.printf('[rovernet]  dims %0.0f width %0.0f height %0.0f\n', img_dim, img_width, img_height)
	print(img_tensor)
	
end

