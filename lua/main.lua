-- example test script
-- (do not use)

require 'torch'
local ffi = require 'ffi'

ffi.cdef[[
	int printf(const char *fmt, ...);
	void lua_ffi_msg( const char* str );
	bool fillTensor(int rows, int cols, THByteTensor* emptyTensor);
	THByteTensor* nextTensor(int rows, int cols);
]]

ffi.C.printf("ffi FFI Hello %s!\n", "world")


function ex_lua_func( x, y )
	print('HELLO from function inside LUA')
	print(x)
	print(y)
	return x * y
end

function user_epoch( img_tensor )

	ffi.C.printf('[roverNet]  user_epoch(%f)\n', os.clock())

	img_dim    = img_tensor:dim()
	img_width  = img_tensor:size(1)	-- indexes start at 1 ;)
	img_height = img_tensor:size(2)

	ffi.C.printf('[roverNet]  dims %0.0f width %0.0f height %0.0f\n', img_dim, img_width, img_height)
	print(img_tensor)
	

end

print("Hello from Lua")
print(my_square(4))

torch.setdefaulttensortype('torch.FloatTensor')
test_tensor = torch.rand(3, 4, 2)
print(test_tensor)

ffi.C.lua_ffi_msg("AL0HA!")

test_tensor2 = torch.ByteTensor()
ffi.C.fillTensor(4, 5, test_tensor2:cdata())
print(test_tensor2)
print(torch.numel(test_tensor2))
print(torch.trace(test_tensor2))
