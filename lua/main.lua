--
-- 

require 'torch'
local ffi = require 'ffi'

ffi.cdef[[
	int printf(const char *fmt, ...);
	void lua_ffi_msg( const char* );
	bool fillTensor(int rows, int cols, THByteTensor* emptyTensor);
]]

ffi.C.printf("ffi FFI Hello %s!\n", "world")


function ex_lua_func( x, y )
	print('HELLO from function inside LUA')
	print(x)
	print(y)
	return x * y
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
