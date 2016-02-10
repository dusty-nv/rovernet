--
-- 

require 'torch'
require 'ffi'


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
