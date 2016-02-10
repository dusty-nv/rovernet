--
-- 

require 'torch'


function ex_lua_func()
   ret = 'Called function inside lua'
   print()
   return ret
end

print("Hello from Lua")

torch.setdefaulttensortype('torch.FloatTensor')
test_tensor = torch.rand(3, 4, 2)
print(test_tensor)
