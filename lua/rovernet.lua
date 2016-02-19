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

--Attempt at a basic control solutions on a rover using neural networks. Utilizes simple
--MLP's to solve the problem. Outputs raw PWM between (-1,1). Inputs are currHeading and targetHeading
--Each network solves a motor (r1=motor1, r2=motor2) and outputs its PWM independently of the other
outputSize = 1
inputSize = 2
hiddenSize = 300
learningRate = .0001

--First Network
r1 = nn.Sequential()
  r1:add(nn.Linear(inputSize, hiddenSize))
  r1:add(nn.Tanh())
  r1:add(nn.Linear(hiddenSize, outputSize))
  
--Second Network
r2 = nn.Sequential()
  r2:add(nn.Linear(inputSize, hiddenSize))
  r2:add(nn.Tanh())
  r2:add(nn.Linear(hiddenSize, outputSize))
  
criterion = nn.MSECriterion()


print("Model1: ")
print(r1)

print("Model2: ")
print(r2)


--
-- run the next iteration of the network (called from C main loop)
--TO DUSTIN: input tensors need to be [0,360] to function properly
function update_network( imu_tensor, goal_tensor, output_tensor )

	--print('[rovernet]  user_epoch(' .. os.clock() .. ')')
	print( imu_tensor:size(1) .. ' ' .. goal_tensor:size(1) .. ' ' .. output_tensor:size(1) )
	
	bearing = imu_tensor[1][1]
	goal  = goal_tensor[1][1]
	print( 'bearing:  ' .. bearing )
  print( 'goal:     ' .. goal)
  
  --TO DUSTIN: Manage tensors for input here
  output1 = r1:forward(inputs)
  output2 = r2:forward(inputs)
  
  r1:zeroGradParameters()
  r2:zeroGradParameters()
  
  --Targets are computed by calculating minDiff(goal_tensor)-theta(other). Tensors should be stored as floats and passed back
  --For all of the fancy networks, this is really just linear regression to find the solution to two equations where we define
  --the targets as the ideal solution to the other and utilize learning rate to slow down the solution to prevent oscillations
  target1 = minDiff(goal_tensor, imu_tensor)-output2
  target2 = minDiff(goal_tensor, imu_tensor)-output1
  
  --Online target calculation and backpropagation
  gradOutputs = criterion:backward(output1, target1)
  gradInputs = r1:backward(inputs, gradOutputs)
  
  r1:updateParameters(learningRate)
  
  gradOutputs = criterion:backward(output2, target2)
  gradInputs = r2:backward(inputs, gradOutputs)
  
  r2:updateParameters(learningRate)
  
  --and iterate by sending a new goal_tensor to the function. It should be capable of generalizing.
end

function minDiff( target, current)
  if math.abs(target-current) < math.abs(current-target) then
      return math.abs(target-current) 
  else 
    return (math.abs(current-target)) 
  end
end
