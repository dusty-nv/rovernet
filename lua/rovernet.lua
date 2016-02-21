-- rovernet top-level script

require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'image'
require 'math'


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
learningRate = .00001


--Model Squashes for (0,1). Backwards motion 
--not supported at this time.
--First Network
r1 = nn.Sequential()
  r1:add(nn.Linear(inputSize, hiddenSize))
  r1:add(nn.Tanh())
  r1:add(nn.Linear(hiddenSize, outputSize))
  r1:add(nn.SoftMax())
  
--Second Network
r2 = nn.Sequential()
  r2:add(nn.Linear(inputSize, hiddenSize))
  r2:add(nn.Tanh())
  r2:add(nn.Linear(hiddenSize, outputSize))
  r2:add(nn.SoftMax())
  
criterion = nn.MSECriterion()

inputs = torch.FloatTensor(2)
target1 = torch.FloatTensor(1)
target2 = torch.FloatTensor(1)
print("Model1: ")
print(r1)

print("Model2: ")
print(r2)


--
-- run the next iteration of the network (called from C main loop)
-- Note:overrides C set goal tensor
function update_network( imu_tensor, goal_tensor, output_tensor )
  
  goal_tensor[1][1] = math.random()*360

	bearing = imu_tensor[1][1]
	goal    = goal_tensor[1][1]
	print( 'bearing:  ' .. bearing  )
  print( 'goal:     ' .. goal)
  
	--TO DUSTIN: Manage tensors for input here
	inputs[1] = bearing
	inputs[2] = goal
	

	output1 = r1:forward(inputs)
	output2 = r2:forward(inputs)

	print('motor1:  ' .. output1[1])
	print('motor2:  ' .. output2[1])

	print(output_tensor)

	
	print('output tensor')
	r1:zeroGradParameters()
	r2:zeroGradParameters()

--Here, PWideal represents the time it would take for the rover to turn a full 360 degrees. Needs to 
--be precalculated ahead of runtime and declared. This avoids the necessity of changing the softmax 
--to (softmax()-1)*2
  if (bearing < goal) then
    target1[1] = 0
    target2[1] = (bearing - goal) / 360 * PWideal
  else
    target1[1] = (bearing - goal) / 360 * PWideal
    target2[1] = 0
  
	print('gradout 1 breakpoint')
	--Online target calculation and backpropagation
	gradOutputs = criterion:backward(output1, target1)
	gradInputs = r1:backward(inputs, gradOutputs)
	
	print(gradOutputs)
	r1:updateParameters(learningRate)

	gradOutputs = criterion:backward(output2, target2)
	gradInputs = r2:backward(inputs, gradOutputs)

	r2:updateParameters(learningRate)
  
  --and iterate by sending a new goal_tensor to the function. It should be capable of generalizing.
end