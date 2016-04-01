-- rover top-level script

require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'image'


print('[rover]  hello from within Torch/Lua environment (time=' .. os.clock() .. ')')
torch.setdefaulttensortype('torch.FloatTensor')


local learnStart = 0
local rewardCounts = {}
local episodeCounts = {}
local timeHistory = {}
local vHistory = {}
local qmaxHistory = {}
local tdHistory = {}
local rewardHistory = {}
local step = 0
local maxStep = 100000
local evalFreq = 100
timeHistory[1] = 0

local totalReward
local nRewards
local nEpisodes
local episodeReward



print('Setting up rover')

local agentArgs = {
	width    = 256,
	height   = 256,
	nActions = 5
}

local Agent = require 'Agent'
print(Agent)
local agent = Agent(agentArgs)



function update_network( input_tensor, reward_tensor, output_tensor )

	image.save('/home/ubuntu/Pictures/rpLIDAR-' .. os.time() .. '.jpg', input_tensor)
	
	--local screen, reward, terminal = env:getState()
	local reward   = reward_tensor[1][1]
	local terminal = reward_tensor[1][2]
	
	print('updating RoverNet')
	--while step < maxStep do
		step = step + 1
		local action = agent:perceive(reward, input_tensor, terminal)
		print('RoverNet decided action ' .. action)
		
		local l = 0
		local r = 0
		
		if action == 1 then --{0,0}
			l = 0
			r = 0
		elseif action == 2 then --{1,0} + Xdeg
			l = 1
			r = 1
		elseif action == 3 then
			l = -1
			r = 1
		elseif action == 4 then
			l = 1
			r = -1
		elseif action == 5 then
			l = -1
			r = -1
		end
		
		
		--Reset start if failure
		--[[if not terminal then
			input_tensor, reward, terminal = env:step(gameActions[actionIndex], true)
		else
			input_tensor, reward, terminal = env:newStart()
		end--]]
		--Since the steps are slow enough we dont care
		if true then
			print("steps: ", step)
			agent:report()
			--collectgarbage()
		end

		if step % 500 == 0 then
			torch.save('rovernet-'.. os.time() .. '.t7', agent.network, 'ascii')
			collectgarbage() 
		end

		--[[if step % evalFreq == 0 and step > learnStart then
			input_tensor, reward, terminal = env:newStart()

			totalReward=0
			nReward = 0
			nEpisodes = 0
			episodeReward = 0

			for eStep = 1, 20 do
				local actionIndex = agent:perceive(reward, input_tensor, terminal, true, 0)
				input_tensor, reward, termianl = env:step(gameActions[actionIndex])

				episodeReward = episodeReward + rewardCounts
				if reward ~= 0 then
					nRewards = nRewards+1
				end

				if terminal then
					totalReward = totalReward + episodeReward
					episodeReward = 0
					nEpisodes = nEpisodes + 1
					input_tensor, reward, terminal = env:newStart()
				end
			end

			Agent:computeValidationStatistics()
			local ind = #rewardHistory+1
			totalReward=totalReward/math.max(1,nEpisodes)

			if #rewardHistory==0 or totalReward > torch.Tensor(rewardHistory):max() then
				agent.bestNet = Agent.network:clone()
			end

			if steps % 500 == 0 then
				local network = agent.theta
				torch.save('rovernet-'. os.time() . '.t7', network, 'ascii')
			end
		end--]]
	--end
end
