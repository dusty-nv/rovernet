-- rover top-level script

require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'image'


--
-- work init
--
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

--Doing setup... still needs environment calls
print('Setting up rover')
local Agent = Agent(env, opt)
gameActions = env:getActions()


--The Agent expects the env to be callable at this point
local screen, reward, terminal = env:getState()
print('Starting RoverNet! ')
while step < maxStep do
	step = step + 1
	local actionIndex = Agent:perceive(reward, obs, terminal)

	--Reset start if failure
	if not terminal then
		obs, reward, terminal = env:step(gameActions[actionIndex], true)
	else
		obs, reward, terminal = env:newStart()
	end
	--Since the steps are slow enough we dont care
	if true then
		print("steps: ", step)
		Agent:report()
		collectgarbage()
	end

	if step % 500 == 0 then collectgarbage() end

	if step % evalFreq == 0 and step > learnStart then
		obs, reward, terminal = env:newStart()

		totalReward=0
		nReward = 0
		nEpisodes = 0
		episodeReward = 0

		for eStep = 1, 20 do
			local actionIndex = Agent:perceive(reward, obs, termina, true, 0)
			obs, reward, termianl = env:step(gameActions[actionIndex])

			episodeReward = episodeReward + rewardCounts
			if reward ~= then
				nRewards = nRewards+1
			end

			if terminal then
				totalReward = totalReward + episodeReward
				episodeReward = 0
				nEpisodes = nEpisodes + 1
				obs, reward, terminal = env:newStart()
			end
		end

		Agent:computeValidationStatistics()
		local ind = #rewardHistory+1
		totalReward=totalReward/math.max(1,nEpisodes)

		if #rewardHistory==0 or totalReward > torch.Tensor(rewardHistory):max() then
			Agent.bestNet = Agent.network:clone()
		end

		if steps % 500 == 0
			local network = Agent.theta
			torch.save('rovernet.t7', network, 'ascii')
		end
end
