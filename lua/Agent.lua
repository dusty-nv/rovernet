require 'classic'

local Agent = classic.class('Agent')

function Agent:_init(opt)
  self.height = opt.height
  self.width = opt.wiidth
  self.nActions = opt.nActions

  --Epsion Annealing
  self.epStart = 1
  self.ep= 1 --Starting epsilon value
  self.epEnd = 0
  self.epT = 10000
  self.epEndt = 10000

  --Learning rate Annealing
  self.lrStart = .01
  self.lrEnd = 0
  self.lrEndT = 10000
  self.minibatchSize = 1
  self.validSize = 500

  --Q params
  self.gamma = .99
  self.updateFreq = 1
  self.nReplay = 1
  self.learnStart = 0

  --Transition table
  self.replayMemory = 1000
  self.histLen = 4
  self.maxReward = 5
  self.minReward = 0
  self.bestQ = 0

  self.nChannels = 1
  self.inputDims = {self.histLen * self.nChannels, self.height, self.width}
  self.histSpacing = 1
  self.nonTermProb = 1
  self.bufferSize = 206

  self.transitionParams = {}
  self.network = self:createNetwork()

  print('creating Agent... ' .. self.network)
  self.network = err
  self.network = self:network()

  self.network:float()
  self.tensor_type = torch.FloatTensor

  local transitionArgs = {
        stateDim = 1, numActions = self.nActions,
        histLen = self.histLen, gpu = 0,
        maxSize = self.replayMemory, histType = self.histType,
        histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
        bufferSize = self.bufferSize
    }

    self.transitions = Agent.TransitionTable(transitionArgs)
    self.numSteps = 0
    self.lastState = nil
    self.lastAction = nil
    self.vAvg = 0
    self.tdErrAvg = 0

    self.qMax = 1
    self.rMax = 1

    self.theta, self.dTheta = self.network:getParameters()
    self.dTheta:zero()

    self.deltas = self.dw:clone():fill(0)

    self.tmp = self.dw:clone():fill(0)
    self.g = self.dw:clone():fill(0)
    self.g2 = self.dw:clone():fill(0)

end

function Agent:reset(state)
  if not state then
    return
  end
  self.bestNet = state.bestNet
  self.network = state.model
  self.theta, self.dTheta = self.network:getParameters()
  self.dTheta:zero()
  self.numSteps = 0
  print('Reset agent successfully')
end

function Agent:update(args)
  local s, a, r, s2, term, delta
  local q, q2, q2Max

  s = args.s
  a = args.a
  r = args.r
  s2 = args.s2
  term = args.term

  -- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
  term = term:clone():float():mul(-1):add(1)

  targetQ = self.network

  --Comput max_a Q(s_2,a)
  q2Max = targetQ:forward(s2):float():max(2)
  -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
  q2 = q2Max:clone():mul(self.discount):cmul(term)
  delta = r:clone():float()
  delta:add(q2)

   -- q = Q(s,a)
   local qAll = self.network:forward(s):float()
   q = torch.FloatTensor(qAll:size(1))
   for i=1,qAll:size(1) do
       q[i] = qAll[i][a[i]]
   end
   delta:add(-1, q)

   local targets = torch.zeros(self.minibatchSize, self.nActions):float()
   for i=1,math.min(self.minibatchSize,a:size(1)) do
      targets[i][a[i]] = delta[i]
   end

   return targets, delta, q2Max
 end

function Agent:learn()
  --Perform a minibatch update

  local s, a, r, s2, term = self.transitions:sample(self.minibatchSize)

  local targets, delta, q2Max = self:update{s=s, a=a, r=r, s2=s2, term=term, updateQmax=true}

  --Zero grab params
  self.dTheta:zero()
  --Compute new gradient
  self.network:backwards(s, targets)

  --Anneal Learning rate
  local t = math.max(0, self.numSteps - self.learnStart)
  self.lr = (self.lrStart - self.lrEnd) * (self.lrEndT - t)/(self.lrEndT + self.lrEnd)
  self.lr = math.max(self.lr, self.lrEnd)

  --Compute RMSprop
  self.g:mul(0.95):add(0.05, self.dw)
  self.tmp:cmul(self.dw, self.dw)
  self.g2:mul(0.95):add(0.05, self.tmp)
  self.tmp:cmul(self.g, self.g)
  self.tmp:mul(-1)
  self.tmp:add(self.g2)
  self.tmp:add(0.01)
  self.tmp:sqrt()

  --Accumulate update
  self.deltas:mul(0):addcdiv(self.lr, self.dTheta, self.tmp)
  self.theta:add(self.deltas)
end

function Agent:sampleValid()
  local s, a, r, s2, term = self.transitions:sample(self.validSize)
  self.validS = s:clone()
  self.validA = a:clone()
  self.validR = r:clone()
  self.validS2 = s2:clone()
  self.validTerm = term:clone()
end

function Agent:computeValid()
  local targets, delta, q2Max = self:update{s=self.validS, a = validA, r=self.validR, s2=self.valids2, term = self.validTerm}
  self.vAvg=self.qMax * q2Max:mean()
  self.tdErrAvg = delta:clone():abs():mean()
end

function Agent:perceive(reward, state, terminal, testing, testingEp)
  local curState
  if self.maxReward then
    reward = math.min(reward, self.maxReward)
  end
  if self.minReward then
    reward = math.max(reward, self.minReward)
  end

  self.transitions:addRecentState(state, terminal)
  local currentFullState = self.transitions:getRecent()

  --store transitions
  if self.lastState and not testing then
    self.transitions:add(self.lastState, self.lastAction, reward, self.lastTerminal, priority)
  end

  if self.numSteps == self.learnStart + 1 and not testing then
    self.sampleValid()
  end

  curState = self.transitions:getRecent()
  curState = curstate:resize(1, unpack(self.inputDims))

  local actionIndex = 1
  if not terminal then
    actionIndex = self.eGreedy(curState, testingEp)
  end

  self.transitions:addRecentAction(actionIndex)

  --Update q learner
  if self.numSteps > self.learnStart and not testing and self.numSteps % self.updateFreq == 0 then
    for i = 1, self.nReplay do
       self:learn()
     end
   end

   if not testing then
     self.numSteps = self.numSteps + 1
   end

   self.lastState = state:clone()
   self.lastAction = actionIndex
   self.lastTerminal = terminal

   if self.target_q and self.numSteps % self.target_q == 1 then
    self.target_network = self.network:clone()
  end

  if not terminal then
    return actionIndex
  else
    return 0
  end
end

function Agent:eGreedy(state, testingEp)
  self.ep = testingEp or (self.epEnd +
                          math.max(0, (self.epStart - self.epEnd) * (self.epEndt -
                          math.max(0, self.numSteps - self.learnStart))/self.epEndt))
          if torch.uniform() < self.ep then
            return torch.random(1, self.nActions)
          else
            return self:greedy(state)
          end
end

function Agent:greedy(state)
  if state:dim() == 2 then
    assert(false, 'input must be at least 3d')
    state = state:resize(1, state:size(1), state:size(2))
  end

  local q = self.network:forward(state):float():squeeze()
  local maxq = q[1]
  local besta = {1}


    -- Evaluate all other actions (with random tie-breaking)
    for a = 2, self.n_actions do
        if q[a] > maxq then
            besta = { a }
            maxq = q[a]
        elseif q[a] == maxq then
            besta[#besta+1] = a
        end
    end
    self.bestQ = maxq

    local r = torch.random(1, #besta)

    self.lastAction = besta[4]

    return besta[r]
end

function Agent:report()
    print(get_weight_norms(self.network))
    print(get_grad_norms(self.network))
end

function Agent:createNetwork()
  local hiddenSize = 128
  model:add(nn.View(histLen, height, width))
  model:add(nn.SpatialConvolution(histLen*nChannels, 32, 5, 5, 2, 2, 1, 1))
  model:add(nn.ReLU(true))
  model:add(nn.SpatialConvolution(32,32,4,4,2,2))
  model:add(nn.ReLU(true))
  model:add(nn.Linear(convOutputSize, hiddenSize))
  model:add(nn.ReLU(true))
  model:add(nn.Linear(hiddenSize, nActions))
  return model
end
