local class = require 'classic'
local trans = classic.class('TransitionTable')

function trans:_init(args)

  print('TransitionTable:_init()')
  
  self.stateDim = args.stateDim
  self.numActions = args.numActions
  self.histLen = args.histLen
  self.maxSize = args.maxSize or 512^2
  self.bufferSize = args.bufferSize or 1024
  self.histType = "linear"
  self.histSpacing = args.histSpacing or 1
  self.zeroFrames = args.zeroFrames or 1
  self.nonTermProb = args.nonTermProb or 1
  self.nonEventProb = args.nonEventProb or 1
  self.numEntries = 0
  self.insertIndex = 0

  self.histIndicies = {}
  local histLen = self.histLen

  self.recentMemSize = self.histSpacing*histLen
  for i=1, histLen do
    self.histIndicies[i] = i*self.histSpacing
  end

  self.s = torch.ByteTensor(self.maxSize, self.stateDim):fill(0)
  self.a = torch.LongTensor(self.maxSize):fill(0)
  self.r = torch.zeros(self.maxSize)
  self.t = torch.ByteTensor(self.maxSize):fill(0)
  self.actionEncodings = torch.eye(self.numActions)

  self.recentS = {}
  self.recentA = {}
  self.recentT = {}

  local sSize = self.stateDim * histLen
  self.bufA = torch.LongTensor(self.bufferSize):fill(0)
  self.bufR = torch.zeros(self.bufferSize):fill(0)
  self.bufTerm = torch.ByteTensor(self.bufferSize, sSize):fill(0)
  self.bufS = torch.ByteTensor(self.bufferSize, sSize):fill(0)
  self.bufS2 = torch.ByteTensor(self.bufferSize, sSize):fill(0)

  print('done constructing TransitionTable')
end

function trans:GetNumActions()
	return self.numActions
end

function trans:reset()
  self.numEntries=0
  self.insertIndex=0
end

function trans:size()
  return self.numEntries
end

function trans:empty()
  return self.numEntries == 0
end

function trans:fillBuffer()
  print('trans.numEntries = ' .. self.numEntries .. ' trans.bufferSize = ' .. self.bufferSize)
  assert(self.numEntries >= self.bufferSize)
  self.bufInd = 1
  local ind
  for bufInd=1,self.bufferSize do
    local s, a, r, s2, term = self:sampleOne(1)
    self.bufS[bufInd]:copy(s)
    self.bufA[bufInd] = a
    self.bufR[bufInd] = r
    self.bufS2[bufInd]:copy(s2)
    self.bufTerm[bufInd] = term
  end
  self.bufS = self.bufS:float():div(255)
  self.bufS2 = self.bufS2:float():div(255)
end



function trans:sampleOne()
  assert(self.numEntries > 1)
  local index
  local valid = false
  while not valid do
    --Start at 2 becuase of previous action
    index = torch.random(2, self.numEntries-self.recentMemSize)
    if self.t[index+self.recentMemSize-1] == 0 then
      valid = true
    end
    if self.nonTermProb < 1 and self.t[index+self.recentMemSize] == 0 and
      torch.unform > self.nonTermProb then
        valid = false
    end
    if self.nonEventProb < 1 and self.t[index+self.recentMemSize] == 0 and
      self.r[index+self.recentMemSize-1]==0 and
      torch.unform() > self.nonTermProb then
        valid = false
    end
  end
  return self:get(index)
end


function trans:sample(batchSize)
    local batchSize = batchSize or 1
    assert(batchSize < self.bufferSize)

	if batchSize == 1 then
		print('trans:sample( batchSize=1 )')
	end
	
    if not self.bufInd or self.bufInd + batchSize - 1 > self.bufferSize then
      self:fillBuffer()
    end

    local index = self.bufInd

    self.bufInd = self.bufInd + batchSize
    local range = {{index, index+batchSize-1}}

    local bufS, bufS2, bufA, bufR, bufTerm = self.bufS, self.bufS2,
        self.bufA, self.bufR, self.bufTerm

    return bufS[range], bufA[range], bufR[range], bufS2[range], bufTerm[range]
end



function trans:concatFrames(index, useRecent)
  if useRecent then
    s, t = self.recentS, self.recentT
  else
    s, t = self.s, self.t
  end

  local fullstate = s[1].new()
  fullstate:resize(self.histLen, unpack(s[1]:size():totable()))

  --Zero out frames from all but the recent episode
  local zeroOut = false
  local episodeStart = self.histLen

  for i=self.histLen-1,1,-1 do
    if not zeroOut then
      for j=index+self.histIndicies[i]-1, index+self.histIndicies[i+1]-2 do
        if t[j] == 1 then
          zeroOut = true
          break
        end
      end
    end

    if zeroOut then
      fullstate[i]:zero()
    else
      episodeStart = 1
    end
  end
  
    --Get new frames
    for i = episodeStart, self.histLen do
      fullstate[i]:copy(s[index+self.histIndicies[i]-1])
    end

    return fullstate
end


function trans:concatActions(index, useRecent)
  local actHist = torch.FloatTensor(self.histLen, self.numActions)
  if useRecent then
    a, t = self.recentA, self.recentT
  else
    a, t = self.a, self.t
  end

  --Zero out all but most recent
  local zeroOut = false
  local episodeStart = self.histLen

  for i=self.histLen -1, 1, -1 do
    if not zeroOut then
      j=index+self.histIndicies[i]-1,index+self.histIndicies[i+1]-2 do
        if t[j]==1 then
          zeroOut = true
          break
        end
      end
    end

    if zeroOut then
      actHist[i]:zero()
    else
      episodeStart = i
    end
  end

  if self.zeroFrames == 0 then
    episodeStart = 1
  end

  --Copy current frames
  for i = episodeStart, self.histLen do
    actHist[i]:copy(self.actionEncodings[a[index+self.histIndicies[i]-1]])
  end

  return actHist
end


function trans:getRecent()
  return self:concatFrames(1, true):float():div(255)
end

function trans:get(index)
  local s = self:concatFrames(index)
  local s2 = self:concatFrames(index + 1)
  local ARindex = index+self.recentMemSize-1
  return s, self.a[ARindex], self.r[ARindex], s2, self.t[ARindex + 1]
end

function trans:add(s, a, r, term)
  assert(s, 'State cannot be nil')
  assert(a, 'Action cannot be nil')
  assert(r, 'Reward cannot be nil')

  print('trans:add(s, a=' .. a .. ', r, term)')
  print(s:size())

  --increment until full
  if self.numEntries < self.maxSize then
    self.numEntries = self.numEntries + 1
  end

  --Always insert at next index, then wrap
  self.insertIndex = self.insertIndex + 1
  --Overwrite oldest once at capacity
  if self.insertIndex > self.maxSize then
    self.insertIndex = 1
  end

  --Overwrite (s,a,r,t) at insertIndex
  print('overwriting @ index ' .. self.insertIndex)
  print('s:size = ')
  print(s:size())
  print('self.s:size = ')
  print(self.s:size())
  print('self.s[self.insertIndex]:size = ')
  print(self.s[self.insertIndex]:size())
  
  self.s[self.insertIndex] = s:clone():float():mul(255)
  print('done s')
  self.a[self.insertIndex] = a
  self.r[self.insertIndex] = r
  if term then
    self.t[self.insertIndex] = 1
  else
    self.t[self.insertIndex] = 0
  end
  print('done TransitionTable:add')
end




function trans:addRecentState(s, term)
	print('trans:addRecentState()')
  local s = s:clone():float():mul(255):byte()
  if #self.recentS == 0 then
    for i=1, self.recentMemSize do
      table.insert(self.recentS, s:clone():zero())
      table.insert(self.recentT, 1)
    end
  end

  table.insert(self.recentS, s)
  if term then
    table.insert(self.recentT, 1)
  else
    table.insert(self.recentT, 0)
  end

  --keep recentmemsize states
  if #self.recentS > self.recentMemSize then
      table.remove(self.recentS, 1)
      table.remove(self.recentT, 1)
  end
end


function trans:addRecentAction(a)
  if #self.recentA == 0 then
    for i=1, self.recentMemSize do
      table.insert(self.recentA, 1)
    end
  end

  table.insert(self.recentA, a)

  --keep recentmemsize steps
  if #self.recentA > self.recentMemSize then
    table.remove(self.recentA, 1)
  end
end


return trans

