local classic = require 'classic'

local Rover, super = classic.class('Rover', env)

--Constructor
function Rover:_init(opts)
  opts = opts or {}

  print("Rover.lua -- environment constructor")
  
  --Noise
  self.noise = opts.noise or 0

  --Width and Height of Environment 9square)
  self.size = opt.size or 50
  self.screen = torch.Tensor(1, self.size, self.size):zero()

  --Player State
  self.player = {
    x = math.ceil(self.size / 2)
    y = math.ceil(self.size / 2)
    angle = 0
  }
end

function Rover:start()

end

function Rover:getActionSpec()
  return {'float', 1, {0, 1, 2}}
end

function Rover:getRewardSpec()
  return 0,99999999999
end

function Rover:step(action)
  --Reward is 0 by default
  local reward = 0

  --Each action corresponds to {0,1},{1,0},{1,1}
  if action == 1 then --{0,1}
    --compute angle changes here
  elseif action == 2 then --{1,0} + Xdeg
    --compute angle changes here
  elseif action == 3 then
    --compute movement here
end


function Rover:redraw()
  --Generate map during runtime map here
end

function Rover:start()
self.player.x = math.ceil(self.size /2)
self.player.y = math.ceil(self.size/2)

--Generate Initial mappings above
self:redraw()
end


end