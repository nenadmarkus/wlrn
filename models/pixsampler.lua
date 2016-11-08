require 'cunn'
require 'stn' -- https://github.com/qassemoquab/stnbhwd

--
local PixSampler, parent = torch.class('nn.PixSampler', 'nn.Module')

function PixSampler:__init(n)
	--
	self.n = n
	self.yx = torch.rand(n, 2):mul(2):add(-1):cuda()

	self.sampler = nn.BilinearSamplerBHWD()

	self.weight = self.yx
	self.gradWeight = torch.CudaTensor(self.yx:size()):zero()

	--
	self.gradInput = torch.CudaTensor()
end

function PixSampler:updateOutput(input)
	--
	local nbatch = input:size(1)
	local idepth = input:size(4)

	--
	self.grids = torch.CudaTensor(nbatch, 1, self.n, 2)
	self.grids = self.yx:view(1, 1, self.n, 2):expandAs(self.grids)

	-- +1e-6 to avoid division by zero
	self.output = 1e-6 + self.sampler:forward({input, self.grids}):view(nbatch, self.n, idepth)

	--
	return self.output
end

function PixSampler:updateGradInput(input, gradOutput)
	--
	local nbatch = input:size(1)
	local idepth = input:size(4)

	--
	local pair = self.sampler:backward({input, self.grids}, gradOutput:view(nbatch, 1, self.n, idepth))

	--
	self.gradInput = pair[1]
	self.gradWeight:add( pair[2]:mean(1):view(self.yx:size()) )

	--
	return self.gradInput
end

--
function get_pixdiff(p)
	--
	--
	--

	local nchn = 3

	--
	local encoder = nn.Sequential()

	--
	encoder:add( nn.View(nchn, 32, 32) )

	-- pixel-diff sampler
	pds = nn.Sequential()
	pds:add( nn.Transpose({2, 4}, {2, 3}) )
	pds:add( nn.ConcatTable():add(nn.PixSampler(256)):add(nn.PixSampler(256)) )
	pds:add( nn.ConcatTable():add(nn.CSubTable()):add(nn.CAddTable()) )
	pds:add( nn.CDivTable() )

	--
	encoder:add(pds)
	encoder:add( nn.View(nchn*256) )

	--
	encoder:add( nn.Linear(nchn*256, 128, false) )
	encoder:add( nn.ReLU() )
	encoder:add( nn.Linear(128, 128, false) )
	encoder:add( nn.ReLU() )
	encoder:add( nn.Linear(128, 128, false) )

	--
	encoder:add(nn.Normalize(2))

	--
	if p then
		--
		local params = encoder:getParameters()
		params:copy(p)
	end

	--
    return encoder:cuda()
end

--
--x = torch.rand(4, 3, 32, 32):cuda()
--n = get_pixdiff()
--y = n:forward(x)

--
return get_pixdiff