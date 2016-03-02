--
require 'torch'

require 'cunn'
require 'optim'

----------------------------------------------------------------------------------------------------
---------------- pairwise cosine similarity between keypoint bags ----------------------------------
----------------------------------------------------------------------------------------------------

local PCosSim, parent = torch.class('nn.PCosSim', 'nn.Module')

function PCosSim:__init()
	--
	parent.__init(self)

	--
	self.gradInput = {torch.Tensor(), torch.Tensor()}
end

function PCosSim:updateOutput(input)
	--
	local X = input[1]
	local Y = input[2]

	--
	self.output = X*Y:t()

	return self.output
end

function PCosSim:updateGradInput(input, gradOutput)
	--
	local X = input[1]
	local Y = input[2]

	--
	local dLdS = gradOutput

	local dLdX = dLdS*Y
	local dLdY = dLdS:t()*X

	--
	self.gradInput[1] = dLdX
	self.gradInput[2] = dLdY

	--
	return self.gradInput
end

----------------------------------------------------------------------------------------------------
--------------------- parse command line options ---------------------------------------------------
----------------------------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text("Arguments")
cmd:argument("-e", "encoder structure (neural net in Torch7 format)")
cmd:argument("-t", "training data")
cmd:text("Options")
cmd:option("-v", "", "validation data")
cmd:option("-r", "", "read weights in Torch7 format")
cmd:option("-w", "", "write weights in Torch7 format")
cmd:option("-n", "", "number of epochs")

params = cmd:parse(arg)

----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------

--
torch.setdefaulttensortype('torch.FloatTensor')

--
E = torch.load(params.e):float()
print(E)

--
E3 = nn.ParallelTable()
E3:add(E:clone('weight', 'bias', 'gradWeight', 'gradBias'))
E3:add(E:clone('weight', 'bias', 'gradWeight', 'gradBias'))
E3:add(E:clone('weight', 'bias', 'gradWeight', 'gradBias'))

--
M = nn.Sequential()

thr = 0.8
beta = -math.log(1.0/0.99 - 1)/(1.0-thr)

--M:add( nn.PCosSim() ) -- we don't need PCosSim ???
M:add( nn.MM(false, true) )
M:add( nn.AddConstant(1.0)):add(nn.MulConstant(0.5) ) -- rescale similarities to [0, 1]
M:add( nn.Sequential():add(nn.AddConstant(-thr)):add(nn.MulConstant(beta)):add(nn.Sigmoid()) ) -- kill all scores below the threshold
M:add( nn.Max(2) )

--
C = nn.ConcatTable()

C:add(nn.Sequential():add(nn.ConcatTable():add(nn.SelectTable(1)):add(nn.SelectTable(2))):add(M:clone('weight','bias', 'gradWeight','gradBias')))
C:add(nn.Sequential():add(nn.ConcatTable():add(nn.SelectTable(1)):add(nn.SelectTable(3))):add(M:clone('weight','bias', 'gradWeight','gradBias')))

--
T = nn.Sequential()
T:add(E3)
T:add(C)

-- cuda
T = T:cuda()

-- if you don't have cuDNN installed, please comment out the following four line (beware: cunn is *significantly* slower than cuDNN)
require 'cudnn'
cudnn.benchmark = true
cudnn.fastest = true
cudnn.convert(T, cudnn)

--
pT, gT = T:getParameters()

if params.r ~= "" then
	--
	print("* loading model parameters from '" .. params.r .. "'")

	p = torch.load(params.r)

	--
	pT:copy(p)
end

print('* the model has ' .. pT:size()[1] .. ' parameters')

--
function model_forward(triplet)
	--
	--
	return T:forward( triplet )
end

function model_backward(triplet, dloss)
	--
	--
	return T:backward(triplet, dloss)
end

---
---
---

----------------------------------------------------------------------------------------------------
----------------------------- loss computation -----------------------------------------------------
----------------------------------------------------------------------------------------------------

eps = 1e-6

function loss_forward(v)
	--
	n = v[1]:size()[1]

	--
	sap = v[1]:sum()/n
	san = v[2]:sum()/n

	--
	return san/(sap + eps)
end

function loss_backward(v)
	--
	gap = torch.ones(n):mul(-san/(sap+eps)^2):mul(1.0/n)
	gan = torch.ones(n):mul(1.0/(sap+eps)):mul(1.0/n)

	--
	gap = gap:cuda()
	gan = gan:cuda()

	--
	return {gap, gan}
end

function compute_average_loss(triplets)
	--
	local avgloss = 0.0

	for i=1, #triplets do
		--
		local triplet = {
			triplets[i][1]:clone():cuda(),
			triplets[i][2]:clone():cuda(),
			triplets[i][3]:clone():cuda()
		}

		--
		local v = model_forward(triplet)

		avgloss = avgloss + loss_forward(v)
	end

	avgloss = avgloss/#triplets

	--
	return avgloss
end

----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------

function generate_triplets(bags, n)
	--
	local i, j

	--
	if not n then
		--
		n = #bags
	else
		n = math.min(n, #bags)
	end

	--
	-- select a random subset of bags (this is important for memory issues)
	--

	local p = torch.randperm(#bags)
	local subset = {}

	for i=1, n do
		--
		local bag = {}

		bag.magic = bags[ p[i] ].magic
		bag.label = bags[ p[i] ].label
		bag.filename = bags[ p[i] ].filename

		--
		bag.data = bags[ p[i] ].data

		--
		subset[1+#subset] = bag
	end

	--
	local triplets = {}

	for i=1, #subset do
		for j=i+1, #subset do
			if subset[i].label == subset[j].label then
				for k=1, 3 do -- generate 3 triplets for each matching pair
					--
					local stop = false

					while not stop do
						--
						q = math.random(1, #subset)

						--
						if subset[i].label ~= subset[q].label then
							--
							stop = true
						end
					end

					--
					table.insert(triplets, {subset[i].data, subset[j].data, subset[q].data})
				end
			end
		end
	end

	--
	return triplets
end

function apply_optim_sgd_step(triplets, batch, eta)
	--
	local feval = function(x)
		--
		if x ~= pT then
			pT:copy(x)
		end

		--
		gT:zero()

		local loss = 0.0

		for i=1, #batch do
			--
			local triplet = {
				triplets[batch[i]][1]:clone():cuda(),
				triplets[batch[i]][2]:clone():cuda(),
				triplets[batch[i]][3]:clone():cuda()
			}

			-- forward pass
			--

			--
			local v = model_forward(triplet)

			--
			loss = loss + loss_forward(v)

			-- backward pass
			--

			--
			local dloss = loss_backward(v)

			--
			model_backward(triplet, dloss)

			--
		end

		--
		loss = loss/#batch
		gT:div(#batch)

		-- regularization
		--coeff = 1e-4
		--loss = loss + coeff*torch.norm(x)^2/2
		--gT:add( x:clone():mul(coeff) )

		--
		return loss, gT
	end

	--
	cfg = cfg or {}
	cfg.learningRate = eta
	optim.rmsprop(feval, pT, cfg)
end

function train_with_sgd(triplets, niters, bsize, eta)
	--
    local time = sys.clock()

	--
	T:zeroGradParameters()

	--
	for i=1, niters do
		--
		local batch = {}

		for j=1, bsize do
			table.insert(batch, math.random(1, #triplets))
		end

		--
		apply_optim_sgd_step(triplets, batch, eta)
	end

	--
end

----------------------------------------------------------------------------------------------------
----------------------------- initialize stuff -----------------------------------------------------
----------------------------------------------------------------------------------------------------

--
print('* thr = ' .. thr)

--
time = sys.clock()
tbags = torch.load(params.t) --load_bags(params.t)
time = sys.clock() - time
print('* training dataset loaded from "' .. params.t .. '" in ' .. time .. ' [s]')

if params.v ~= "" then
	--
	time = sys.clock()
	vbags = torch.load(params.v) --load_bags(params.v)
	time = sys.clock() - time
	print('* validation dataset loaded from "' .. params.v .. '" in ' .. time .. ' [s]')
else
	--
	vbags = tbags
end

--
if params.n ~= "" then
	--
	nrounds = tonumber(params.n)
else
	--
	nrounds = 128
end

----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------

--
time = sys.clock()
vtriplets = generate_triplets(vbags)
time = sys.clock() - time
print('* ' .. #vtriplets .. ' triplets generated in ' .. time .. ' [s]')

time = sys.clock()
ebest = compute_average_loss(vtriplets)
elast = ebest
print('* initial score: ' .. ebest)

time = sys.clock() - time
print("    ** elapsed time: " .. time .. " [s]")

--
eta = 1e-3
bsize = 8

for i = 1, nrounds do
	--
	--
	print("* SGD (" .. i .. ")")

	--
	time = sys.clock()
	ttriplets = generate_triplets(tbags, #tbags/3)
	time = sys.clock() - time
	print('    ** ' .. #ttriplets .. ' triplets generated in ' .. time .. ' [s]')

	--
	print('    ** eta=' .. eta .. ', bsize=' .. bsize)

	--
	time = sys.clock()
	train_with_sgd(ttriplets, 512, bsize, eta)
	time = sys.clock() - time

	print("    ** elapsed time: " .. time .. " [s]")

	e = compute_average_loss(ttriplets)
	print("    ** average loss (trn): " .. e)
	e = compute_average_loss(vtriplets)
	print("    ** average loss (val): " .. e)

	if e<ebest then
		--
		if params.w ~= "" then
			torch.save(params.w, pT:float())
		end

		--
		ebest = e

		--
	elseif elast < e then
		--
		if 64==bsize then
			--
			eta = eta/2.0
			--
			bsize = 8
		else
			--
			bsize = 2*bsize
		end

		--
	end

	--
	elast = e

	--
	ttriplets = {}

	collectgarbage()

	--
end
--