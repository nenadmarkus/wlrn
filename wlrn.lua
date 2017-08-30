--
require 'torch'

require 'cunn'
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------------------------------------
--------------------- parse command line options ---------------------------------------------------
----------------------------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text("Arguments")
cmd:argument("-m", "path to the model (in Torch7 nn format)")
cmd:argument("-t", "training/validation data-loading routines")
cmd:text("Options")
cmd:option("-w", "", "model write destination (in Torch7 nn format)")
cmd:option("-g", "", "GPU ID")
cmd:option("-n", "128", "number of training rounds")
cmd:option("-l", "0.0001", "learning rate")
cmd:option("-b", "32", "batch size")

params = cmd:parse(arg)

-- move computations to a specific GPU (if requested)
if params.g ~= "" then
	--
	cutorch.setDevice( tonumber(params.g) )
end

----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------

--
T = torch.load(params.m):cuda()
print('* model architecture:')
print(T)

-- if you don't have cuDNN installed, please comment out the following lines
-- (beware: cunn is *significantly* slower than cuDNN)
usecudnn = true
if usecudnn then
	require 'cudnn'
	cudnn.benchmark = true
	cudnn.fastest = true
	T = cudnn.convert(T, cudnn, function(module)
		return torch.type(module):find('SpatialBatchNormalization') -- don't use cudnn for batch normalization
	end)
end

--
pT, gT = T:getParameters()
print('* the model has ' .. pT:size(1) .. ' parameters')

function model_forward(triplet)
	--
	--
	local na=triplet[1]:size(1)
	local np=triplet[2]:size(1)
	local nn=triplet[3]:size(1)
	--
	local descs = T:forward( torch.cat(triplet, 1) )
	--
	return {
		descs:narrow(1, 1, na),
		descs:narrow(1, na+1, np),
		descs:narrow(1, na+np+1, nn)
	}
end

function model_backward(triplet, dloss)
	--
	--
	return T:backward(torch.cat(triplet, 1), torch.cat(dloss, 1))
end

function select_hard_negatives(triplet)
	--
	--
	local negs = {}
	for i=1, #triplet[3] do
		--
		table.insert(negs, T:forward(triplet[3][i]:cuda()):clone())
	end
	negs = torch.cat(negs, 1)
	--
	local _, inds = torch.max(torch.mm(T:forward(triplet[1]:cuda()), negs:t()), 2)
	inds = inds:squeeze():long()
	--
	return {triplet[1], triplet[2], torch.cat(triplet[3], 1):index(1, inds)}
end

----------------------------------------------------------------------------------------------------
----------------------------- loss computation -----------------------------------------------------
----------------------------------------------------------------------------------------------------

--
thr = 0.8
beta = -math.log(1.0/0.99 - 1)/(1.0-thr)

print('* matching threshold set to ' .. thr)

--
M = nn.Sequential()
M:add( nn.MM(false, true) )
M:add( nn.AddConstant(1.0)):add(nn.MulConstant(0.5) ) -- rescale similarities to [0, 1]
M:add( nn.Sequential():add(nn.AddConstant(-thr)):add(nn.MulConstant(beta)):add(nn.Sigmoid()) ) -- kill all scores below the threshold
M:add( nn.Max(2) )
M:add( nn.Contiguous() )
M:add( nn.Sum() )
-- Instead of adding a small constant, eps, just to the score of the positive bag,
-- we add a 1.0 to both the scores of the positive and negative bags.
-- This prevents the division-by-zero error and can be seen as a form of additive regularization:
-- https://en.wikipedia.org/wiki/Additive_smoothing
M:add( nn.AddConstant(1) )

--
C = nn.ConcatTable()
C:add(nn.Sequential():add(nn.ConcatTable():add(nn.SelectTable(1)):add(nn.SelectTable(3))):add(M:clone('weight','bias', 'gradWeight','gradBias')))
C:add(nn.Sequential():add(nn.ConcatTable():add(nn.SelectTable(1)):add(nn.SelectTable(2))):add(M:clone('weight','bias', 'gradWeight','gradBias')))

--
L = nn.Sequential()
L:add(C)
L:add(nn.CDivTable())
L = L:cuda()

--
function loss_forward(triplet)
	--
	return L:forward(triplet)[1]
end

function loss_backward(triplet)
	--
	return L:backward(triplet, torch.ones(1):cuda())
end

function compute_average_loss(triplets)
	--
	-- switch to validation mode
	--
	T:evaluate()
	L:evaluate()

	--
	local avgloss = 0.0

	for i=1, #triplets do
		--
		local triplet = triplets[i]
		if type(triplet[3]) == 'table' then
			--
			triplet = select_hard_negatives(triplet)
		end
		triplet = {triplet[1]:cuda(), triplet[2]:cuda(), triplet[3]:cuda()}

		--
		local descs = model_forward(triplet)

		avgloss = avgloss + loss_forward(descs)
	end

	avgloss = avgloss/#triplets

	--
	T:clearState()
	L:clearState()

	--
	return avgloss
end

----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------

function apply_optim_sgd_step(triplets, eta)
	--
	local feval = function(x)
		--
		if x ~= pT then
			pT:copy(x)
		end

		--
		gT:zero()

		local loss = 0.0

		for i=1, #triplets do
			--
			local triplet = {
				triplets[i][1]:cuda(),
				triplets[i][2]:cuda(),
				triplets[i][3]:cuda()
			}

			-- forward pass
			--
			local v = model_forward(triplet)
			--
			loss = loss + loss_forward(v)

			-- backward pass
			--
			local dloss = loss_backward(v)
			--
			model_backward(triplet, dloss)
		end

		--
		loss = loss/#triplets
		gT:div(#triplets)

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
	-- switch to train mode
	--
	T:training()
	L:training()

	--
	T:zeroGradParameters()

	--
	for i=1, niters do
		--
		local batch = {}

		for j=1, bsize do
			--
			local triplet = triplets[math.random(1, #triplets)]
			if type(triplet[3]) == 'table' then
				--
				triplet = select_hard_negatives(triplet)
			end
			table.insert(batch, triplet)
		end

		--
		apply_optim_sgd_step(batch, eta)
	end

	--
	T:clearState()
	L:clearState()
end

----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------

--
get_trn_triplets, get_vld_triplets = dofile(params.t)

--
time = sys.clock()
vtriplets = get_vld_triplets()
time = sys.clock() - time
print('* ' .. #vtriplets .. ' validation triplets generated in ' .. time .. ' [s]')

time = sys.clock()
ebest = compute_average_loss(vtriplets)
elast = ebest
print('* initial validation loss: ' .. ebest)

time = sys.clock() - time
print("    ** elapsed time: " .. time .. " [s]")

--
eta = tonumber(params.l)
bsize = tonumber(params.b)
nrounds = tonumber(params.n)


for i = 1, nrounds do
	--
	--
	print("* ROUND (" .. i .. ")")

	--
	time = sys.clock()
	ttriplets = get_trn_triplets()
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
	print("    ** average loss (vld): " .. e)

	if e<ebest then
		--
		if params.w ~= "" then
			--
			local clone = T:clone()
			clone:float()
			if usecudnn then
				clone = cudnn.convert(clone, nn)
			end
			--
			print("* saving the model to `" .. params.w .. "`")
			torch.save(params.w, clone)
		end

		--
		ebest = e
	end

	--
	elast = e

	--
	ttriplets = {}
	collectgarbage()
end