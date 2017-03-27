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
cmd:argument("-e", "path to convnet (in Torch7 nn format)")
cmd:argument("-t", "training/validation data-loading routines")
cmd:text("Options")
cmd:option("-w", "", "write weights in Torch7 format")
cmd:option("-n", "", "number of training rounds")
cmd:option("-g", "", "GPU ID")

params = cmd:parse(arg)

----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------

--
T = torch.load(params.e):cuda()
print('* model architecture:')
print(T)

-- if you don't have cuDNN installed, please comment out the following lines
-- (beware: cunn is *significantly* slower than cuDNN)
require 'cudnn'
cudnn.benchmark = true
cudnn.fastest = true
T = cudnn.convert(T, cudnn, function(module)
	return torch.type(module):find('SpatialBatchNormalization') -- don't use cudnn for batch normalization
end)

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
		local triplet = {
			triplets[i][1]:cuda(),
			triplets[i][2]:cuda(),
			triplets[i][3]:cuda()
		}

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
				triplets[batch[i]][1]:cuda(),
				triplets[batch[i]][2]:cuda(),
				triplets[batch[i]][3]:cuda()
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
			table.insert(batch, math.random(1, #triplets))
		end

		--
		apply_optim_sgd_step(triplets, batch, eta)
	end

	--
	T:clearState()
	L:clearState()
end

----------------------------------------------------------------------------------------------------
----------------------------- initialize stuff -----------------------------------------------------
----------------------------------------------------------------------------------------------------

--
get_trn_triplets, get_vld_triplets = dofile(params.t)

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
eta = 1e-4
bsize = 16

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
			print("* saving the model to `" .. params.w .. "`")
			--
			T:float()
			T = cudnn.convert(T, nn)
			--
			torch.save(params.w, T)
			--
			T = cudnn.convert(T, cudnn)
			T:cuda()
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
			bsize = 16
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
end