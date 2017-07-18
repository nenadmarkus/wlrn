--
require 'torch'
require 'image'
require 'cunn'

torch.setdefaulttensortype('torch.FloatTensor')

--
if #arg~=5 then
	--
	print('* usage: <net-in> <desc-dim> <dataset> <pca-proj-dim> <net-out>')
	do return end
end

--
net = torch.load(arg[1]):cuda()
net:evaluate()

--
DB = {}
p = io.popen('ls ' .. arg[3] .. '/*.jpg')
for path in p:lines() do
	if math.random() <= 0.4 then
		--
		-- load patches
		local data = image.load(path, 3, 'byte')
		data = data:view(3, data:size(2)/data:size(3), data:size(3)*data:size(3)):transpose(1, 2):contiguous()
		data = data:view(data:size(1), 3*data:size(3))
		--
		-- apply nn
		data = net:forward(data:cuda()):float()
		--
		-- store for later use
		table.insert(DB, data)
    end
end
p:close()

--
-- select a subset of descriptors
maxn = 32768
X = torch.zeros(maxn, tonumber(arg[2]))

for i=1, maxn do
	--
	j = math.random(1, #DB)
	k = math.random(1, DB[j]:size(1))
	--
	X[i] = DB[j][k]
end

--
-- do PCA
ndims = tonumber(arg[2])
projdim = tonumber(arg[4])
--
mean = torch.mean(X, 1):view(ndims)
X = X - torch.expand(torch.view(mean, 1, ndims), X:size(1), ndims)
--
cov = torch.mm(X:t(), X):div(X:size(1) - 1)
e, v = torch.symeig(cov, 'V')
e = e[{{ndims-projdim+1, ndims}}]
v = v[ {{1, ndims}, {ndims-projdim+1, ndims}} ]
proj = torch.mm(torch.sqrt(torch.diag(torch.cdiv(torch.ones(e:size()), 1e-6+e))), v:t())

--
-- append the linear, pca-projection layer to the net
linear = nn.Linear(tonumber(arg[2]), tonumber(arg[4])):float()
linear.weight = proj
linear.bias = -torch.mv(proj, mean)

net:add( linear )
net:add( nn.Normalize(2) )
net = net:float()

--x = torch.randn(4, 3, 32, 32)
--y = net:forward(x)
--print(y:size())

--
-- save the network to disk
net:clearState()
torch.save(arg[5], net)