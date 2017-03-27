--
require 'nn'

--
-- this model was used in our ICPR paper
--

--
function get_3x32x32_to_64(p)
	--
	--
	--

	--
	local conv = nn.Sequential()

	conv:add(nn.SpatialConvolution(3, 32, 3, 3))
	conv:add(nn.ReLU())
	conv:add(nn.SpatialConvolution(32, 64, 4, 4, 2, 2))
	conv:add(nn.ReLU())
	conv:add(nn.SpatialConvolution(64, 128, 3, 3))
	conv:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	conv:add(nn.SpatialConvolution(128, 32, 1, 1))
	conv:add(nn.SpatialConvolution(32, 64, 6, 6))

	--
	local encoder = nn.Sequential()

	encoder:add(nn.MulConstant(1.0/255.0))
	encoder:add(nn.View(3, 32, 32))
	encoder:add(conv)
	encoder:add(nn.View(64))
	encoder:add(nn.Normalize(2))

	--
	if p then
		--
		local params = encoder:getParameters()
		params:copy(p)
	end

	--
    return encoder
end

--
net = get_3x32x32_to_64():float()
torch.save(arg[1], net)