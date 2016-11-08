--
require 'nn'

--
-- this model was used in the ECCV2016 workshop "Local Features: State of the art, open problems and performance evaluation"
-- the parameters of the model were learned with the code that can be found in this repository
-- (training set: HPatches training set)
--

--
function get_desc(p)
	--
	--
	--

	--
	local conv = nn.Sequential()

	conv:add(nn.SpatialConvolution(1, 16, 3, 3, 1, 1, 1, 1))
	conv:add(nn.Tanh())
	conv:add(nn.SpatialConvolution(16, 32, 5, 5, 2, 2, 2, 2))
	conv:add(nn.Tanh())
	conv:add(nn.SpatialConvolution(32, 64, 3, 3))
	conv:add(nn.Tanh())
	conv:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	conv:add(nn.SpatialConvolution(64, 64, 5, 5, 2, 2, 2, 2))
	conv:add(nn.Tanh())
	conv:add(nn.SpatialConvolution(64, 64, 3, 3))
	conv:add(nn.Tanh())

	--
	local encoder = nn.Sequential()

	encoder:add(nn.MulConstant(1.0/255.0))
	encoder:add(nn.View(1, 32, 32))
	encoder:add(conv)
	encoder:add(nn.View(256))
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
return get_desc