--
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
