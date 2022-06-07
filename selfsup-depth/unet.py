import torch
import torch.nn as nn

class UNet(nn.Module):
	def __init__(self, n_inchn, n_outchn, depth=5, wf=16, growthtype='pow2'):
		super(UNet, self).__init__()
		self.inconv = make_conv3x3_block(n_inchn, wf)
		#
		inp = wf
		down = []
		for i in range(0, depth):
			if growthtype == 'pow2':
				otp = 2*inp
			else:
				otp = inp + wf
			down.append(
				make_down_layer(inp, otp)
			)
			inp = otp
		self.down = nn.ModuleList(down)
		#
		up = []
		for i in range(0, depth):
			if growthtype == 'pow2':
				otp = inp//2
			else:
				otp = inp - wf
			up.append(
				make_up_layer(inp, otp)
			)
			inp = otp
		self.up = nn.ModuleList(up)
		#
		self.outconv = make_conv3x3_block(wf, n_outchn)

	def forward(self, x):
		#
		x = self.inconv(x)
		#
		cache = []
		for down in self.down:
			cache.append(x)
			x = down.forward(x)
		#
		for up in self.up:
			x = up.forward(x, cache.pop())
		#
		return self.outconv(x)

#
#
#

def make_conv3x3_block(inp, otp, stride=1):
	return nn.Sequential(
		nn.Conv2d(inp, otp, (3, 3), stride=stride, padding=1, bias=False),
		#nn.BatchNorm2d(otp),
		#nn.GroupNorm(1, otp),
		nn.Hardtanh(min_val=0, max_val=1)
	)

def make_down_layer(inp, otp):
	#
	return nn.Sequential(
		make_conv3x3_block(inp, otp, stride=2),
		make_conv3x3_block(otp, otp)
	)

def make_up_layer(inp, otp):
	return up2x(inp, otp)

class up2x(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(up2x, self).__init__()
		self.conv = make_conv3x3_block(in_ch, out_ch)

	def forward(self, x_low, x_high=None):
		#
		x = torch.nn.functional.interpolate(self.conv(x_low), scale_factor=2, mode='bilinear')
		if x_high is not None:
			return torch.nn.functional.hardtanh(
				x + x_high,
				min_val=0.0,
				max_val=1.0
			)
		else:
			return x

#
#
#

def export_to_onnx(model, savepath):
	dst = "model.onnx"
	model.eval()

	x = torch.randn(1, 1, 512, 512, requires_grad=True)
	o = model.forward(x)

	torch.onnx.export(model,  # model being run
		x,  # model input (or a tuple for multiple inputs)
		savepath,  # where to save the model (can be a file or file-like object)
		export_params=True,  # store the trained parameter weights inside the model file
		opset_version=12,  # the ONNX version to export the model to
		do_constant_folding=True,  # whether to execute constant folding for optimization
		input_names=["inputs"],  # the model's input names
		output_names=["outputs"],  # the model's output names
		dynamic_axes={"inputs": {0: "batch_size", 2: "height", 3: "width"},  # variable length axes
				"outputs": {0: "batch_size", 2: "height", 3: "width"}})

def test_1():
	unet = init()
	x = torch.randn(2, 1, 512, 1024)
	y = unet.forward(x)
	print(y.shape)
	# test backward pass
	y.sum().backward()

def test_2():
	unet = init()
	x = torch.randn(1, 1, 512, 1024)
	with torch.no_grad():
		y = unet.forward(x)
	import time
	t = time.time()
	with torch.no_grad():
		y = unet.forward(x)
	print('* elapsed time: %d [ms]' % int(1000*(time.time() - t)))

def test_3():
	unet = init()
	export_to_onnx(unet, "unet.onnx")

def init():
	return UNet(1, 32, depth=3, wf=16, growthtype="linear")

#if __name__ == "__main__":
#	test_3()
