import torch
import torch.nn as nn
import time
import math
import random

#
#
#

class McCNN(nn.Module):
    def __init__(self, config):
        super(McCNN, self).__init__()
        self.features = config.get('nfeatures', 64)
        self.ksize = 3
        self.padding = 1
        self.inpchn = config.get('inpchn', 1)
        self.unaries = nn.Sequential(
            nn.Conv2d(self.inpchn, self.features, self.ksize, padding=self.padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.features, self.features, self.ksize, padding=self.padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.features, self.features, self.ksize, padding=self.padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.features, self.features, self.ksize, padding=self.padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.features, self.features, self.ksize, padding=self.padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.features, self.features, self.ksize, padding=self.padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.features, self.features, self.ksize, padding=self.padding)
        )

    def forward(self, image):
        return self.unaries.forward(image)

#
#
#

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.0, max_len=1024):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:x.shape[0], :]
		return self.dropout(x)

#
#
#

class StereoModel(nn.Module):
	def __init__(self, D=64):
		super(StereoModel, self).__init__()
		self.D = D
		self.conv = McCNN({"nfeatures": D, "inpchn": 3})

		encoder_layer = nn.TransformerEncoderLayer(d_model=D, nhead=4, dim_feedforward=128, batch_first=True)
		self.selftransf = nn.TransformerEncoder(encoder_layer, num_layers=3)
		self.pe = nn.Identity()#PositionalEncoding(D)
		self.mixtransf = nn.Transformer(d_model=D, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=128, batch_first=True)

	# image1, image2 are Bx3xHxW tensors
	def forward(self, image1, image2, rowinds=None):
		B, _, H, W = image1.shape
		#
		features1 = self.conv(image1).permute( (0, 2, 3, 1) )
		features2 = self.conv(image2).permute( (0, 2, 3, 1) )
		#
		if rowinds is None:
			h = H
		else:
			h = len(rowinds)
			features1 = features1[:, rowinds, :, :]
			features2 = features2[:, rowinds, :, :]
		#
		features1 = features1.reshape( (-1, W, self.D) )
		features2 = features2.reshape( (-1, W, self.D) )
		#
		features1 = self.selftransf(self.pe(features1))
		features2 = self.selftransf(self.pe(features2))
		#
		features1_ = torch.nn.functional.normalize(self.mixtransf(features2, features1), p=2, dim=-1)
		features2_ = torch.nn.functional.normalize(self.mixtransf(features1, features2), p=2, dim=-1)
		#
		return features1_.reshape(B, h, W, self.D), features2_.reshape(B, h, W, self.D)

#
#
#

def test_1():
	H, W = 320, 896
	#H, W = 192, 384
	model = StereoModel().cuda()
	for i in range(0, 16):
		t = time.time()
		with torch.no_grad():
			i1 = torch.randn(1, 3, H, W).cuda()
			i2 = torch.randn(1, 3, H, W).cuda()
			f1, f2 = model.forward(i1, i2)
		print("* elapsed time: %d [ms]" % int(1000 * (time.time() - t)))
	print(f1.shape, f2.shape)

def test_2():
	H, W = 320, 896
	n = 32
	model = StereoModel().cuda()
	for i in range(0, 16):
		t = time.time()
		i1 = torch.randn(1, 3, H, W).cuda()
		i2 = torch.randn(1, 3, H, W).cuda()
		rowinds = [random.randint(32, H-32) for i in range(0, n)]
		f1, f2 = model.forward(i1, i2, rowinds=rowinds)
		(f1 + f2).sum().backward()
		print("* elapsed time: %d [ms]" % int(1000 * (time.time() - t)))
	print(f1.shape, f2.shape)

if __name__ == "__main__":
	test_2()
