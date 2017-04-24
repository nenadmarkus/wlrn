import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

	def __init__(self):
		#
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, 3, 1, 0)
		self.conv2 = nn.Conv2d(32, 64, 4, 2, 0)
		self.conv3 = nn.Conv2d(64, 128, 3, 1, 0)
		self.mpool = nn.MaxPool2d(2, 2)
		self.conv4 = nn.Conv2d(128, 32, 1, 1, 0)
		self.conv5 = nn.Conv2d(32, 128, 6, 1, 0)

	def forward(self, input):
		#
		r = input.div(255)
		r = F.relu(self.conv1(r.view(-1, 3, 32, 32)))
		r = F.relu(self.conv2(r))
		r = self.mpool(self.conv3(r))
		r = self.conv4(r)
		r = self.conv5(r).view(-1, 128)

		# L2 normalization
		n = torch.norm(r, 2, 1).view(-1, 1).expand(r.size())
		r = torch.div(r, n)

		#
		return r

#
def init():
	return Net()