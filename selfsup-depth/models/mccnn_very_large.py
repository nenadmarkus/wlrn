import torch
import torch.nn as nn

class SkipConnConvBlock(nn.Module):
    def __init__(self, n):
        self.conv = nn.Sequential(
            nn.Conv2d(n, n, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv.forward(x) + x

#
class McCNN(nn.Module):
    def __init__(self, **config):
        super(McCNN, self).__init__()
        self.features = config.get('features', 64)
        self.ksize = config.get('ksize', 3)
        self.padding = config.get('padding', 1)
        self.unaries = nn.Sequential(
            nn.Conv2d(1, self.features, self.ksize, padding=self.padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.features, self.features, self.ksize, padding=self.padding),
            nn.ReLU(inplace=True),
            SkipConnConvBlock(self.features),
            SkipConnConvBlock(self.features),
            SkipConnConvBlock(self.features),
            SkipConnConvBlock(self.features),
            SkipConnConvBlock(self.features),
            SkipConnConvBlock(self.features),
            SkipConnConvBlock(self.features),
            SkipConnConvBlock(self.features),
            nn.Conv2d(self.features, self.features, self.ksize, padding=self.padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.features, self.features, self.ksize, padding=self.padding),
             nn.ReLU(inplace=True),
            nn.Conv2d(self.features, self.features, self.ksize, padding=self.padding)
        )

    def forward(self, image):
        return self.unaries.forward(image)

#
def init():
	#
	return McCNN()
