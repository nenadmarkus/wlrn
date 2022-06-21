import torch
import torch.nn as nn

#
class McCNN(nn.Module):
    def __init__(self, **config):
        super(McCNN, self).__init__()
        self.features = config.get('features', 64)
        self.unaries = nn.Sequential(
            nn.Conv2d(1, self.features,3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.features, self.features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.features, self.features, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.features, self.features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.features, self.features, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.features, self.features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.features, self.features, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.features, self.features, 3, padding=1)
        )

    def forward(self, image):
        return self.unaries.forward(image)

#
def init():
	#
	return McCNN()
