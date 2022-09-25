import torch
import torch.nn as nn

#
class McCNN(nn.Module):
    def __init__(self, config):
        super(McCNN, self).__init__()
        self.features = config.get('features', 64)
        self.ksize = config.get('ksize', 3)
        self.padding = config.get('padding', 1)
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
def init(c=1):
	#
	return McCNN({
	    "inpchn": c,
	    "features": 64,
	    "ksize": 3,
	    "padding": 1
	})
