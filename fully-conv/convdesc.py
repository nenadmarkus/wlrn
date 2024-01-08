# parts based on <https://github.com/magicleap/SuperPointPretrainedNetwork>

import torch
from torch import nn

class Conv(nn.Module):
    def __init__(self, config):
        super(Conv, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, config['ddim'], kernel_size=1, stride=1, padding=0)

    def forward(self, image):
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        cDa = self.relu(self.convDa(x))
        convf = self.convDb(cDa)
        return convf

def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    return descriptors.reshape(b, c, -1)

class DescriptorExtractor(nn.Module):

    default_config = {
        'descriptor_dim': 256,
        'weights': None
    }

    def __init__(self, config={}):
        super().__init__()
        self.config = {**self.default_config, **config}
        self.conv = Conv({
            "idim": 1,
            "ddim": self.config["descriptor_dim"]
        })

        if 'weights' in self.config and self.config['weights'] is not None:
            try:
                self.load_state_dict(torch.load(self.config['weights']))
            except:
                print("* loading self.conv weigths (SuperPoint)")
                self.conv.load_state_dict(torch.load(self.config['weights']))
            print('* load weights from ' + self.config['weights'])

    def forward(self, data):
        #
        if type(data) == list:
            r = []
            for d in data: r.append(self.forward(d))
            return r

        # check input shapes
        assert len(data["image"].shape)==3 or len(data["image"].shape)==1, "image shape has to be 3xHxW or 1xHxW"
        assert len(data["keypoints"].shape)==2, "keypoints shape has to be Nx2"

        # we always work with grayscale input
        if data["image"].shape[0]==3: data["image"] = data["image"][1, :, :].unsqueeze(0)

        # convolutional features
        data["features"] = self.conv(data["image"].unsqueeze(0))

        # sample descriptors
        desc = sample_descriptors(data["keypoints"], data["features"], s=8)

        # l2 normalization
        desc = torch.nn.functional.normalize(desc, p=2, dim=1)

        # remove batch dimension and transpose
        desc = desc[0].t()

        # we're done
        return desc

def test_1():
    import os
    if os.path.exists("superpoint_v1.pth"):
        # pretrained weigth from https://github.com/magicleap/SuperPointPretrainedNetwork
        sk = DescriptorExtractor({"weights": "superpoint_v1.pth"})
    else:
        sk = DescriptorExtractor({"weights": None})

    # batch size, descriptor counts, descriptors dimension
    N = 113

    x = {
        "image": torch.randn(3, 512, 768),
        "keypoints": torch.randn(N, 2),
    }

    import time

    for _ in range(4):
        t = time.time()
        with torch.no_grad(): y = sk(x)
        print("* %d [ms]" % int( 1000 * (time.time() - t) ))

    print(y.dtype, y.shape)
    print( (y**2).sum(dim=1) )

if __name__ == "__main__":
    test_1()