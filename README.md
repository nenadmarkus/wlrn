# Learning Local Convolutional Descriptors from Weakly-Labeled Data

Current best local descriptors are learned on a large dataset of matching and non-matching keypoint pairs.
However, data of this kind is not always available since detailed keypoint correspondences can be hard to establish (e.g., for non-image data).
On the other hand, we can often obtain labels for pairs of keypoint bags.
For example, keypoint bags extracted from two images of the same object under different views form a matching pair, and keypoint bags extracted from images of different objects form a non-matching pair.
On average, matching pairs should contain more corresponding keypoints than non-matching pairs.
We propose to learn local descriptors from such information where local correspondences are not known in advance.

## The method

<center><img src="teaser.png" alt="Teaser" style="width: 512px;"/></center>

Each image in the dataset (first row) is processed with a keypoint detector (second row) and transformed into a bag of visual words (third row).
Some bags form matching pairs (green arrow) and some form non-matching pairs (red arrows).
On average, matching pairs should contain more corresponding local visual words than non-matching pairs.
We propose to *learn local descriptors* by optimizing the mentioned local correspondence criterion on a given dataset.
Note that prior work assumes local correspondences are known in advance.

We will add a technical report with more details soon.

## Some results (to be updated soon)

A network trained with our method (code in this repo) can be obtained from `nn/32x32_to_64.net` (in Torch7 format).
This network extracts `64f` descriptors of unit length from local grayscale patches of size `32x32`.

```
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> output]
  (1): nn.View
  (2): nn.SpatialConvolution(1 -> 32, 3x3)
  (3): nn.ReLU
  (4): nn.SpatialConvolution(32 -> 64, 4x4, 2,2)
  (5): nn.ReLU
  (6): nn.SpatialConvolution(64 -> 128, 3x3)
  (7): nn.SpatialMaxPooling(2,2,2,2)
  (8): nn.SpatialConvolution(128 -> 32, 1x1)
  (9): nn.View
  (10): nn.Linear(1152 -> 64)
  (11): nn.Normalize(2)
}
```

### How to use

```Lua
-- load the network first
n = torch.load('nn/32x32_to_64.net')

-- generate a random batch of five 32x32 patches (each pixel should be represented as a float from [0, 1))
p = torch.rand(5, 32, 32):float()

-- propagate the batch through the net and print results
-- (note that the net does not require any patch prepocessing (such as mean substraction) prior to descriptor extraction)
print(n:forward(p))
```

The net parameters are stored as floats to reduce the storage requirements (i.e., the repo size).

### How to repeat the training

First, download the training and validation datasets in Torch7 format from <http://46.101.250.137/data/>.

Next, run the traininig script:

	th wlrn.lua 32x32.ukb-trn.t7 -v 32x32.ukb-val.t7 -w params.t7 -n 128

The training should finish in about 3 days on a GeForce GTX 970.
The file `params.t7` contains the learned parameters of the network.
Use the following code to deploy them:
```Lua
require 'models'
n = get_32x32_to_64():float()
p = n:getParameters()
p:copy(torch.load('params.t7'))
torch.save('nn/32x32_to_64.new.net', n)
```
## Contact

For any additional information contact me at <nenad.markus@fer.hr>.

Copyright (c) 2016, Nenad Markus. All rights reserved.