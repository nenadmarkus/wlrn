# Learning Local Convolutional Descriptors from Weakly-Labeled Data

Current best local descriptors are learned on a large dataset of matching and non-matching keypoint pairs.
However, data of this kind is not always available since detailed keypoint correspondences can be hard to establish (e.g., for non-image data).
On the other hand, we can often obtain labels for pairs of keypoint bags.
For example, keypoint bags extracted from two images of the same object under different views form a matching pair, and keypoint bags extracted from images of different objects form a non-matching pair.
On average, matching pairs should contain more corresponding keypoints than non-matching pairs.
We propose to learn local descriptors from such information where local correspondences are not known in advance.

<center><img src="teaser.png" alt="Teaser" style="width: 512px;"/></center>

Each image in the dataset (first row) is processed with a keypoint detector (second row) and transformed into a bag of visual words (third row).
Some bags form matching pairs (green arrow) and some form non-matching pairs (red arrows).
On average, matching pairs should contain more corresponding local visual words than non-matching pairs.
We propose to *learn local descriptors* by optimizing the mentioned local correspondence criterion on a given dataset.
Note that prior work assumes local correspondences are known in advance.

We will add a technical report with more details soon.

## Some results (to be updated soon)

A network trained with our method (code in this repo) can be obtained from the folder `models/`.
This network extracts `64f` descriptors of unit length from local grayscale patches of size `32x32`.
Here is its structure:

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

The net parameters are stored as a vector of floats at `models/32x32_to_64.params`.
This is to reduce the storage requirements (i.e., the repo size).
Use the following code to deploy and use the net.

```Lua
-- load the network parameters first
params = torch.load('models/32x32_to_64.params')

-- create the network and initialize its weights with loaded data
require 'models/nets'
n = get_32x32_to_64(params)

-- generate a random batch of five 32x32 patches (each pixel is a float from [0, 1])
p = torch.rand(5, 32, 32):float()

-- propagate the batch through the net to obtain descriptors
-- (note that no patch prepocessing is required (such as mean substraction))
d = n:forward(p)

-- an appropriate similarity between descriptors is, for example, a dot product ...
print(d[1]*d[2])

-- ... or you can use the Euclidean distance
print(torch.norm(d[1] - d[2]))
```

## How to repeat the training

Follow these steps.

#### 1. Prepare bags of keypoints

Download <http://46.101.250.137/data/ukb.tar> and extract the archive.
It contains two folders with JPG images: `ukb-trn/` and `ukb-val/`.
Images from the first folder will be used for training and images from the second one for checking the validation error.

Move to the folder `utils/` and compile `fast.cpp` and `extp.cpp` with the provided `makefile`.
These are the keypoint detection and patch extraction programs.
Use the script `batch_extract.sh` to transform the downloaded images into bags of keypoints:
```
bash batch_extract.sh ukb-trn/ ukb-trn-bags/
bash batch_extract.sh ukb-val/ ukb-val-bags/
```

To complete the data preparation procedure, run the script `bagio.lua` to store the generated bags in Torch7 format:
```
th bagio ukb-trn-bags/ ukb-trn.t7
th bagio ukb-val-bags/ ukb-val.t7
```

#### 2. Specify the descriptor extractor structure

The default models are available in `models/nets.lua`.
Enter the following code in the Torch7 terminal:

```Lua
require 'models/nets'
n = get_32x32_to_64():float()
torch.save('net.t7', n)
```

Of course, you can try different networks.
However, to learn their parameters, some parts of the learning script might need additional tweaking (such as learning rates).

#### 3. Start the learning script

Finally, learn the parameters of the network by running the traininig script:

	th wlrn.lua net.t7 ukb-trn.t7 -v ukb-val.t7 -w params.t7

The training should finish in about a day on a GeForce GTX 970 with cuDNN.
The file `params.t7` contains the learned parameters of `net.t7`.
Use the following code to deploy them:
```Lua
n = torch.load('net.t7'):float()
p = n:getParameters()
p:copy(torch.load('params.t7'))
torch.save('net.t7', n)
```

## Contact

For any additional information contact me at <nenad.markus@fer.hr>.

Copyright (c) 2016, Nenad Markus. All rights reserved.