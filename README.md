# Learning Local Descriptors from Weakly-Labeled Data

This repo contains the official implementation of the method from the following paper:

```
@misc
{
	wlrn,
	author = {Nenad Marku\v{s} and Igor S. Pand\v{z}i\'c and J\"{o}rgen Ahlberg},
	title = {{Learning Local Descriptors by Optimizing the Keypoint-Correspondence Criterion}},
	year = {2016},
	eprint = {arXiv:1603.09095}
}
```

This work was presented at the International Conference on Pattern Recognition in December 2016.

A newer (updated) version of the paper is available [here](http://hotlab.fer.hr/_download/repository/wlrn.pdf).
We plan to submit it to a journal at some point.

Some basic information about the method is available [here](INFO.md).

## Requirements

To run the code, you will need:

* Torch;
* a CUDA-capable GPU;
* cuDNN;
* OpenCV 3.

OpenCV is required just for the keypoint-extraction process that prepares the training data (files in `utils/`).
The library is not required by the method core which is in `wlrn.lua`.

## Training

Follow these steps.

#### 1. Prepare bags of keypoints

Download <https://nenadmarkus.com/data/ukb.tar> and extract the archive.
It contains two folders with JPG images: `ukb-trn/` and `ukb-val/`.
Images from the first folder will be used for training and images from the second one for checking the validation error.

Move to the folder `utils/` and compile `fast.cpp` and `extp.cpp` with the provided `makefile`.
These are the keypoint detection and patch extraction programs.
Use the script `batch_extract.sh` to transform the downloaded images into bags of keypoints:
```bash
bash batch_extract.sh ukb-trn/ ukb-trn-bags/ 128 32
bash batch_extract.sh ukb-val/ ukb-val-bags/ 128 32
```

Extracted patches should now be in `ukb-trn-bags/` and `ukb-val-bags/`.
As these are stored in the JPG format, you can inspect them with your favorite image viewer.

### 2. Prepare data-loading scripts

To keep a desirable level of abstraction and enable large-scale learning, this code requires the user to provide his/her routines for generating triplets.
An example can be found in `utils/tripletgen.lua`.
The strings "--TRN-FOLDER--", "--TRN-NCHANNELS--", "--TRN-PROBABILITY--", "--VLD-FOLDER--", "--VLD-NCHANNELS--" and "--VLD-PROBABILITY--" need to be replaced with appropriate ones.
The following shell commands will do this for you (replace each slash in the folder paths with backslash+slash as required by `sed`).
```bash
cp utils/tripletgen.lua tripletgen.lua
sed -i -e 's/--TRN-FOLDER--/"ukb-trn-bags"/g' tripletgen.lua
sed -i -e 's/--TRN-NCHANNELS--/3/g' tripletgen.lua
sed -i -e 's/--TRN-PROBABILITY--/0.33/g' tripletgen.lua
sed -i -e 's/--VLD-FOLDER--/"ukb-val-bags"/g' tripletgen.lua
sed -i -e 's/--VLD-NCHANNELS--/3/g' tripletgen.lua
sed -i -e 's/--VLD-PROBABILITY--/1.0/g' tripletgen.lua
```

After executing them, you should find the script `tripletgen.lua` next to `wlrn.lua`.

#### 3. Specify the descriptor-extractor structure

The model is specified with a Lua script which returns a function for constructing the descriptor extraction network.
See the default model in `models/3x32x32_to_64.lua` for an example.

You can generate the default model with `th models/3x32x32_to_64.lua models/net.t7`.

You are encouraged to try different architectures as the default one does not perform very well in all settings.
However, to learn their parameters, some parts of `wlrn.lua` might need additional tweaking, such as learning rates.

#### 4. Start the learning script

Finally, learn the parameters of the network by running the traininig script:

	th wlrn.lua models/net.t7 tripletgen.lua -w models/net-trained.t7

The training should finish in about a day on a GeForce GTX 970 with cuDNN.

## Pretrained models

You can download some pretrained models from <https://nenadmarkus.com/data/wlrn-pretrained.zip>.

## License

MIT.

## Contact

For any additional information contact me at <nenad.markus@fer.hr>.

Copyright (c) 2016, Nenad Markus. All rights reserved.